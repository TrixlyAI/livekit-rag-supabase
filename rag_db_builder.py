import uuid
import logging
from pathlib import Path
from typing import List, Optional, Union, Any, Dict
from dataclasses import dataclass
import asyncio
import aiohttp
from tqdm import tqdm
from supabase import create_client, Client
from livekit.agents import tokenize
from livekit.plugins import openai
import os
from dotenv import load_dotenv

logger = logging.getLogger("supabase-rag-builder")

@dataclass
class QueryResult:
    id: str
    content: str
    metadata: Dict[str, Any]
    similarity: float

@dataclass
class Item:
    id: str
    content: str
    embedding: List[float]
    metadata: Dict[str, Any]

class SupabaseRAG:
    """
    RAG system using Supabase vector database with pgvector extension.
    """
    
    def __init__(
        self,
        supabase_url: str,
        supabase_key: str,
        table_name: str = "supabase_vector",
        embeddings_dimension: int = 1536,
    ):
        """
        Initialize Supabase RAG system.
        
        Args:
            supabase_url: Your Supabase project URL
            supabase_key: Your Supabase API key
            table_name: Name of the table to store documents
            embeddings_dimension: Dimension of embeddings
        """
        self.supabase: Client = create_client(supabase_url, supabase_key)
        self.table_name = table_name
        self.embeddings_dimension = embeddings_dimension
        print(f"✅ Initialized RAG system with table: {table_name}")
        
    def test_connection(self) -> bool:
        """Test if the connection and table are working."""
        try:
            # Try a simple query
            result = self.supabase.table(self.table_name).select('*').limit(1).execute()
            return True
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False

    def add_item(self, content: str, embedding: List[float], metadata: Dict[str, Any] = None) -> str:
        """Add a single item to the database."""
        if metadata is None:
            metadata = {}
        
        data = {
            'content': content,
            'embedding': embedding,
            'metadata': metadata
        }
        
        try:
            result = self.supabase.table(self.table_name).insert(data).execute()
            if result.data:
                return str(result.data[0]['id'])
            return ""
        except Exception as e:
            logger.error(f"Failed to add item: {e}")
            raise

    def add_items_batch(self, items: List[Item]) -> List[str]:
        """Add multiple items to the database in batch."""
        data = []
        
        for item in items:
            # Don't include ID - let the database auto-generate it
            data.append({
                'content': item.content,
                'embedding': item.embedding,
                'metadata': item.metadata or {}
            })
        
        try:
            result = self.supabase.table(self.table_name).insert(data).execute()
            if result.data:
                return [str(item['id']) for item in result.data]
            return []
        except Exception as e:
            logger.error(f"Failed to add batch items: {e}")
            raise

    def query(
        self, 
        embedding: List[float], 
        limit: int = 10,
        similarity_threshold: float = 0.7,
        metadata_filter: Dict[str, Any] = None
    ) -> List[QueryResult]:
        """
        Query the database for similar documents.
        
        Args:
            embedding: Query embedding vector
            limit: Maximum number of results
            similarity_threshold: Minimum similarity score
            metadata_filter: Optional metadata filters
        """
        try:
            # Convert embedding to string format for SQL
            embedding_str = f"[{','.join(map(str, embedding))}]"
            
            # Build the query - using proper vector similarity syntax
            query = self.supabase.table(self.table_name).select(
                f"id, content, metadata, embedding <-> '{embedding_str}' as distance"
            )
            
            # Add similarity threshold filter (convert similarity to distance)
            distance_threshold = 1 - similarity_threshold
            query = query.lt('distance', distance_threshold)
            
            # Add metadata filters if provided
            if metadata_filter:
                for key, value in metadata_filter.items():
                    query = query.eq(f'metadata->{key}', value)
            
            # Execute query with limit and ordering
            result = query.order('distance').limit(limit).execute()
            
            return [
                QueryResult(
                    id=item['id'],
                    content=item['content'],
                    metadata=item['metadata'] or {},
                    similarity=1 - float(item['distance'])  # Convert distance back to similarity
                )
                for item in result.data
            ]
            
        except Exception as e:
            logger.warning(f"Vector search failed, falling back to simple search: {e}")
            # Fallback: return recent documents if vector search fails
            try:
                result = self.supabase.table(self.table_name).select(
                    "id, content, metadata"
                ).order('created_at', desc=True).limit(limit).execute()
                
                return [
                    QueryResult(
                        id=item['id'],
                        content=item['content'],
                        metadata=item['metadata'] or {},
                        similarity=0.5  # Default similarity for fallback
                    )
                    for item in result.data
                ]
            except Exception as e2:
                logger.error(f"Fallback query also failed: {e2}")
                return []

    def delete_item(self, item_id: str) -> bool:
        """Delete an item by ID."""
        try:
            result = self.supabase.table(self.table_name).delete().eq('id', item_id).execute()
            return len(result.data) > 0
        except Exception as e:
            logger.error(f"Failed to delete item: {e}")
            return False

    def clear_all(self) -> None:
        """Clear all documents from the table."""
        try:
            self.supabase.table(self.table_name).delete().neq('id', '').execute()
            logger.info(f"Cleared all documents from {self.table_name}")
        except Exception as e:
            logger.error(f"Failed to clear table: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        try:
            result = self.supabase.table(self.table_name).select('id', count='exact').execute()
            return {
                'total_documents': result.count or 0,
                'table_name': self.table_name,
                'embedding_dimension': self.embeddings_dimension
            }
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {
                'total_documents': 0,
                'table_name': self.table_name,
                'embedding_dimension': self.embeddings_dimension,
                'error': str(e)
            }

class SentenceChunker:
    """Text chunker for breaking down large texts."""
    
    def __init__(
        self,
        max_chunk_size: int = 500,
        chunk_overlap: int = 50,
    ):
        self.max_chunk_size = max_chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(self, text: str) -> List[str]:
        """Simple chunking based on sentences and character limits."""
        # Split into sentences
        sentences = text.replace('\n', ' ').split('. ')
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            # Add period back if it was removed
            if not sentence.endswith('.'):
                sentence += '.'
            
            # Check if adding this sentence would exceed the limit
            if len(current_chunk) + len(sentence) + 1 > self.max_chunk_size:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    
                    # Start new chunk with overlap
                    if self.chunk_overlap > 0:
                        words = current_chunk.split()
                        overlap_words = words[-self.chunk_overlap:]
                        current_chunk = ' '.join(overlap_words) + ' ' + sentence
                    else:
                        current_chunk = sentence
                else:
                    # Single sentence is too long, just add it
                    chunks.append(sentence)
                    current_chunk = ""
            else:
                if current_chunk:
                    current_chunk += ' ' + sentence
                else:
                    current_chunk = sentence
        
        # Add the last chunk
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks

class SupabaseRAGBuilder:
    """
    Builder for creating RAG databases using Supabase.
    """
    
    def __init__(
        self,
        supabase_url: str,
        supabase_key: str,
        table_name: str = "supabase_vector",
        embeddings_dimension: int = 1536,
        embeddings_model: str = "text-embedding-3-small",
        chunker: Optional[SentenceChunker] = None,
    ):
        """
        Initialize the Supabase RAG builder.
        
        Args:
            supabase_url: Your Supabase project URL
            supabase_key: Your Supabase API key
            table_name: Name of the table to store documents
            embeddings_dimension: Dimension of embeddings
            embeddings_model: OpenAI model for embeddings
            chunker: Text chunker instance
        """
        self.rag = SupabaseRAG(supabase_url, supabase_key, table_name, embeddings_dimension)
        self.embeddings_model = embeddings_model
        self.embeddings_dimension = embeddings_dimension
        self.chunker = chunker or SentenceChunker()

    def test_connection(self) -> bool:
        """Test if everything is working."""
        return self.rag.test_connection()

    def _clean_content(self, text: str) -> str:
        """Clean content by removing navigation and UI elements."""
        lines = text.split('\n')
        cleaned_lines = []
        
        skip_patterns = [
            'Docs', 'Search', 'GitHub', 'Slack', 'Sign in',
            'Home', 'AI Agents', 'Telephony', 'Recipes', 'Reference',
            'On this page', 'Get started with LiveKit today',
            'Content from https://docs.livekit.io/'
        ]
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            if any(pattern in line for pattern in skip_patterns):
                continue
                
            if line.startswith('http') or line.startswith('[') or line.endswith(']'):
                continue
                
            cleaned_lines.append(line)
            
        return '\n'.join(cleaned_lines)

    async def _create_embeddings(
        self, 
        text: str, 
        http_session: Optional[aiohttp.ClientSession] = None
    ) -> openai.EmbeddingData:
        """Create embeddings for text."""
        try:
            results = await openai.create_embeddings(
                input=[text],
                model=self.embeddings_model,
                dimensions=self.embeddings_dimension,
                http_session=http_session,
            )
            return results[0]
        except Exception as e:
            logger.error(f"Failed to create embeddings: {e}")
            raise

    async def build_from_texts(
        self,
        texts: List[str],
        metadata_list: Optional[List[Dict[str, Any]]] = None,
        show_progress: bool = True,
        batch_size: int = 50,
    ) -> None:
        """
        Build RAG database from list of texts.
        
        Args:
            texts: List of text strings
            metadata_list: Optional metadata for each text
            show_progress: Show progress bar
            batch_size: Number of items to process in each batch
        """
        if metadata_list and len(metadata_list) != len(texts):
            raise ValueError("metadata_list must have same length as texts")
        
        async with aiohttp.ClientSession() as http_session:
            # Clean texts
            cleaned_texts = []
            for i, text in enumerate(texts):
                cleaned = self._clean_content(text)
                if cleaned:
                    metadata = metadata_list[i] if metadata_list else {}
                    cleaned_texts.append((cleaned, metadata))
            
            print(f"📝 Processing {len(cleaned_texts)} texts...")
            
            # Process in batches
            items_to_insert = []
            
            # Create iterator with optional progress bar
            iterator = cleaned_texts
            if show_progress:
                iterator = tqdm(cleaned_texts, desc="Creating embeddings")
            
            for text, metadata in iterator:
                try:
                    # Generate embedding
                    resp = await self._create_embeddings(text, http_session)
                    
                    # Create item (don't generate UUID, let database handle ID)
                    item = Item(
                        id="",  # Empty string - database will generate
                        content=text,
                        embedding=resp.embedding,
                        metadata=metadata
                    )
                    items_to_insert.append(item)
                    
                    # Insert batch when we reach batch_size
                    if len(items_to_insert) >= batch_size:
                        self.rag.add_items_batch(items_to_insert)
                        items_to_insert = []
                        
                except Exception as e:
                    logger.error(f"Failed to process text: {e}")
                    continue
            
            # Insert remaining items
            if items_to_insert:
                self.rag.add_items_batch(items_to_insert)
            
            logger.info(f"Successfully added {len(cleaned_texts)} documents to database")

    async def build_from_file(
        self,
        file_path: Union[str, Path],
        metadata: Dict[str, Any] = None,
        show_progress: bool = True,
        chunk_text: bool = True,
    ) -> None:
        """
        Build RAG database from text file.
        
        Args:
            file_path: Path to text file
            metadata: Metadata to attach to all chunks
            show_progress: Show progress bar
            chunk_text: Whether to chunk the text
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        print(f"📖 Reading file: {file_path}")
        
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        if chunk_text:
            chunks = self.chunker.chunk(content)
            print(f"📄 Split into {len(chunks)} chunks")
        else:
            chunks = [content]  # Keep as single document
        
        # Add file metadata
        base_metadata = metadata or {}
        base_metadata.update({
            'source_file': str(file_path),
            'file_name': file_path.name
        })
        
        metadata_list = [base_metadata.copy() for _ in chunks]
        
        await self.build_from_texts(chunks, metadata_list, show_progress)

    async def query(
        self,
        query_text: str,
        limit: int = 5,
        similarity_threshold: float = 0.7,
        metadata_filter: Dict[str, Any] = None
    ) -> List[QueryResult]:
        """Query the RAG database with a text query."""
        try:
            # Generate embedding for the query
            async with aiohttp.ClientSession() as http_session:
                query_embedding = await self._create_embeddings(query_text, http_session)
            
            # Search for similar documents
            return self.rag.query(
                embedding=query_embedding.embedding,
                limit=limit,
                similarity_threshold=similarity_threshold,
                metadata_filter=metadata_filter
            )
        except Exception as e:
            logger.error(f"Query failed: {e}")
            return []

    @classmethod
    async def create_from_file(
        cls,
        file_path: Union[str, Path],
        supabase_url: str,
        supabase_key: str,
        table_name: str = "supabase_vector",
        **kwargs,
    ) -> "SupabaseRAGBuilder":
        """
        Create and build RAG database from file in one step.
        """
        builder = cls(supabase_url, supabase_key, table_name, **kwargs)
        await builder.build_from_file(file_path)
        return builder

# Example usage and testing
async def test_rag_system():
    """Test the RAG system with sample data."""
    
    load_dotenv()
    
    # Initialize builder
    builder = SupabaseRAGBuilder(
        supabase_url=os.getenv("SUPABASE_URL"),
        supabase_key=os.getenv("SUPABASE_ANON_KEY"),
        table_name="supabase_vector"  # Change to "supabase_vector" if you want to use your existing table
    )
    
    # Test connection
    if not builder.test_connection():
        print("❌ Connection test failed!")
        return
    
    print("✅ Connection successful!")
    
    # Add some sample documents
    sample_texts = [
        "Python is a high-level programming language known for its simplicity and readability. It's widely used in web development, data science, and artificial intelligence.",
        "Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed.",
        "Supabase is an open-source alternative to Firebase that provides a PostgreSQL database with real-time capabilities and built-in authentication.",
        "Vector databases are specialized databases designed to store and query high-dimensional vector data, commonly used in AI and machine learning applications.",
        "Natural language processing (NLP) is a branch of AI that helps computers understand, interpret, and manipulate human language."
    ]
    
    metadata_list = [
        {"topic": "programming", "language": "python"},
        {"topic": "ai", "category": "machine_learning"},
        {"topic": "database", "category": "backend"},
        {"topic": "database", "category": "vector_db"},
        {"topic": "ai", "category": "nlp"}
    ]
    
    print("📝 Adding sample documents to RAG database...")
    await builder.build_from_texts(sample_texts, metadata_list)
    
    # Test queries
    test_queries = [
        "What is Python?",
        "Tell me about machine learning",
        "What is a vector database?",
        "How does Supabase work?"
    ]
    
    print("\n🔍 Testing queries...")
    
    for query in test_queries:
        print(f"\n📋 Query: '{query}'")
        results = await builder.query(query, limit=3)
        
        if results:
            for i, result in enumerate(results, 1):
                print(f"  {i}. Similarity: {result.similarity:.3f}")
                print(f"     Content: {result.content[:100]}...")
                print(f"     Metadata: {result.metadata}")
        else:
            print("  No results found")
    
    # Get stats
    stats = builder.rag.get_stats()
    print(f"\n📊 Database stats: {stats}")

async def build_from_file_example():
    """Example of building RAG from a file."""
    
    load_dotenv()
    
    # Create a sample file for testing
    sample_content = """
    Artificial Intelligence and Machine Learning

    Artificial Intelligence (AI) is a broad field of computer science focused on creating systems capable of performing tasks that typically require human intelligence. These tasks include learning, reasoning, problem-solving, perception, and language understanding.

    Machine Learning (ML) is a subset of AI that focuses on the development of algorithms and statistical models that enable computers to improve their performance on a specific task through experience, without being explicitly programmed for every scenario.

    Deep Learning is a subset of machine learning that uses neural networks with multiple layers (hence "deep") to model and understand complex patterns in data. It has been particularly successful in areas like image recognition, natural language processing, and game playing.

    Natural Language Processing (NLP) is a branch of AI that deals with the interaction between computers and human language. It involves enabling computers to understand, interpret, and generate human language in a valuable way.

    Computer Vision is another important area of AI that focuses on enabling computers to interpret and understand visual information from the world, including images and videos.
    """
    
    # Save sample content to file
    sample_file = Path("data/raw_data.txt")
    sample_file.parent.mkdir(parents=True, exist_ok=True)
    with open(sample_file, "w", encoding="utf-8") as f:
        f.write(sample_content)
    
    print(f"📄 Created sample file: {sample_file}")
    
    # Build RAG from file
    builder = SupabaseRAGBuilder(
        supabase_url=os.getenv("SUPABASE_URL"),
        supabase_key=os.getenv("SUPABASE_ANON_KEY"),
        table_name="supabase_vector",  # Change to your table name
    )
    
    await builder.build_from_file(sample_file, metadata={"source": "ai_guide", "topic": "artificial_intelligence"})
    
    # Test query
    results = await builder.query("What is deep learning?", limit=2)
    
    print("\n🔍 Query results for 'What is deep learning?':")
    for i, result in enumerate(results, 1):
        print(f"  {i}. Similarity: {result.similarity:.3f}")
        print(f"     Content: {result.content[:200]}...")
    
    # Clean up
    sample_file.unlink()
    print(f"🧹 Cleaned up sample file")

if __name__ == "__main__":
    print("🚀 Testing Supabase RAG System")
    print("=" * 50)
    
    # Run the test
    asyncio.run(test_rag_system())
    
    print("\n" + "=" * 50)
    print("📄 Testing file-based RAG building")
    
    # Test file-based building
    asyncio.run(build_from_file_example())
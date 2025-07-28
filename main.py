#!/usr/bin/env python3
"""
RAG-enabled voice agent for LiveKit Agents 1.0 with Supabase backend and timing measurements
"""
import logging
import os
import time
import uuid
from pathlib import Path
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from contextlib import contextmanager
from dotenv import load_dotenv
import aiohttp
from supabase import create_client, Client
from livekit.agents import (
    JobContext,
    WorkerOptions,
    cli,
    RunContext,
    function_tool,
    RoomInputOptions,
    Agent,
    AgentSession,
    AutoSubscribe,
)
from livekit.plugins import openai, silero, deepgram, hume

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("supabase-rag-agent")

# Timing utilities
@contextmanager
def timing_context(operation_name: str, log_level=logging.INFO):
    """Context manager for timing operations"""
    start_time = time.perf_counter()
    logger.log(log_level, f"Starting {operation_name}...")
    try:
        yield
    finally:
        end_time = time.perf_counter()
        duration = end_time - start_time
        logger.log(log_level, f"✓ {operation_name} completed in {duration:.3f}s")

def time_function(func_name: str = None):
    """Decorator for timing function calls"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            name = func_name or f"{func.__name__}"
            with timing_context(name):
                return func(*args, **kwargs)
        return wrapper
    return decorator

# Supabase RAG Classes
@dataclass
class QueryResult:
    id: str
    content: str
    metadata: Dict[str, Any]
    similarity: float

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
        """Initialize Supabase RAG system."""
        self.supabase: Client = create_client(supabase_url, supabase_key)
        self.table_name = table_name
        self.embeddings_dimension = embeddings_dimension
        self._seen_results = set()  # Track seen results to avoid repetition
        logger.info(f"✅ Initialized Supabase RAG with table: {table_name}")
        
    def test_connection(self) -> bool:
        """Test if the connection and table are working."""
        try:
            result = self.supabase.table(self.table_name).select('*').limit(1).execute()
            return True
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False

    def query(
        self, 
        embedding: List[float], 
        limit: int = 10,
        similarity_threshold: float = 0.7,
        metadata_filter: Dict[str, Any] = None
    ) -> List[QueryResult]:
        """
        Query the database for similar documents.
        """
        try:
            with timing_context(f"Supabase vector search (limit={limit})"):
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
                        id=str(item['id']),
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
                with timing_context("Fallback: simple content search"):
                    result = self.supabase.table(self.table_name).select(
                        "id, content, metadata"
                    ).order('created_at', desc=True).limit(limit).execute()
                    
                    return [
                        QueryResult(
                            id=str(item['id']),
                            content=item['content'],
                            metadata=item['metadata'] or {},
                            similarity=0.5  # Default similarity for fallback
                        )
                        for item in result.data
                    ]
            except Exception as e2:
                logger.error(f"Fallback query also failed: {e2}")
                return []

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

class RAGEnrichedAgent(Agent):
    """
    A simplified agent that can answer questions using Supabase RAG.
    """
    
    def __init__(self) -> None:
        """Initialize the RAG-enabled agent."""
        with timing_context("RAG Agent initialization"):
            super().__init__(
                instructions="""
You are a helpful voice assistant with access to a comprehensive knowledge database.
You can answer questions using your knowledge base and provide detailed, accurate information.
Your responses should always be concise and suitable for text-to-speech output.
Provide natural backchanneling responses like "mm-hmm", "I see", "right", and "okay" 
during appropriate pauses to show active listening. Keep backchannels under 3 words.

When you receive information from the knowledge base, integrate it naturally into your response
without explicitly mentioning that you're using a database or RAG system.
""",
            )
            
            # Initialize Supabase RAG components
            self._initialize_supabase_rag()

    def _initialize_supabase_rag(self):
        """Initialize Supabase RAG components with timing"""
        with timing_context("Supabase RAG initialization"):
            # Get Supabase credentials from environment
            supabase_url = os.getenv("SUPABASE_URL")
            supabase_key = os.getenv("SUPABASE_ANON_KEY")
            
            if not supabase_url or not supabase_key:
                logger.error("Supabase credentials not found in environment variables")
                self._rag_available = False
                return

            try:
                # Initialize Supabase RAG
                self._supabase_rag = SupabaseRAG(
                    supabase_url=supabase_url,
                    supabase_key=supabase_key,
                    table_name=os.getenv("SUPABASE_TABLE", "supabase_vector"),
                    embeddings_dimension=1536
                )
                
                # Test connection
                with timing_context("Testing Supabase connection"):
                    if self._supabase_rag.test_connection():
                        self._rag_available = True
                        
                        # Get database stats
                        stats = self._supabase_rag.get_stats()
                        logger.info(f"✅ Supabase RAG connected successfully")
                        logger.info(f"📊 Database stats: {stats}")
                    else:
                        logger.error("Supabase connection test failed")
                        self._rag_available = False
                
                # Configuration
                self._embeddings_model = "text-embedding-3-small"
                self._embeddings_dimension = 1536
                
            except Exception as e:
                logger.error(f"Failed to initialize Supabase RAG: {e}")
                self._rag_available = False

    @function_tool
    async def knowledge_search(self, context: RunContext, query: str):
        """Search the knowledge database for relevant information."""
        overall_start = time.perf_counter()
        logger.info(f"🔍 Knowledge search query: '{query}'")
        
        if not self._rag_available:
            logger.warning("Knowledge database is not available")
            return "Knowledge database is not available."
            
        try:
            # Generate embeddings for the query
            embedding_start = time.perf_counter()
            logger.info("Generating query embeddings...")
            
            query_embedding = await openai.create_embeddings(
                input=[query],
                model=self._embeddings_model,
                dimensions=self._embeddings_dimension,
            )
            
            embedding_end = time.perf_counter()
            embedding_time = embedding_end - embedding_start
            logger.info(f"✓ Query embeddings generated in {embedding_time:.3f}s")
            
            # Query the Supabase database
            search_start = time.perf_counter()
            logger.info("Searching Supabase vector database...")
            
            all_results = self._supabase_rag.query(
                embedding=query_embedding[0].embedding,
                limit=5,
                similarity_threshold=0.6
            )
            
            search_end = time.perf_counter()
            search_time = search_end - search_start
            logger.info(f"✓ Vector search completed in {search_time:.3f}s, found {len(all_results)} results")
            
            # Filter out previously seen results to provide variety
            filter_start = time.perf_counter()
            logger.info("Filtering results for variety...")
            
            new_results = [
                r for r in all_results if r.id not in self._supabase_rag._seen_results
            ]
            
            # If no new results, reset seen results and use all
            if not new_results and all_results:
                logger.info("No new results, resetting seen results cache")
                self._supabase_rag._seen_results.clear()
                new_results = all_results
            
            filter_end = time.perf_counter()
            filter_time = filter_end - filter_start
            logger.info(f"✓ Result filtering completed in {filter_time:.3f}s, {len(new_results)} results available")
            
            if not new_results:
                logger.info("No relevant results found")
                return "No relevant information found for that query."

            # Take top 2-3 results
            selected_results = new_results[:3]
            
            # Build context from relevant documents
            context_start = time.perf_counter()
            logger.info("Building context from search results...")
            
            context_parts = []
            for i, result in enumerate(selected_results):
                logger.debug(f"Processing result {i+1}: similarity={result.similarity:.4f}")
                
                # Mark as seen
                self._supabase_rag._seen_results.add(result.id)
                
                # Add content with metadata context if available
                content = result.content.strip()
                if content:
                    # Add source information if available in metadata
                    source_info = ""
                    if result.metadata:
                        if 'source_file' in result.metadata:
                            source_info = f" (Source: {result.metadata['source_file']})"
                        elif 'topic' in result.metadata:
                            source_info = f" (Topic: {result.metadata['topic']})"
                    
                    context_parts.append(f"{content}{source_info}")
                    logger.debug(f"Added content ({len(content)} chars) with similarity {result.similarity:.3f}")

            context_end = time.perf_counter()
            context_time = context_end - context_start
            logger.info(f"✓ Context building completed in {context_time:.3f}s")
            
            if not context_parts:
                logger.warning("No relevant content found in results")
                return "No relevant information found."

            full_context = "\n\n".join(context_parts)
            
            # Log overall timing
            overall_end = time.perf_counter()
            overall_time = overall_end - overall_start
            
            logger.info(f"""
📊 Knowledge Search Timing Summary:
- Embedding generation: {embedding_time:.3f}s ({embedding_time/overall_time*100:.1f}%)
- Vector search: {search_time:.3f}s ({search_time/overall_time*100:.1f}%)
- Result filtering: {filter_time:.3f}s ({filter_time/overall_time*100:.1f}%)
- Context building: {context_time:.3f}s ({context_time/overall_time*100:.1f}%)
- Total time: {overall_time:.3f}s
- Context length: {len(full_context)} characters
- Results used: {len(selected_results)}
            """)
            
            return full_context
            
        except Exception as e:
            overall_end = time.perf_counter()
            overall_time = overall_end - overall_start
            logger.error(f"❌ Error in knowledge search after {overall_time:.3f}s: {e}")
            return "Could not retrieve relevant information for that query."

    @function_tool
    async def get_database_stats(self, context: RunContext):
        """Get statistics about the knowledge database."""
        if not self._rag_available:
            return "Knowledge database is not available."
        
        try:
            stats = self._supabase_rag.get_stats()
            return f"Knowledge database contains {stats['total_documents']} documents across {stats['embedding_dimension']} dimensional embeddings in table '{stats['table_name']}'."
        except Exception as e:
            logger.error(f"Error getting database stats: {e}")
            return "Could not retrieve database statistics."

async def entrypoint(ctx: JobContext):
    """Main entrypoint for the agent."""
    logger.info("🚀 Starting LiveKit Supabase RAG agent...")
    
    # Create a session with RAG capabilities
    with timing_context("Agent session creation"):
        session = AgentSession(
            stt=deepgram.STT(),
            llm=openai.LLM(model="gpt-4o"),
            tts=hume.TTS(
                voice=hume.VoiceByName(
                    name="Ava Song", 
                    provider=hume.VoiceProvider.hume
                ),
                description="Previously Blunt Female Voice. An Asian American woman speaking with a lot of sass and personality.",
            ),
        )
    
    # Connect to the room
    with timing_context("Room connection"):
        await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
        logger.info(f"Connected to room: {ctx.room.name}")
    
    # Start the session with the RAG-enriched agent
    with timing_context("Agent session start"):
        await session.start(
            agent=RAGEnrichedAgent(),
            room=ctx.room,
        )

if __name__ == "__main__":
    # Verify environment variables
    required_env_vars = ["SUPABASE_URL", "SUPABASE_ANON_KEY", "OPENAI_API_KEY"]
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]
    
    if missing_vars:
        logger.error(f"Missing required environment variables: {missing_vars}")
        logger.error("Please check your .env file")
        exit(1)
    
    with timing_context("Application startup"):
        cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
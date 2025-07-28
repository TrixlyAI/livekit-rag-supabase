import os
import asyncio
from dotenv import load_dotenv
from supabase import create_client

async def check_table_structure():
    """Check the structure of your existing supabase_vector table."""
    
    load_dotenv()
    
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_ANON_KEY")
    
    try:
        supabase = create_client(supabase_url, supabase_key)
        
        print("🔍 Checking your supabase_vector table...")
        
        # Try to get one row to see the structure
        result = supabase.table('supabase_vector').select('*').limit(1).execute()
        
        if result.data:
            print("✅ Found data in supabase_vector table")
            
            # Show column names and sample data
            sample_row = result.data[0]
            columns = list(sample_row.keys())
            
            print(f"\n📋 Table columns ({len(columns)} total):")
            for col in columns:
                value = sample_row[col]
                value_type = type(value).__name__
                
                # Show sample value (truncated if long)
                if isinstance(value, str) and len(value) > 50:
                    sample_value = value[:50] + "..."
                elif isinstance(value, list) and len(value) > 5:
                    sample_value = f"[list with {len(value)} elements]"
                else:
                    sample_value = str(value)
                
                print(f"   - {col}: {value_type} = {sample_value}")
            
            # Check for required columns
            required_cols = ['content', 'embedding']
            missing_cols = [col for col in required_cols if col not in columns]
            
            print(f"\n🎯 RAG Compatibility Check:")
            if not missing_cols:
                print("✅ Perfect! Your table has all required columns (content, embedding)")
                print("🚀 You can use this table directly with the RAG system!")
                return True, 'supabase_vector', columns
            else:
                print(f"⚠️ Missing required columns: {missing_cols}")
                
                # Check for similar column names
                content_like = [col for col in columns if 'content' in col.lower() or 'text' in col.lower() or 'data' in col.lower()]
                embedding_like = [col for col in columns if 'embedding' in col.lower() or 'vector' in col.lower()]
                
                if content_like:
                    print(f"💡 Found content-like columns: {content_like}")
                if embedding_like:
                    print(f"💡 Found embedding-like columns: {embedding_like}")
                
                return False, 'supabase_vector', columns
        
        else:
            print("📭 Table exists but is empty")
            print("💡 We can still use it - let's check if it has the right structure")
            
            # Try to insert a test row to see what columns are expected
            test_data = {
                'content': 'test content',
                'embedding': [0.1] * 1536,  # 1536-dimensional vector
                'metadata': {'test': True}
            }
            
            try:
                insert_result = supabase.table('supabase_vector').insert(test_data).execute()
                if insert_result.data:
                    print("✅ Test insert successful - table has correct structure!")
                    
                    # Clean up
                    supabase.table('supabase_vector').delete().eq('id', insert_result.data[0]['id']).execute()
                    
                    return True, 'supabase_vector', ['content', 'embedding', 'metadata', 'id']
            except Exception as e:
                print(f"❌ Test insert failed: {e}")
                return False, 'supabase_vector', []
    
    except Exception as e:
        print(f"❌ Error checking table: {e}")
        return False, None, []

async def create_documents_table_option():
    """Alternative: Create a documents table."""
    
    create_table_sql = """
-- Create the documents table (since supabase_vector doesn't work)
CREATE TABLE IF NOT EXISTS documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    content TEXT NOT NULL,
    embedding VECTOR(1536),
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create index for vector similarity search
CREATE INDEX IF NOT EXISTS documents_embedding_idx 
ON documents 
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);
    """
    
    print("\n" + "="*60)
    print("📋 Alternative: Create documents table")
    print("="*60)
    print(create_table_sql)
    print("="*60)
    print("\nSteps:")
    print("1. Go to Supabase Dashboard → SQL Editor")
    print("2. Copy and paste the SQL above")
    print("3. Click 'Run'")
    print("4. Then we can use the documents table for RAG")

if __name__ == "__main__":
    print("🔍 Analyzing your existing table structure...")
    
    success, table_name, columns = asyncio.run(check_table_structure())
    
    if success:
        print(f"\n🎉 Great! We can use your '{table_name}' table for RAG!")
        print("\n🚀 Next step: Test the RAG system with your existing table")
    else:
        print(f"\n⚠️ Your existing table isn't compatible with RAG requirements")
        asyncio.run(create_documents_table_option())
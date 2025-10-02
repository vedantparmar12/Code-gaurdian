-- ============================================================
-- Supabase Schema for MCP Doc Fetcher Enhanced
-- Run this in Supabase SQL Editor
-- ============================================================

-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- ============ Sources Table ============
-- Tracks documentation sources (domains)
CREATE TABLE IF NOT EXISTS sources (
    source_id TEXT PRIMARY KEY,
    summary TEXT,
    total_words INTEGER DEFAULT 0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Index for faster lookups
CREATE INDEX IF NOT EXISTS idx_sources_updated ON sources(updated_at DESC);

-- ============ Crawled Pages Table ============
-- Stores documentation chunks with embeddings
CREATE TABLE IF NOT EXISTS crawled_pages (
    id BIGSERIAL PRIMARY KEY,
    url TEXT NOT NULL,
    chunk_number INTEGER NOT NULL,
    content TEXT NOT NULL,
    embedding vector(768),  -- nomic-embed-text dimension
    metadata JSONB,
    source_id TEXT REFERENCES sources(source_id) ON DELETE CASCADE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(url, chunk_number)
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_crawled_pages_source ON crawled_pages(source_id);
CREATE INDEX IF NOT EXISTS idx_crawled_pages_url ON crawled_pages(url);
CREATE INDEX IF NOT EXISTS idx_crawled_pages_embedding ON crawled_pages
    USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);

-- ============ Code Examples Table ============
-- Stores extracted code examples separately
CREATE TABLE IF NOT EXISTS code_examples (
    id BIGSERIAL PRIMARY KEY,
    url TEXT NOT NULL,
    chunk_number INTEGER NOT NULL,
    content TEXT NOT NULL,  -- The code itself
    summary TEXT,           -- AI-generated summary
    embedding vector(768),
    metadata JSONB,
    source_id TEXT REFERENCES sources(source_id) ON DELETE CASCADE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(url, chunk_number)
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_code_examples_source ON code_examples(source_id);
CREATE INDEX IF NOT EXISTS idx_code_examples_url ON code_examples(url);
CREATE INDEX IF NOT EXISTS idx_code_examples_embedding ON code_examples
    USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);

-- ============ Vector Search Function ============
-- Match documents by vector similarity
CREATE OR REPLACE FUNCTION match_documents(
    query_embedding vector(768),
    match_count INT DEFAULT 5,
    source_filter TEXT DEFAULT NULL
)
RETURNS TABLE (
    id BIGINT,
    url TEXT,
    chunk_number INT,
    content TEXT,
    metadata JSONB,
    source_id TEXT,
    similarity FLOAT
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT
        crawled_pages.id,
        crawled_pages.url,
        crawled_pages.chunk_number,
        crawled_pages.content,
        crawled_pages.metadata,
        crawled_pages.source_id,
        1 - (crawled_pages.embedding <=> query_embedding) AS similarity
    FROM crawled_pages
    WHERE (source_filter IS NULL OR crawled_pages.source_id = source_filter)
    ORDER BY crawled_pages.embedding <=> query_embedding
    LIMIT match_count;
END;
$$;

-- ============ Code Example Search Function ============
-- Match code examples by vector similarity
CREATE OR REPLACE FUNCTION match_code_examples(
    query_embedding vector(768),
    match_count INT DEFAULT 5,
    source_filter TEXT DEFAULT NULL
)
RETURNS TABLE (
    id BIGINT,
    url TEXT,
    chunk_number INT,
    content TEXT,
    summary TEXT,
    metadata JSONB,
    source_id TEXT,
    similarity FLOAT
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT
        code_examples.id,
        code_examples.url,
        code_examples.chunk_number,
        code_examples.content,
        code_examples.summary,
        code_examples.metadata,
        code_examples.source_id,
        1 - (code_examples.embedding <=> query_embedding) AS similarity
    FROM code_examples
    WHERE (source_filter IS NULL OR code_examples.source_id = source_filter)
    ORDER BY code_examples.embedding <=> query_embedding
    LIMIT match_count;
END;
$$;

-- ============ Update Timestamp Trigger ============
-- Auto-update updated_at on sources
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_sources_updated_at BEFORE UPDATE ON sources
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- ============ Row Level Security (Optional) ============
-- Enable RLS for security
ALTER TABLE sources ENABLE ROW LEVEL SECURITY;
ALTER TABLE crawled_pages ENABLE ROW LEVEL SECURITY;
ALTER TABLE code_examples ENABLE ROW LEVEL SECURITY;

-- Policy: Allow service role full access
CREATE POLICY "Service role can do anything" ON sources
    FOR ALL USING (auth.role() = 'service_role');

CREATE POLICY "Service role can do anything" ON crawled_pages
    FOR ALL USING (auth.role() = 'service_role');

CREATE POLICY "Service role can do anything" ON code_examples
    FOR ALL USING (auth.role() = 'service_role');

-- ============ Test Data (Optional) ============
-- Uncomment to insert test data
/*
INSERT INTO sources (source_id, summary, total_words) VALUES
    ('example.com', 'Example documentation site', 1000);

INSERT INTO crawled_pages (url, chunk_number, content, source_id) VALUES
    ('https://example.com/docs', 0, 'Test content', 'example.com');
*/

-- ============ Useful Queries ============

-- Count documents per source
-- SELECT source_id, COUNT(*) as doc_count
-- FROM crawled_pages
-- GROUP BY source_id
-- ORDER BY doc_count DESC;

-- Count code examples per source
-- SELECT source_id, COUNT(*) as code_count
-- FROM code_examples
-- GROUP BY source_id
-- ORDER BY code_count DESC;

-- Get database size
-- SELECT
--     pg_size_pretty(pg_database_size(current_database())) as db_size,
--     pg_size_pretty(pg_total_relation_size('crawled_pages')) as docs_size,
--     pg_size_pretty(pg_total_relation_size('code_examples')) as code_size;

"""
Supabase Client for Production Vector Storage

Handles:
- Document storage with embeddings
- Source tracking and management
- Code example storage
- Vector similarity search
- Hybrid search (vector + keyword)
"""
import logging
from typing import List, Dict, Any, Optional
from supabase import create_client, Client
import asyncio
from concurrent.futures import ThreadPoolExecutor

from config.settings import get_rag_config

logger = logging.getLogger(__name__)


class SupabaseVectorStore:
    """Production-ready vector storage using Supabase + pgvector."""

    def __init__(self, url: Optional[str] = None, key: Optional[str] = None):
        """
        Initialize Supabase client.

        Args:
            url: Supabase project URL (defaults to config)
            key: Supabase service key (defaults to config)
        """
        config = get_rag_config()
        self.url = url or config.supabase_url
        self.key = key or config.supabase_service_key

        if not self.url or not self.key:
            raise ValueError(
                "Supabase credentials not configured. "
                "Set SUPABASE_URL and SUPABASE_SERVICE_KEY in .env"
            )

        self.client: Client = create_client(self.url, self.key)
        self.config = config
        logger.info("Supabase client initialized successfully")

    # ============ Document Storage ============

    async def add_documents(
        self,
        urls: List[str],
        chunk_numbers: List[int],
        contents: List[str],
        embeddings: List[List[float]],
        metadatas: List[Dict[str, Any]],
        batch_size: int = 20
    ) -> Dict[str, Any]:
        """
        Add documents to Supabase in batches.

        Args:
            urls: List of source URLs
            chunk_numbers: Chunk indices
            contents: Document contents
            embeddings: Vector embeddings
            metadatas: Metadata dicts
            batch_size: Batch size for inserts

        Returns:
            Dict with success status and inserted count
        """
        if not all(len(lst) == len(urls) for lst in [chunk_numbers, contents, embeddings, metadatas]):
            raise ValueError("All input lists must have same length")

        total = len(urls)
        inserted = 0
        errors = []

        logger.info(f"Inserting {total} documents in batches of {batch_size}")

        # Process in batches
        for i in range(0, total, batch_size):
            batch_end = min(i + batch_size, total)
            batch_data = []

            for j in range(i, batch_end):
                # Extract source_id from metadata
                source_id = metadatas[j].get('source', 'unknown')

                batch_data.append({
                    'url': urls[j],
                    'chunk_number': chunk_numbers[j],
                    'content': contents[j],
                    'embedding': embeddings[j],
                    'metadata': metadatas[j],
                    'source_id': source_id
                })

            try:
                # Insert batch
                result = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self.client.table('crawled_pages').insert(batch_data).execute()
                )
                inserted += len(batch_data)
                logger.info(f"Batch {i//batch_size + 1}: Inserted {len(batch_data)} documents")

            except Exception as e:
                error_msg = f"Batch {i//batch_size + 1} failed: {str(e)}"
                logger.error(error_msg)
                errors.append(error_msg)

        return {
            'success': inserted > 0,
            'total': total,
            'inserted': inserted,
            'errors': errors
        }

    # ============ Vector Search ============

    async def vector_search(
        self,
        query_embedding: List[float],
        match_count: int = 5,
        source_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Perform vector similarity search.

        Args:
            query_embedding: Query vector
            match_count: Number of results
            source_filter: Optional source_id filter

        Returns:
            List of matching documents with similarity scores
        """
        try:
            # Call Supabase RPC function for vector search
            params = {
                'query_embedding': query_embedding,
                'match_count': match_count
            }

            if source_filter:
                params['source_filter'] = source_filter

            result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.client.rpc('match_documents', params).execute()
            )

            return result.data if result.data else []

        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return []

    async def keyword_search(
        self,
        query: str,
        match_count: int = 5,
        source_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Perform keyword search using ILIKE.

        Args:
            query: Search query
            match_count: Number of results
            source_filter: Optional source_id filter

        Returns:
            List of matching documents
        """
        try:
            # Build query
            search_query = self.client.table('crawled_pages')\
                .select('id, url, chunk_number, content, metadata, source_id')\
                .ilike('content', f'%{query}%')

            # Apply source filter if provided
            if source_filter:
                search_query = search_query.eq('source_id', source_filter)

            # Execute
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: search_query.limit(match_count).execute()
            )

            return result.data if result.data else []

        except Exception as e:
            logger.error(f"Keyword search failed: {e}")
            return []

    # ============ Hybrid Search ============

    async def hybrid_search(
        self,
        query: str,
        query_embedding: List[float],
        match_count: int = 5,
        source_filter: Optional[str] = None,
        boost_factor: float = 1.2
    ) -> List[Dict[str, Any]]:
        """
        Combine vector and keyword search for better results.

        Args:
            query: Search query text
            query_embedding: Query vector
            match_count: Number of final results
            source_filter: Optional source filter
            boost_factor: Boost for documents in both results

        Returns:
            Merged and ranked results
        """
        logger.info(f"Hybrid search: '{query}' (filter: {source_filter})")

        # Run both searches in parallel
        vector_task = self.vector_search(query_embedding, match_count * 2, source_filter)
        keyword_task = self.keyword_search(query, match_count * 2, source_filter)

        vector_results, keyword_results = await asyncio.gather(vector_task, keyword_task)

        # Merge results
        seen_ids = set()
        merged = []

        # 1. Add documents in both (highest priority)
        vector_ids = {r.get('id') for r in vector_results if r.get('id')}
        for kr in keyword_results:
            if kr['id'] in vector_ids and kr['id'] not in seen_ids:
                # Find matching vector result
                for vr in vector_results:
                    if vr.get('id') == kr['id']:
                        # Boost similarity
                        vr['similarity'] = min(1.0, vr.get('similarity', 0) * boost_factor)
                        vr['match_type'] = 'both'
                        merged.append(vr)
                        seen_ids.add(kr['id'])
                        break

        # 2. Add remaining vector results (semantic only)
        for vr in vector_results:
            if vr.get('id') and vr['id'] not in seen_ids and len(merged) < match_count:
                vr['match_type'] = 'semantic'
                merged.append(vr)
                seen_ids.add(vr['id'])

        # 3. Add remaining keyword results (exact keyword only)
        for kr in keyword_results:
            if kr['id'] not in seen_ids and len(merged) < match_count:
                kr['similarity'] = 0.5  # Default for keyword-only
                kr['match_type'] = 'keyword'
                merged.append(kr)
                seen_ids.add(kr['id'])

        logger.info(f"Hybrid search returned {len(merged)} results")
        return merged[:match_count]

    # ============ Source Management ============

    async def update_source(
        self,
        source_id: str,
        summary: str,
        total_words: int
    ) -> bool:
        """
        Update or create source entry.

        Args:
            source_id: Source identifier (domain)
            summary: AI-generated summary
            total_words: Total word count

        Returns:
            Success status
        """
        try:
            data = {
                'source_id': source_id,
                'summary': summary,
                'total_words': total_words
            }

            # Upsert (insert or update)
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.client.table('sources')\
                    .upsert(data, on_conflict='source_id')\
                    .execute()
            )

            logger.info(f"Updated source: {source_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to update source {source_id}: {e}")
            return False

    async def get_sources(self) -> List[Dict[str, Any]]:
        """
        Get all available sources.

        Returns:
            List of sources with metadata
        """
        try:
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.client.table('sources')\
                    .select('*')\
                    .order('source_id')\
                    .execute()
            )

            return result.data if result.data else []

        except Exception as e:
            logger.error(f"Failed to get sources: {e}")
            return []

    # ============ Code Examples ============

    async def add_code_examples(
        self,
        urls: List[str],
        chunk_numbers: List[int],
        codes: List[str],
        summaries: List[str],
        embeddings: List[List[float]],
        metadatas: List[Dict[str, Any]],
        batch_size: int = 20
    ) -> Dict[str, Any]:
        """
        Add code examples to separate table.

        Args:
            urls: Source URLs
            chunk_numbers: Chunk indices
            codes: Code blocks
            summaries: AI-generated summaries
            embeddings: Vector embeddings
            metadatas: Metadata dicts
            batch_size: Batch size

        Returns:
            Dict with success status
        """
        if not all(len(lst) == len(urls) for lst in [chunk_numbers, codes, summaries, embeddings, metadatas]):
            raise ValueError("All input lists must have same length")

        total = len(urls)
        inserted = 0
        errors = []

        logger.info(f"Inserting {total} code examples in batches of {batch_size}")

        for i in range(0, total, batch_size):
            batch_end = min(i + batch_size, total)
            batch_data = []

            for j in range(i, batch_end):
                source_id = metadatas[j].get('source', 'unknown')

                batch_data.append({
                    'url': urls[j],
                    'chunk_number': chunk_numbers[j],
                    'content': codes[j],  # Code content
                    'summary': summaries[j],
                    'embedding': embeddings[j],
                    'metadata': metadatas[j],
                    'source_id': source_id
                })

            try:
                result = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self.client.table('code_examples').insert(batch_data).execute()
                )
                inserted += len(batch_data)
                logger.info(f"Code batch {i//batch_size + 1}: Inserted {len(batch_data)} examples")

            except Exception as e:
                error_msg = f"Code batch {i//batch_size + 1} failed: {str(e)}"
                logger.error(error_msg)
                errors.append(error_msg)

        return {
            'success': inserted > 0,
            'total': total,
            'inserted': inserted,
            'errors': errors
        }

    async def search_code_examples(
        self,
        query_embedding: List[float],
        match_count: int = 5,
        source_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Search code examples using vector similarity.

        Args:
            query_embedding: Query vector
            match_count: Number of results
            source_filter: Optional source filter

        Returns:
            List of matching code examples
        """
        try:
            params = {
                'query_embedding': query_embedding,
                'match_count': match_count
            }

            if source_filter:
                params['source_filter'] = source_filter

            result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.client.rpc('match_code_examples', params).execute()
            )

            return result.data if result.data else []

        except Exception as e:
            logger.error(f"Code search failed: {e}")
            return []

    # ============ Utilities ============

    async def delete_by_source(self, source_id: str) -> bool:
        """
        Delete all documents from a source.

        Args:
            source_id: Source identifier

        Returns:
            Success status
        """
        try:
            # Delete from crawled_pages
            await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.client.table('crawled_pages')\
                    .delete()\
                    .eq('source_id', source_id)\
                    .execute()
            )

            # Delete from code_examples
            await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.client.table('code_examples')\
                    .delete()\
                    .eq('source_id', source_id)\
                    .execute()
            )

            # Delete source entry
            await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.client.table('sources')\
                    .delete()\
                    .eq('source_id', source_id)\
                    .execute()
            )

            logger.info(f"Deleted all data for source: {source_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to delete source {source_id}: {e}")
            return False

    async def get_stats(self) -> Dict[str, Any]:
        """
        Get database statistics.

        Returns:
            Dict with counts and stats
        """
        try:
            # Count documents
            docs_result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.client.table('crawled_pages')\
                    .select('id', count='exact')\
                    .execute()
            )

            # Count code examples
            code_result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.client.table('code_examples')\
                    .select('id', count='exact')\
                    .execute()
            )

            # Count sources
            sources_result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.client.table('sources')\
                    .select('source_id', count='exact')\
                    .execute()
            )

            return {
                'total_documents': docs_result.count if docs_result else 0,
                'total_code_examples': code_result.count if code_result else 0,
                'total_sources': sources_result.count if sources_result else 0
            }

        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {}


# ============ Helper Functions ============

def get_supabase_client() -> SupabaseVectorStore:
    """Get configured Supabase client (singleton pattern)."""
    return SupabaseVectorStore()


if __name__ == "__main__":
    # Test connection
    async def test():
        try:
            client = get_supabase_client()
            stats = await client.get_stats()
            print(f"Connected! Stats: {stats}")
        except Exception as e:
            print(f"Connection failed: {e}")

    asyncio.run(test())

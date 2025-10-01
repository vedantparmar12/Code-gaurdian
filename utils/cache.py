"""SQLite-based cache for documentation pages and embeddings."""

import json
import logging
import os
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any, Tuple

import aiosqlite
import numpy as np

from ..config import Settings
from ..models import LibraryDocumentation, DocumentPage, EmbeddingChunk, CacheStats, SearchResult

logger = logging.getLogger(__name__)


class DocumentCache:
    """Async SQLite-based cache for documentation and embeddings."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.db_path = settings.cache_db_path
        self._ensure_cache_directory()

    def _ensure_cache_directory(self):
        """Ensure cache directory exists."""
        cache_dir = os.path.dirname(self.db_path)
        if cache_dir and not os.path.exists(cache_dir):
            os.makedirs(cache_dir, exist_ok=True)

    async def initialize(self):
        """Initialize database schema."""
        async with aiosqlite.connect(self.db_path) as db:
            await self._create_tables(db)
            await self._create_indexes(db)
            await db.commit()

        logger.info(f"Initialized cache database at: {self.db_path}")

    async def _create_tables(self, db: aiosqlite.Connection):
        """Create database tables."""
        # Libraries table
        await db.execute("""
            CREATE TABLE IF NOT EXISTS libraries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                version TEXT NOT NULL,
                cache_key TEXT UNIQUE NOT NULL,
                total_pages INTEGER NOT NULL,
                created_at TIMESTAMP NOT NULL,
                updated_at TIMESTAMP NOT NULL,
                UNIQUE(name, version)
            )
        """)

        # Pages table
        await db.execute("""
            CREATE TABLE IF NOT EXISTS pages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                library_id INTEGER NOT NULL,
                url TEXT NOT NULL,
                title TEXT NOT NULL,
                content TEXT NOT NULL,
                markdown TEXT NOT NULL,
                word_count INTEGER NOT NULL,
                fetched_at TIMESTAMP NOT NULL,
                FOREIGN KEY (library_id) REFERENCES libraries (id) ON DELETE CASCADE
            )
        """)

        # Embeddings table
        await db.execute("""
            CREATE TABLE IF NOT EXISTS embeddings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                page_id INTEGER NOT NULL,
                chunk_index INTEGER NOT NULL,
                text_content TEXT NOT NULL,
                token_count INTEGER NOT NULL,
                embedding_vector BLOB NOT NULL,
                FOREIGN KEY (page_id) REFERENCES pages (id) ON DELETE CASCADE,
                UNIQUE(page_id, chunk_index)
            )
        """)

        # Cache metadata table for tracking stats
        await db.execute("""
            CREATE TABLE IF NOT EXISTS cache_metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at TIMESTAMP NOT NULL
            )
        """)

    async def _create_indexes(self, db: aiosqlite.Connection):
        """Create database indexes for performance."""
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_library_name_version ON libraries(name, version)",
            "CREATE INDEX IF NOT EXISTS idx_library_cache_key ON libraries(cache_key)",
            "CREATE INDEX IF NOT EXISTS idx_library_updated_at ON libraries(updated_at)",
            "CREATE INDEX IF NOT EXISTS idx_page_library_id ON pages(library_id)",
            "CREATE INDEX IF NOT EXISTS idx_page_url ON pages(url)",
            "CREATE INDEX IF NOT EXISTS idx_embedding_page_id ON embeddings(page_id)",
            "CREATE INDEX IF NOT EXISTS idx_embedding_page_chunk ON embeddings(page_id, chunk_index)",
        ]

        for index_sql in indexes:
            await db.execute(index_sql)

    async def store_library_documentation(
        self,
        documentation: LibraryDocumentation,
        embedding_chunks: List[EmbeddingChunk]
    ) -> bool:
        """
        Store complete library documentation with embeddings.

        Args:
            documentation: Library documentation to store
            embedding_chunks: Associated embedding chunks

        Returns:
            True if stored successfully, False otherwise
        """
        try:
            async with aiosqlite.connect(self.db_path) as db:
                # Start transaction
                await db.execute("BEGIN TRANSACTION")

                try:
                    # Insert or update library record
                    library_id = await self._upsert_library(db, documentation)

                    # Clear existing pages and embeddings for this library
                    await db.execute(
                        "DELETE FROM pages WHERE library_id = ?",
                        (library_id,)
                    )

                    # Insert pages
                    page_ids = await self._insert_pages(db, library_id, documentation.pages)

                    # Insert embeddings
                    await self._insert_embeddings(db, page_ids, embedding_chunks)

                    # Update metadata
                    await self._update_cache_metadata(db)

                    # Commit transaction
                    await db.execute("COMMIT")

                    logger.info(
                        f"Stored documentation for {documentation.library_name}:{documentation.version} "
                        f"({len(documentation.pages)} pages, {len(embedding_chunks)} embeddings)"
                    )
                    return True

                except Exception as e:
                    await db.execute("ROLLBACK")
                    logger.error(f"Error in transaction: {e}")
                    return False

        except Exception as e:
            logger.error(f"Error storing library documentation: {e}")
            return False

    async def _upsert_library(
        self,
        db: aiosqlite.Connection,
        documentation: LibraryDocumentation
    ) -> int:
        """Upsert library record and return library_id."""
        # Try to update existing record
        cursor = await db.execute("""
            UPDATE libraries
            SET total_pages = ?, updated_at = ?
            WHERE cache_key = ?
        """, (
            documentation.total_pages,
            documentation.updated_at,
            documentation.cache_key
        ))

        if cursor.rowcount > 0:
            # Get the existing library_id
            cursor = await db.execute(
                "SELECT id FROM libraries WHERE cache_key = ?",
                (documentation.cache_key,)
            )
            row = await cursor.fetchone()
            return row[0]

        # Insert new record
        cursor = await db.execute("""
            INSERT INTO libraries (name, version, cache_key, total_pages, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            documentation.library_name,
            documentation.version,
            documentation.cache_key,
            documentation.total_pages,
            documentation.created_at,
            documentation.updated_at
        ))

        return cursor.lastrowid

    async def _insert_pages(
        self,
        db: aiosqlite.Connection,
        library_id: int,
        pages: List[DocumentPage]
    ) -> List[int]:
        """Insert pages and return list of page_ids."""
        page_ids = []

        for page in pages:
            cursor = await db.execute("""
                INSERT INTO pages (library_id, url, title, content, markdown, word_count, fetched_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                library_id,
                str(page.url),
                page.title,
                page.content,
                page.markdown,
                page.word_count,
                page.fetched_at
            ))

            page_ids.append(cursor.lastrowid)

        return page_ids

    async def _insert_embeddings(
        self,
        db: aiosqlite.Connection,
        page_ids: List[int],
        embedding_chunks: List[EmbeddingChunk]
    ) -> None:
        """Insert embedding chunks."""
        # Create mapping from original page_id to database page_id
        page_id_mapping = {i: page_ids[i] for i in range(len(page_ids))}

        for chunk in embedding_chunks:
            # Map original page_id to database page_id
            db_page_id = page_id_mapping.get(chunk.page_id)
            if db_page_id is None:
                logger.warning(f"No page_id mapping found for chunk page_id {chunk.page_id}")
                continue

            # Serialize embedding vector as binary data
            vector_blob = self._serialize_embedding_vector(chunk.embedding_vector)

            await db.execute("""
                INSERT INTO embeddings (page_id, chunk_index, text_content, token_count, embedding_vector)
                VALUES (?, ?, ?, ?, ?)
            """, (
                db_page_id,
                chunk.chunk_index,
                chunk.text_content,
                chunk.token_count,
                vector_blob
            ))

    def _serialize_embedding_vector(self, vector: List[float]) -> bytes:
        """Serialize embedding vector to binary format."""
        np_array = np.array(vector, dtype=np.float32)
        return np_array.tobytes()

    def _deserialize_embedding_vector(self, blob: bytes) -> List[float]:
        """Deserialize embedding vector from binary format."""
        np_array = np.frombuffer(blob, dtype=np.float32)
        return np_array.tolist()

    async def _update_cache_metadata(self, db: aiosqlite.Connection):
        """Update cache metadata for statistics."""
        now = datetime.now()

        await db.execute("""
            INSERT OR REPLACE INTO cache_metadata (key, value, updated_at)
            VALUES ('last_updated', ?, ?)
        """, (now.isoformat(), now))

    async def get_library_documentation(
        self,
        cache_key: str
    ) -> Optional[LibraryDocumentation]:
        """
        Retrieve library documentation by cache key.

        Args:
            cache_key: Cache key for the library

        Returns:
            LibraryDocumentation if found, None otherwise
        """
        try:
            async with aiosqlite.connect(self.db_path) as db:
                db.row_factory = aiosqlite.Row

                # Get library info
                cursor = await db.execute("""
                    SELECT * FROM libraries WHERE cache_key = ?
                """, (cache_key,))

                library_row = await cursor.fetchone()
                if not library_row:
                    return None

                # Check if cache is still valid
                if self._is_cache_expired(library_row['updated_at']):
                    logger.debug(f"Cache expired for {cache_key}")
                    return None

                # Get pages
                cursor = await db.execute("""
                    SELECT * FROM pages WHERE library_id = ? ORDER BY id
                """, (library_row['id'],))

                page_rows = await cursor.fetchall()

                # Convert to DocumentPage objects
                pages = []
                for row in page_rows:
                    page = DocumentPage(
                        url=row['url'],
                        title=row['title'],
                        content=row['content'],
                        markdown=row['markdown'],
                        word_count=row['word_count'],
                        fetched_at=datetime.fromisoformat(row['fetched_at'])
                    )
                    pages.append(page)

                # Create LibraryDocumentation object
                documentation = LibraryDocumentation(
                    library_name=library_row['name'],
                    version=library_row['version'],
                    pages=pages,
                    total_pages=library_row['total_pages'],
                    cache_key=library_row['cache_key'],
                    created_at=datetime.fromisoformat(library_row['created_at']),
                    updated_at=datetime.fromisoformat(library_row['updated_at'])
                )

                return documentation

        except Exception as e:
            logger.error(f"Error retrieving library documentation: {e}")
            return None

    def _is_cache_expired(self, updated_at: str) -> bool:
        """Check if cache entry is expired."""
        try:
            last_updated = datetime.fromisoformat(updated_at)
            max_age = timedelta(hours=self.settings.cache_max_age_hours)
            return datetime.now() - last_updated > max_age
        except Exception:
            return True

    async def search_embeddings(
        self,
        library_name: str,
        query_embedding: List[float],
        max_results: int = 10,
        min_similarity: float = 0.3
    ) -> List[Tuple[str, str, float]]:
        """
        Search embeddings for semantic similarity.

        Args:
            library_name: Library to search in
            query_embedding: Query embedding vector
            max_results: Maximum results to return
            min_similarity: Minimum similarity threshold

        Returns:
            List of (text_content, page_title, similarity_score) tuples
        """
        try:
            async with aiosqlite.connect(self.db_path) as db:
                db.row_factory = aiosqlite.Row

                # Get all embeddings for the library
                cursor = await db.execute("""
                    SELECT e.text_content, e.embedding_vector, p.title, p.url
                    FROM embeddings e
                    JOIN pages p ON e.page_id = p.id
                    JOIN libraries l ON p.library_id = l.id
                    WHERE l.name = ?
                    ORDER BY e.page_id, e.chunk_index
                """, (library_name,))

                rows = await cursor.fetchall()

                if not rows:
                    return []

                # Compute similarities
                results = []
                query_vec = np.array(query_embedding, dtype=np.float32)

                for row in rows:
                    # Deserialize embedding vector
                    doc_embedding = self._deserialize_embedding_vector(row['embedding_vector'])
                    doc_vec = np.array(doc_embedding, dtype=np.float32)

                    # Compute cosine similarity
                    similarity = float(np.dot(query_vec, doc_vec))

                    if similarity >= min_similarity:
                        results.append((
                            row['text_content'],
                            row['title'],
                            similarity
                        ))

                # Sort by similarity (descending) and limit results
                results.sort(key=lambda x: x[2], reverse=True)
                return results[:max_results]

        except Exception as e:
            logger.error(f"Error searching embeddings: {e}")
            return []

    async def get_cache_stats(self) -> CacheStats:
        """Get cache statistics."""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                # Count libraries
                cursor = await db.execute("SELECT COUNT(*) FROM libraries")
                total_libraries = (await cursor.fetchone())[0]

                # Count pages
                cursor = await db.execute("SELECT COUNT(*) FROM pages")
                total_pages = (await cursor.fetchone())[0]

                # Get oldest and newest entries
                cursor = await db.execute("""
                    SELECT MIN(created_at), MAX(updated_at) FROM libraries
                """)
                date_row = await cursor.fetchone()

                oldest_entry = None
                newest_entry = None

                if date_row and date_row[0]:
                    oldest_entry = datetime.fromisoformat(date_row[0])
                if date_row and date_row[1]:
                    newest_entry = datetime.fromisoformat(date_row[1])

                # Calculate cache size
                cache_size_mb = 0.0
                if os.path.exists(self.db_path):
                    cache_size_mb = os.path.getsize(self.db_path) / (1024 * 1024)

                return CacheStats(
                    total_libraries=total_libraries,
                    total_pages=total_pages,
                    cache_size_mb=cache_size_mb,
                    oldest_entry=oldest_entry,
                    newest_entry=newest_entry
                )

        except Exception as e:
            logger.error(f"Error getting cache stats: {e}")
            return CacheStats(
                total_libraries=0,
                total_pages=0,
                cache_size_mb=0.0,
                oldest_entry=None,
                newest_entry=None
            )

    async def clear_cache(self, library_name: Optional[str] = None) -> int:
        """
        Clear cache entries.

        Args:
            library_name: If provided, only clear this library. Otherwise clear all.

        Returns:
            Number of libraries cleared
        """
        try:
            async with aiosqlite.connect(self.db_path) as db:
                if library_name:
                    # Clear specific library
                    cursor = await db.execute("""
                        DELETE FROM libraries WHERE name = ?
                    """, (library_name,))
                    cleared = cursor.rowcount
                    logger.info(f"Cleared cache for library: {library_name}")
                else:
                    # Clear all
                    cursor = await db.execute("SELECT COUNT(*) FROM libraries")
                    cleared = (await cursor.fetchone())[0]

                    await db.execute("DELETE FROM libraries")
                    await db.execute("DELETE FROM pages")
                    await db.execute("DELETE FROM embeddings")
                    await db.execute("DELETE FROM cache_metadata")

                    logger.info("Cleared all cache entries")

                await db.commit()
                return cleared

        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
            return 0

    async def list_cached_libraries(self) -> List[Dict[str, Any]]:
        """List all cached libraries with metadata."""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                db.row_factory = aiosqlite.Row

                cursor = await db.execute("""
                    SELECT name, version, cache_key, total_pages, created_at, updated_at
                    FROM libraries
                    ORDER BY updated_at DESC
                """)

                rows = await cursor.fetchall()

                libraries = []
                for row in rows:
                    libraries.append({
                        'name': row['name'],
                        'version': row['version'],
                        'cache_key': row['cache_key'],
                        'total_pages': row['total_pages'],
                        'created_at': row['created_at'],
                        'updated_at': row['updated_at'],
                        'is_expired': self._is_cache_expired(row['updated_at'])
                    })

                return libraries

        except Exception as e:
            logger.error(f"Error listing cached libraries: {e}")
            return []

    async def close(self):
        """Close any persistent connections (placeholder for future use)."""
        pass


# Convenience functions
async def get_cached_documentation(
    library_name: str,
    version: str = "latest",
    settings: Optional[Settings] = None
) -> Optional[LibraryDocumentation]:
    """
    Convenience function to get cached documentation.

    Args:
        library_name: Name of the library
        version: Version of the library
        settings: Application settings

    Returns:
        LibraryDocumentation if cached, None otherwise
    """
    from ..config import get_settings

    if settings is None:
        settings = get_settings()

    cache_key = f"{library_name}:{version}"
    cache = DocumentCache(settings)
    await cache.initialize()

    return await cache.get_library_documentation(cache_key)
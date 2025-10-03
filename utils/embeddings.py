"""Ollama embeddings integration for semantic search of documentation."""

import asyncio
import logging
from typing import List, Tuple, Optional, Dict, Any
import re

import numpy as np
import tiktoken
import ollama

# Try to import langchain_ollama, fallback to direct ollama client
try:
    from langchain_ollama import OllamaEmbeddings
    HAS_LANGCHAIN_OLLAMA = True
except ImportError:
    HAS_LANGCHAIN_OLLAMA = False
    OllamaEmbeddings = None

from ..config import Settings
from ..models import DocumentPage, EmbeddingChunk
from .docling_chunker import create_chunker, ChunkResult

logger = logging.getLogger(__name__)


class DocumentEmbedder:
    """Handles text chunking and embedding generation using Ollama."""

    def __init__(self, settings: Settings):
        self.settings = settings

        # Initialize embeddings client
        if HAS_LANGCHAIN_OLLAMA and OllamaEmbeddings:
            self.embeddings = OllamaEmbeddings(
                base_url=settings.ollama_url,
                model=settings.ollama_embedding_model,
            )
            self.use_langchain = True
        else:
            # Use direct ollama client as fallback
            self.embeddings = None
            self.use_langchain = False
            logger.info("Using direct Ollama client for embeddings (langchain_ollama not available)")

        # Initialize tokenizer for accurate token counting
        try:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        except Exception as e:
            logger.warning(f"Failed to load tiktoken encoder: {e}")
            self.tokenizer = None

        # Initialize Docling chunker if enabled
        self.chunker = None
        if settings.use_docling_chunking:
            try:
                self.chunker = create_chunker(
                    use_docling=True,
                    max_tokens=settings.max_tokens_per_chunk,
                    embedding_model=settings.chunking_embedding_model
                )
                logger.info(
                    f"Initialized Docling chunker (max_tokens={settings.max_tokens_per_chunk})"
                )
            except Exception as e:
                logger.warning(f"Failed to initialize Docling chunker: {e}, using fallback")
                self.chunker = create_chunker(use_docling=False, max_tokens=settings.max_tokens_per_chunk)

    async def generate_embeddings_for_pages(
        self,
        pages: List[DocumentPage]
    ) -> List[EmbeddingChunk]:
        """
        Generate embeddings for a list of documentation pages.

        Args:
            pages: List of DocumentPage objects to embed

        Returns:
            List of EmbeddingChunk objects with embeddings
        """
        all_chunks = []

        for page_idx, page in enumerate(pages):
            logger.info(f"Generating embeddings for page {page_idx + 1}/{len(pages)}: {page.title}")

            # Chunk the page content
            text_chunks = await self._chunk_page_content(page)

            # Generate embeddings for chunks in batches
            chunk_embeddings = await self._generate_chunk_embeddings(
                text_chunks,
                page_id=page_idx,
                batch_size=5  # Process 5 chunks at a time
            )

            all_chunks.extend(chunk_embeddings)

        logger.info(f"Generated {len(all_chunks)} embedding chunks from {len(pages)} pages")
        return all_chunks

    async def _chunk_page_content(self, page: DocumentPage) -> List[str]:
        """
        Split page content into optimally-sized chunks for embedding.

        Args:
            page: DocumentPage to chunk

        Returns:
            List of text chunks
        """
        # Use markdown content as it's cleaner
        content = page.markdown

        # Clean and prepare content
        content = self._preprocess_content(content)

        if not content.strip():
            logger.warning(f"Empty content for page: {page.title}")
            return []

        # Use Docling chunker if available
        if self.chunker:
            try:
                # Use asyncio to run async chunker
                chunk_results = await self.chunker.chunk_markdown(
                    markdown=content,
                    title=page.title,
                    url=str(page.url),
                    metadata={"source": "documentation"}
                )

                # Extract text content from ChunkResult objects
                chunks = [chunk.content for chunk in chunk_results if len(chunk.content.strip()) >= 100]

                # Store token counts for later use
                for i, chunk_result in enumerate(chunk_results):
                    if hasattr(page, 'chunk_token_counts'):
                        page.chunk_token_counts = getattr(page, 'chunk_token_counts', [])
                        page.chunk_token_counts.append(chunk_result.token_count)

                logger.debug(
                    f"Split '{page.title}' into {len(chunks)} chunks using Docling "
                    f"(avg tokens: {sum(c.token_count for c in chunk_results) // len(chunk_results) if chunk_results else 0})"
                )
                return chunks

            except Exception as e:
                logger.warning(f"Docling chunking failed for '{page.title}': {e}, using fallback")

        # Fallback to original chunking
        chunks = self._split_content_intelligently(
            content,
            max_chunk_size=self.settings.embedding_chunk_size,
            overlap_size=self.settings.embedding_chunk_overlap
        )

        # Filter out chunks that are too small or not meaningful
        filtered_chunks = []
        for chunk in chunks:
            if len(chunk.strip()) >= 100:  # Minimum 100 characters
                filtered_chunks.append(chunk.strip())

        logger.debug(f"Split '{page.title}' into {len(filtered_chunks)} chunks (fallback method)")
        return filtered_chunks

    def _preprocess_content(self, content: str) -> str:
        """
        Clean and preprocess content for better embedding quality.

        Args:
            content: Raw content to preprocess

        Returns:
            Cleaned content
        """
        if not content:
            return ""

        # Remove excessive whitespace
        content = re.sub(r'\n{3,}', '\n\n', content)
        content = re.sub(r' {3,}', '  ', content)

        # Remove markdown artifacts that don't add semantic meaning
        content = re.sub(r'```[\w]*\n', '```\n', content)  # Clean code block languages
        content = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', content)  # Convert [text](url) to text
        content = re.sub(r'#+\s*', '', content)  # Remove heading markers but keep text

        # Remove navigation and UI elements
        ui_patterns = [
            r'Edit this page.*',
            r'Improve this page.*',
            r'Last updated.*',
            r'Table of contents.*',
            r'Previous.*Next.*',
            r'Back to top.*',
        ]

        for pattern in ui_patterns:
            content = re.sub(pattern, '', content, flags=re.IGNORECASE)

        return content.strip()

    def _split_content_intelligently(
        self,
        content: str,
        max_chunk_size: int,
        overlap_size: int
    ) -> List[str]:
        """
        Split content intelligently preserving semantic boundaries.

        Args:
            content: Content to split
            max_chunk_size: Maximum tokens per chunk
            overlap_size: Overlap tokens between chunks

        Returns:
            List of content chunks
        """
        # Split on different levels of semantic boundaries
        split_patterns = [
            r'\n#{1,2}\s+',  # H1/H2 headers
            r'\n#{3,4}\s+',  # H3/H4 headers
            r'\n\n\s*',      # Paragraph breaks
            r'\n\s*[-*+]\s+', # List items
            r'\.\s+',        # Sentences
            r',\s+',         # Clauses
        ]

        chunks = [content]

        for pattern in split_patterns:
            new_chunks = []

            for chunk in chunks:
                if self._count_tokens(chunk) <= max_chunk_size:
                    new_chunks.append(chunk)
                    continue

                # Split this chunk using the current pattern
                parts = re.split(pattern, chunk)
                if len(parts) > 1:
                    # Rejoin with overlap
                    current_chunk = ""

                    for i, part in enumerate(parts):
                        # Add the split pattern back (except for the first part)
                        if i > 0:
                            match = re.search(pattern, chunk)
                            if match:
                                part = match.group(0) + part

                        if not current_chunk:
                            current_chunk = part
                        elif self._count_tokens(current_chunk + part) <= max_chunk_size:
                            current_chunk += part
                        else:
                            # Finalize current chunk
                            if current_chunk.strip():
                                new_chunks.append(current_chunk.strip())

                            # Start new chunk with overlap
                            overlap_text = self._get_overlap_text(current_chunk, overlap_size)
                            current_chunk = overlap_text + part

                    # Add final chunk
                    if current_chunk.strip():
                        new_chunks.append(current_chunk.strip())
                else:
                    # Pattern didn't match, force split by character count
                    new_chunks.extend(self._force_split_chunk(chunk, max_chunk_size, overlap_size))

            chunks = new_chunks

            # Check if all chunks are now under the limit
            oversized = [c for c in chunks if self._count_tokens(c) > max_chunk_size]
            if not oversized:
                break

        return [chunk for chunk in chunks if chunk.strip()]

    def _get_overlap_text(self, text: str, overlap_size: int) -> str:
        """
        Get the last overlap_size tokens from text for chunk overlap.

        Args:
            text: Source text
            overlap_size: Number of tokens for overlap

        Returns:
            Overlap text
        """
        if overlap_size <= 0:
            return ""

        words = text.split()
        if len(words) <= overlap_size:
            return text

        overlap_words = words[-overlap_size:]
        return " ".join(overlap_words) + " "

    def _force_split_chunk(self, chunk: str, max_size: int, overlap_size: int) -> List[str]:
        """
        Force split a chunk that's too large by character count.

        Args:
            chunk: Chunk to split
            max_size: Maximum size in tokens
            overlap_size: Overlap size in tokens

        Returns:
            List of smaller chunks
        """
        # Estimate characters per token (rough approximation)
        chars_per_token = 4
        max_chars = max_size * chars_per_token
        overlap_chars = overlap_size * chars_per_token

        chunks = []
        start = 0

        while start < len(chunk):
            end = min(start + max_chars, len(chunk))

            # Try to end at a word boundary
            if end < len(chunk):
                # Look back for a word boundary
                space_pos = chunk.rfind(' ', start, end)
                if space_pos > start:
                    end = space_pos

            chunk_text = chunk[start:end].strip()
            if chunk_text:
                chunks.append(chunk_text)

            # Move start forward with overlap
            start = max(start + max_chars - overlap_chars, end)
            if start >= len(chunk):
                break

        return chunks

    def _count_tokens(self, text: str) -> int:
        """
        Count tokens in text using tiktoken or fallback estimation.

        Args:
            text: Text to count tokens for

        Returns:
            Number of tokens
        """
        if self.tokenizer:
            try:
                return len(self.tokenizer.encode(text))
            except Exception:
                pass

        # Fallback estimation: roughly 1 token per 4 characters
        return max(1, len(text) // 4)

    async def _generate_chunk_embeddings(
        self,
        chunks: List[str],
        page_id: int,
        batch_size: int = 5
    ) -> List[EmbeddingChunk]:
        """
        Generate embeddings for text chunks in batches.

        Args:
            chunks: Text chunks to embed
            page_id: ID of the source page
            batch_size: Number of chunks to process at once

        Returns:
            List of EmbeddingChunk objects
        """
        embedding_chunks = []

        # Process chunks in batches to manage memory and API limits
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]

            try:
                # Generate embeddings for the batch
                logger.debug(f"Generating embeddings for batch {i//batch_size + 1} ({len(batch)} chunks)")

                # Use langchain or direct ollama client
                if self.use_langchain and self.embeddings:
                    # Use asyncio to run the synchronous embedding generation
                    embeddings = await asyncio.get_event_loop().run_in_executor(
                        None,
                        self.embeddings.embed_documents,
                        batch
                    )
                else:
                    # Use direct Ollama client
                    embeddings = []
                    for text in batch:
                        response = await asyncio.get_event_loop().run_in_executor(
                            None,
                            lambda t: ollama.embeddings(
                                model=self.settings.ollama_embedding_model,
                                prompt=t
                            ),
                            text
                        )
                        embeddings.append(response['embedding'])

                # Create EmbeddingChunk objects
                for j, (text, embedding) in enumerate(zip(batch, embeddings)):
                    # Normalize embedding vector for cosine similarity
                    normalized_embedding = self._normalize_vector(embedding)

                    chunk = EmbeddingChunk(
                        page_id=page_id,
                        chunk_index=i + j,
                        text_content=text,
                        token_count=self._count_tokens(text),
                        embedding_vector=normalized_embedding
                    )

                    embedding_chunks.append(chunk)

            except Exception as e:
                logger.error(f"Error generating embeddings for batch {i//batch_size + 1}: {e}")
                continue

            # Small delay between batches to be respectful to the API
            if i + batch_size < len(chunks):
                await asyncio.sleep(0.1)

        logger.debug(f"Generated {len(embedding_chunks)} embeddings for page_id {page_id}")
        return embedding_chunks

    def _normalize_vector(self, vector: List[float]) -> List[float]:
        """
        Normalize embedding vector for cosine similarity search.

        Args:
            vector: Raw embedding vector

        Returns:
            Normalized vector
        """
        try:
            np_vector = np.array(vector, dtype=np.float32)
            norm = np.linalg.norm(np_vector)

            if norm == 0:
                logger.warning("Zero norm vector encountered")
                return vector

            normalized = np_vector / norm
            return normalized.tolist()

        except Exception as e:
            logger.error(f"Error normalizing vector: {e}")
            return vector

    async def generate_query_embedding(self, query: str) -> Optional[List[float]]:
        """
        Generate embedding for a search query.

        Args:
            query: Search query text

        Returns:
            Normalized embedding vector or None if failed
        """
        try:
            logger.debug(f"Generating query embedding for: {query[:50]}...")

            # Generate embedding using langchain or direct ollama
            if self.use_langchain and self.embeddings:
                embedding = await asyncio.get_event_loop().run_in_executor(
                    None,
                    self.embeddings.embed_query,
                    query
                )
            else:
                # Use direct Ollama client
                response = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: ollama.embeddings(
                        model=self.settings.ollama_embedding_model,
                        prompt=query
                    )
                )
                embedding = response['embedding']

            # Normalize for cosine similarity
            normalized_embedding = self._normalize_vector(embedding)

            return normalized_embedding

        except Exception as e:
            logger.error(f"Error generating query embedding: {e}")
            return None

    def compute_similarity(
        self,
        query_embedding: List[float],
        document_embeddings: List[List[float]]
    ) -> List[float]:
        """
        Compute cosine similarity scores between query and document embeddings.

        Args:
            query_embedding: Query embedding vector
            document_embeddings: List of document embedding vectors

        Returns:
            List of similarity scores (0-1, higher is more similar)
        """
        try:
            query_vec = np.array(query_embedding, dtype=np.float32)
            doc_vecs = np.array(document_embeddings, dtype=np.float32)

            # Compute cosine similarity using dot product (vectors are already normalized)
            similarities = np.dot(doc_vecs, query_vec)

            # Ensure scores are in [0, 1] range
            similarities = np.clip(similarities, 0, 1)

            return similarities.tolist()

        except Exception as e:
            logger.error(f"Error computing similarities: {e}")
            return [0.0] * len(document_embeddings)


# Convenience functions
async def generate_page_embeddings(
    pages: List[DocumentPage],
    settings: Optional[Settings] = None
) -> List[EmbeddingChunk]:
    """
    Convenience function to generate embeddings for documentation pages.

    Args:
        pages: Pages to generate embeddings for
        settings: Application settings

    Returns:
        List of embedding chunks
    """
    from ..config import get_settings

    if settings is None:
        settings = get_settings()

    embedder = DocumentEmbedder(settings)
    return await embedder.generate_embeddings_for_pages(pages)


async def search_similar_chunks(
    query: str,
    embedding_chunks: List[EmbeddingChunk],
    top_k: int = 5,
    settings: Optional[Settings] = None
) -> List[Tuple[EmbeddingChunk, float]]:
    """
    Search for similar chunks using semantic similarity.

    Args:
        query: Search query
        embedding_chunks: Available embedding chunks
        top_k: Number of top results to return
        settings: Application settings

    Returns:
        List of (chunk, similarity_score) tuples, sorted by relevance
    """
    from ..config import get_settings

    if settings is None:
        settings = get_settings()

    if not embedding_chunks:
        return []

    embedder = DocumentEmbedder(settings)

    # Generate query embedding
    query_embedding = await embedder.generate_query_embedding(query)
    if not query_embedding:
        return []

    # Extract document embeddings
    doc_embeddings = [chunk.embedding_vector for chunk in embedding_chunks]

    # Compute similarities
    similarities = embedder.compute_similarity(query_embedding, doc_embeddings)

    # Create results with scores
    results = [
        (chunk, score)
        for chunk, score in zip(embedding_chunks, similarities)
    ]

    # Sort by similarity score (descending) and take top_k
    results.sort(key=lambda x: x[1], reverse=True)

    return results[:top_k]
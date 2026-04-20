import hashlib
from typing import List

from app.processing.models import Chunk, NormalisedDocument


class DocumentChunker:
    def __init__(
        self,
        chunk_size: int = 1200,
        chunk_overlap: int = 200,
        min_chunk_size: int = 100,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be > 0")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap must be >= 0")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        if min_chunk_size < 1:
            raise ValueError("min_chunk_size must be >= 1")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size

    def chunk(self, document: NormalisedDocument) -> List[Chunk]:
        text = document.normalised_text

        if not text.strip():
            return []

        spans = self._split_into_spans(text)
        chunks: List[Chunk] = []

        for index, (start, end) in enumerate(spans):
            chunk_text = text[start:end].strip()

            if len(chunk_text) < self.min_chunk_size and index != 0:
                continue

            chunk_id = self._build_chunk_id(document.doc_id, index, chunk_text)

            metadata = {
                **document.metadata,
                "source_path": document.source_path,
                "file_type": document.file_type,
                "chunk_index": index,
                "start_char": start,
                "end_char": end,
                "chunk_char_count": len(chunk_text),
            }

            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    doc_id=document.doc_id,
                    chunk_index=index,
                    text=chunk_text,
                    metadata=metadata,
                )
            )

        return chunks

    def _split_into_spans(self, text: str) -> List[tuple[int, int]]:
        spans: List[tuple[int, int]] = []
        text_length = len(text)

        start = 0
        while start < text_length:
            candidate_end = min(start + self.chunk_size, text_length)

            if candidate_end < text_length:
                adjusted_end = self._find_good_boundary(text, start, candidate_end)
            else:
                adjusted_end = candidate_end

            if adjusted_end <= start:
                adjusted_end = candidate_end

            spans.append((start, adjusted_end))

            if adjusted_end >= text_length:
                break

            start = max(adjusted_end - self.chunk_overlap, 0)

        return spans

    def _find_good_boundary(self, text: str, start: int, candidate_end: int) -> int:
        """
        Prefer chunk endings at:
        1. paragraph boundary
        2. newline
        3. sentence-ish boundary
        4. hard cutoff
        """
        search_window_start = max(start, candidate_end - 200)
        window = text[search_window_start:candidate_end]

        paragraph_break = window.rfind("\n\n")
        if paragraph_break != -1:
            return search_window_start + paragraph_break + 2

        line_break = window.rfind("\n")
        if line_break != -1:
            return search_window_start + line_break + 1

        sentence_breaks = [". ", "! ", "? "]
        best_sentence_break = -1
        for marker in sentence_breaks:
            pos = window.rfind(marker)
            if pos > best_sentence_break:
                best_sentence_break = pos

        if best_sentence_break != -1:
            return search_window_start + best_sentence_break + 2

        return candidate_end

    def _build_chunk_id(self, doc_id: str, chunk_index: int, chunk_text: str) -> str:
        digest = hashlib.sha256(chunk_text.encode("utf-8")).hexdigest()[:16]
        return f"{doc_id}_{chunk_index}_{digest}"
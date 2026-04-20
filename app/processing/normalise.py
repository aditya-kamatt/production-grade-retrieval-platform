import re

from app.ingestion.models import ExtractedDocument
from app.processing.models import NormalisedDocument


class DocumentNormaliser:
    def normalise(self, document: ExtractedDocument) -> NormalisedDocument:
        text = document.raw_text

        text = self._normalise_line_endings(text)
        text = self._remove_null_bytes(text)
        text = self._replace_nonbreaking_spaces(text)
        text = self._strip_trailing_whitespace(text)
        text = self._collapse_excessive_blank_lines(text)
        text = self._collapse_inline_whitespace(text)
        text = text.strip()

        metadata = {
            **document.metadata,
            "normalised_char_count": len(text),
        }

        return NormalisedDocument(
            doc_id=document.doc_id,
            source_path=document.source_path,
            file_type=document.file_type,
            normalised_text=text,
            metadata=metadata,
            pages=document.pages,
        )

    def _normalise_line_endings(self, text: str) -> str:
        return text.replace("\r\n", "\n").replace("\r", "\n")

    def _remove_null_bytes(self, text: str) -> str:
        return text.replace("\x00", "")

    def _replace_nonbreaking_spaces(self, text: str) -> str:
        return text.replace("\u00a0", " ")

    def _strip_trailing_whitespace(self, text: str) -> str:
        lines = [line.rstrip() for line in text.split("\n")]
        return "\n".join(lines)

    def _collapse_excessive_blank_lines(self, text: str) -> str:
        return re.sub(r"\n{3,}", "\n\n", text)

    def _collapse_inline_whitespace(self, text: str) -> str:
        cleaned_lines = []
        for line in text.split("\n"):
            # Preserve line boundaries, only collapse repeated spaces/tabs within a line
            cleaned_line = re.sub(r"[ \t]{2,}", " ", line)
            cleaned_lines.append(cleaned_line)
        return "\n".join(cleaned_lines)
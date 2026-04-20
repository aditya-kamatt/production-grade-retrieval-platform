import os
import hashlib
import time 
from app.ingestion.models import DiscoveredFile 
from typing import List, Optional, Iterator, Set
from pathlib import Path 
from app.ingestion.models import DiscoveredFile


SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".csv"}

class FileDiscovery:

    def __init__(
        self,
        root_dir: str,
        supported_extensions: Optional[set] = None,
        ignore_hidden: bool = True,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.supported_extensions = supported_extensions or SUPPORTED_EXTENSIONS
        self.ignore_hidden = ignore_hidden

    def discover(self) -> List[DiscoveredFile]:
        """
        Recursively discover files and return metadata + hashes.
        """

        discovered_files = []

        for file_path in self._walk_directory():
            try:
                discovered_files.append(self._build_metadata(file_path))
            except Exception as e:
                print(f"[WARN] Failed processing {file_path}: {e}")
        
        return discovered_files

    def _walk_directory(self) -> Iterator[Path]:
        """
        Generator that yields valid file paths.
        """

        for root, dirs, files in os.walk(self.root_dir):
            if self.ignore_hidden:
                dirs[:] = [dir_name for dir_name in dirs if not dir_name.startswith(".")]

            for file_name in files:
                if self.ignore_hidden and file_name.startswith("."):
                    continue

                file_path = Path(root) / file_name

                if not self._is_supported(file_path):
                    continue

                yield file_path

    def _is_supported(self, file_path: Path) -> bool:
        return file_path.suffix.lower() in self.supported_extensions

    def _build_metadata(self, file_path: Path) -> DiscoveredFile:
        """
        Extract metadata + compute content hash.
        """

        stat = file_path.stat()
        content_hash = self._compute_hash(file_path)

        return DiscoveredFile(
            doc_id=content_hash,
            path=str(file_path.resolve()),
            file_type=file_path.suffix.lower().replace(".", ""),
            size_bytes=stat.st_size,
            modified_time=stat.st_mtime,
            content_hash=content_hash,
        )

    def _compute_hash(self, file_path: Path, chunk_size: int = 8192) -> str:
        """
        Compute SHA256 hash of file content
        """

        sha256 = hashlib.sha256()

        with open(file_path, "rb") as f:
            while chunk := f.read(chunk_size):
                sha256.update(chunk)

        return sha256.hexdigest()

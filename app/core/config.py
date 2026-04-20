from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    processed_dir: Path = Path("data/processed")
    sqlite_db_path: Path = Path("data/processed/metadata.db")
    faiss_index_path: Path = Path("data/processed/vector.index")
    faiss_id_map_path: Path = Path("data/processed/vector_ids.json")
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dimension: int = 384


settings = Settings()
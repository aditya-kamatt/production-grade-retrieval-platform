from pathlib import Path

from app.ingestion.discovery import FileDiscovery


def test_discovery_finds_supported_files(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("hello", encoding="utf-8")
    (tmp_path / "b.md").write_text("# title", encoding="utf-8")
    (tmp_path / "c.csv").write_text("col1,col2\nx,y", encoding="utf-8")
    (tmp_path / "ignore.json").write_text("{}", encoding="utf-8")

    files = FileDiscovery(root_dir=str(tmp_path)).discover()

    assert len(files) == 3
    file_types = sorted(file.file_type for file in files)
    assert file_types == ["csv", "md", "txt"]


def test_discovery_ignores_hidden_files_and_dirs(tmp_path: Path) -> None:
    (tmp_path / ".hidden.txt").write_text("secret", encoding="utf-8")
    hidden_dir = tmp_path / ".private"
    hidden_dir.mkdir()
    (hidden_dir / "inside.txt").write_text("secret", encoding="utf-8")
    (tmp_path / "visible.txt").write_text("ok", encoding="utf-8")

    files = FileDiscovery(root_dir=str(tmp_path), ignore_hidden=True).discover()

    assert len(files) == 1
    assert files[0].file_type == "txt"
    assert files[0].path.endswith("visible.txt")


def test_discovery_hash_is_stable_for_same_content(tmp_path: Path) -> None:
    file_path = tmp_path / "sample.txt"
    file_path.write_text("same content", encoding="utf-8")

    discovery = FileDiscovery(root_dir=str(tmp_path))
    files_first = discovery.discover()
    files_second = discovery.discover()

    assert len(files_first) == 1
    assert len(files_second) == 1
    assert files_first[0].content_hash == files_second[0].content_hash
    assert files_first[0].doc_id == files_second[0].doc_id


def test_discovery_hash_changes_when_content_changes(tmp_path: Path) -> None:
    file_path = tmp_path / "sample.txt"
    file_path.write_text("first", encoding="utf-8")

    discovery = FileDiscovery(root_dir=str(tmp_path))
    first_hash = discovery.discover()[0].content_hash

    file_path.write_text("second", encoding="utf-8")
    second_hash = discovery.discover()[0].content_hash

    assert first_hash != second_hash
from app.rag.document_loader import DocumentLoader, clean_text


def test_text_loader_preserves_filename(tmp_path) -> None:
    path = tmp_path / "notes.txt"
    path.write_text("First  line.\n\n\nSecond line.", encoding="utf-8")

    documents = DocumentLoader().load(path)

    assert len(documents) == 1
    assert documents[0].filename == "notes.txt"
    assert documents[0].page is None
    assert documents[0].text == "First line.\n\nSecond line."


def test_clean_text_removes_null_and_extra_spacing() -> None:
    assert clean_text(" A\x00   sentence. \r\n Next. ") == "A sentence.\nNext."

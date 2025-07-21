from unittest.mock import patch

import pytest


@pytest.fixture
def document_dir(tmp_path):
    """Fixture to create a temporary document directory."""
    dir_path = tmp_path / "documents"
    dir_path.mkdir()
    # Create some sample documents
    sample_docs = [
        "contracts/acme/2023/Contract with Acme Corp.pdf",
        "invoices/beta/2022/Beta Ltd Invoice",
        "reports/alpha/Project Alpha summary",
    ]
    for doc in sample_docs:
        (dir_path / doc).parent.mkdir(parents=True)
        (dir_path / doc).touch()
    with patch("found.user_cache_dir", return_value=dir_path):
        yield dir_path


def test_doc(document_dir):
    """Test the document finder functionality."""
    from found import DocumentFinder

    app = DocumentFinder()
    documents = app.get_documents(document_dir)

    assert len(documents) > 0, "No documents found."

    # Test building the index
    index = app.build_index(documents=documents)
    assert index.ntotal > 0, "Index should contain some vectors."

    # Test finding candidates
    query = "Agreement with some firm"
    num_candidates = 1
    candidates = app.find_candidates(query, num_candidates, document_dir)

    assert len(candidates) > 0, "No candidates found for the query."
    assert candidates[0].startswith(
        f"{document_dir}/contracts/acme/2023/Contract with Acme Corp.pdf (distance: "
    ), "The first candidate should match the expected document."

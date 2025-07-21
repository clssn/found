import numpy as np
from sentence_transformers import SentenceTransformer
import faiss


def main() -> None:
    # Load model
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Get metadata
    documents = get_documents()

    # Create embeddings
    doc_embeddings: np.ndarray = model.encode(documents)

    # Create a FAISS index
    index: faiss.IndexFlatL2 = faiss.IndexFlatL2(doc_embeddings.shape[1])
    index.add(doc_embeddings)

    # Embed user query
    query = "Agreement with Acme Corporation"
    query_vector: np.ndarray = model.encode([query])

    # Find closest document
    D, I = index.search(query_vector, k=1)
    match: str = documents[I[0][0]]

    print(f"Query: {query}")
    print(f"Best match: {match}")


def get_documents() -> list[str]:
    """
    This function should return a list of document metadata entries. For
    demonstration purposes, we will return a static list. In a real
    application, this could be replaced with a database query or file system
    scan.
    """
    return [
        "contracts/acme/2023 - Contract with Acme Corp",
        "invoices/beta/2022 - Beta Ltd Invoice",
        "reports/alpha - Project Alpha summary",
    ]

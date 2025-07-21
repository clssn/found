import functools
import hashlib
import importlib
import json
import logging
import os
import re
from pathlib import Path
from typing_extensions import Annotated

import faiss
import numpy as np
import typer
from platformdirs import user_cache_dir
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)
logger.setLevel(logging.FATAL)
handler = logging.StreamHandler(stream=os.sys.stdout)
formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
handler.setFormatter(formatter)
logger.handlers.clear()
logger.addHandler(handler)


def cache_index(cachedir, cache_key_arg):
    """Decorator to cache the FAISS index returned by the decorated function.

    Hash and index will be stored in files in the specified `cachedir`.

    This decorator will hash the content of the decorated function's argument
    specified with the `cache_key_arg` decorator argument. It will check if the
    hash matches the previously stored hash. If it matches and the index file
    exists, it will load the index from the cache. If it does not match or the
    index file does not exist, it will call the decorated function to build the
    index, save it to the cache, and update the hash.

    Raises:
        ValueError: If `cache_key_arg` is not found in the function's kwargs.
        OSError: If the index or hash are inaccessible.
    """

    def decorator(func):
        index_path = os.path.join(cachedir, "faiss.index")
        hash_path = os.path.join(cachedir, "filelist.hash.json")

        def content_hash(obj) -> str:
            obj_bytes = json.dumps(obj, sort_keys=True).encode("utf-8")
            return hashlib.sha256(obj_bytes).hexdigest()

        def load_hash():
            if os.path.exists(hash_path):
                with open(hash_path, "r") as f:
                    return json.load(f).get("hash")
            return None

        def save_hash(h):
            with open(hash_path, "w") as f:
                json.dump({"hash": h}, f)

        def save_index(index):
            faiss.write_index(index, index_path)

        def load_index():
            return faiss.read_index(index_path)

        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> faiss.IndexFlatL2:
            try:
                hash_value = content_hash(kwargs[cache_key_arg])

            except KeyError:
                logger.error(f"cache_key_arg {cache_key_arg} not found in kwargs.")
                raise ValueError("cache_key_arg must be provided in kwargs.")
            logger.debug(f"Hash over {cache_key_arg} data: {hash_value}.")
            cached_hash = load_hash()
            if cached_hash == hash_value and Path(index_path).is_file():
                logger.debug("Restoring index from cache.")
                return load_index()
            else:
                logger.debug("Building index.")
                index = func(*args, **kwargs)
                save_index(index)
                save_hash(hash_value)
                return index

        return wrapper

    return decorator


class DocumentFinder:
    appname: str = "found"

    def __init__(self):
        self.cachedir = user_cache_dir(self.appname)
        os.makedirs(self.cachedir, exist_ok=True)
        self.index_path = os.path.join(self.cachedir, "faiss.index")
        self.hash_path = os.path.join(self.cachedir, "faiss.hash.json")
        self.logger = logger
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def get_documents(self, document_base: Path) -> list[str]:
        documents: list[str] = []
        for path in document_base.rglob("*"):
            if path.is_file():
                rel_dir = str(path.relative_to(document_base))
                fname_parts = re.sub(r"\W+", " ", str(path))
                entry = f"{rel_dir} - {fname_parts}"
                documents.append(entry)
        logger.debug(f"Found {len(documents)} documents.")
        return documents

    @cache_index(user_cache_dir(appname), cache_key_arg="documents")
    def build_index(self, *, documents: list[str]) -> faiss.IndexFlatL2:
        doc_embeddings: np.ndarray = self.model.encode(documents)
        index: faiss.IndexFlatL2 = faiss.IndexFlatL2(doc_embeddings.shape[1])
        index.add(doc_embeddings)
        return index

    def find_candidates(
        self, query: str, num_candidates: int, documents_base: Path
    ) -> str:
        """Find the best candidating documents for the given query."""
        documents = self.get_documents(documents_base)
        index: faiss.IndexFlatL2 = self.build_index(documents=documents)
        query_vector: np.ndarray = self.model.encode([query])
        D, I = index.search(query_vector, k=num_candidates)
        candidates = [documents[I[0][i]] for i in range(I.shape[1])]
        return candidates


cli = typer.Typer()


@cli.command()
def version():
    """Show the version of pyfound."""
    try:
        ver = importlib.metadata.version("pyfound")
    except importlib.metadata.PackageNotFoundError:
        ver = "unknown (package not installed)"
    print(f"pyfound version {ver}")


@cli.command()
def doc(
    query: str = typer.Argument(..., help="Search query for your documents."),
    num_candidates: Annotated[
        int,
        typer.Option(
            "--candidates",
            "-n",
            help="Number of candidate documents to consider.",
        ),
    ] = 5,
    document_dir: Annotated[
        str,
        typer.Option(
            "--document-dir",
            "-d",
            help="Directory containing documents.",
        ),
    ] = str(Path("~/Documents").expanduser()),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable debug logging."
    ),
):
    """Search for the best matching document candidates."""
    if verbose:
        logger.setLevel(logging.DEBUG)
    app = DocumentFinder()
    candidates = app.find_candidates(query, num_candidates, Path(document_dir))

    print("Found! candidates:\n" + "\n".join(candidates))


if __name__ == "__main__":
    cli()

# Found!

_Found!_ is a command-line tool for fast semantic search over your personal documents using vector embeddings and FAISS. It is designed to be simple, fast, and privacy-friendly, running entirely on your machine.

## Features
- Semantic search for documents using natural language queries
- Fast similarity search powered by FAISS
- Caches document embeddings and index for speed
- Rich CLI interface with Typer

## Installation and Usage with uv/uvx

_Found!_ is compatible with [uv](https://github.com/astral-sh/uv) and [uvx](https://github.com/astral-sh/uvx) for fast Python package management and execution.


### 1. Install uv (if not available)

See the official [uv installation instructions](https://docs.astral.sh/uv/getting-started/installation/#standalone-installer) for the recommended standalone installer and platform-specific details.

### 2. Install _found_ with uv

```bash
uv tool install --git https://github.com/clssn/found.git
```

## Usage

### Search for a document

```bash
found doc "your query"
```

#### Options
- `--document-dir`, `-d`: Specify the directory containing documents (default: `~/Documents`)
- `--verbose`, `-v`: Enable debug logging

#### Example
```bash
found doc "Tax certificate 2024" -d ~/Documents
```

## How it works
- Recursively lists files in the specified document directory
- Generates semantic embeddings for each document using SentenceTransformers
- Builds a FAISS index for fast similarity search
- Caches the index and document list for future queries
- Returns the best matching document for your query

## Requirements
- Python 3.8+
- [faiss](https://github.com/facebookresearch/faiss)
- [sentence-transformers](https://www.sbert.net/)
- [typer](https://typer.tiangolo.com/)
- [platformdirs](https://github.com/platformdirs/platformdirs)

## License
MIT

## Contributing
Pull requests and issues are welcome!

---

Made with ❤️ by clssn

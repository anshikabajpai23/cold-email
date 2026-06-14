from pathlib import Path

import pypdf
from llama_index.core import (
    Document,
    Settings,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

DOCS_DIR = Path(__file__).parent / "docs"
CACHE_DIR = Path(__file__).parent / "cache"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"


def _configure():
    Settings.embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL)
    Settings.llm = None


def _read_docs() -> list[Document]:
    docs = []
    for path in sorted(DOCS_DIR.iterdir()):
        if path.suffix.lower() == ".pdf":
            reader = pypdf.PdfReader(str(path))
            text = "\n".join(
                page.extract_text() or "" for page in reader.pages
            ).strip()
            if text:
                docs.append(Document(text=text, metadata={"filename": path.name}))
        elif path.suffix.lower() in {".txt", ".md"}:
            text = path.read_text(errors="ignore").strip()
            if text:
                docs.append(Document(text=text, metadata={"filename": path.name}))
    if not docs:
        raise FileNotFoundError("No readable text found in docs/. Check your files.")
    return docs


def build_index() -> VectorStoreIndex:
    if not DOCS_DIR.exists() or not any(DOCS_DIR.iterdir()):
        raise FileNotFoundError(f"No docs found in {DOCS_DIR}.")
    _configure()
    documents = _read_docs()
    index = VectorStoreIndex.from_documents(documents)
    CACHE_DIR.mkdir(exist_ok=True)
    index.storage_context.persist(persist_dir=str(CACHE_DIR))
    return index


def load_index() -> VectorStoreIndex:
    _configure()
    if not CACHE_DIR.exists() or not any(CACHE_DIR.iterdir()):
        return build_index()
    storage_context = StorageContext.from_defaults(persist_dir=str(CACHE_DIR))
    return load_index_from_storage(storage_context)


def query_background(query: str, top_k: int = 6) -> str:
    index = load_index()
    retriever = index.as_retriever(similarity_top_k=top_k)
    nodes = retriever.retrieve(query)
    return "\n\n".join(n.get_content() for n in nodes)

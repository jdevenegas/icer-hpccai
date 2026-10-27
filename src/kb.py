import os
from typing import List

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document

from .md_reader import PreProcMd
from .config import DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP


def conditional_split(docs: List[Document], chunk_size: int, chunk_overlap: int) -> List[Document]:
    """
    Conditionally apply chunking only to documents longer than chunk_size.
    Docs <= chunk_size pass through unchanged (1 chunk).
    Docs > chunk_size are split using RecursiveCharacterTextSplitter with overlap.

    Params:
        docs: Input list of Documents (typically already structurally split).
        chunk_size: Target max characters per chunk.
        chunk_overlap: Number of overlapping characters between consecutive chunks.

    Returns:
        A list of Documents where longer inputs are expanded into multiple chunks.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
    )

    out = []

    for doc in docs:
        if len(doc.page_content) <= chunk_size:
            out.append(doc)
        else:
            out.extend(splitter.split_documents([doc]))

    return out


def inject_md_context(docs: List[Document]) -> List[Document]:
    """
    Prepend lightweight metadata context into each Document's page_content.

    Params:
        docs: Documents whose metadata contains markdown structural fields.

    Returns:
        A list of Documents with updated page_content that includes a context prefix.
    """
    out = []

    for doc in docs:
        md = doc.metadata or {}

        source = os.path.basename(str(md.get("source", "")))
        title = md.get("title", "")
        section = md.get("section", "")
        subsection = md.get("subsection", "")
        tags = md.get("tags", [])

        header_line = " | ".join(
            [x for x in [title, section, subsection] if x]
        ).strip()

        tags_line = ", ".join(tags) if isinstance(tags, list) else ""

        prefix_parts = []

        if source:
            prefix_parts.append(f"source: {source}")

        if header_line:
            prefix_parts.append(f"heading: {header_line}")

        if tags_line:
            prefix_parts.append(f"tags: {tags_line}")

        if prefix_parts:
            prefix = "\n".join(prefix_parts) + "\n\n"
            doc.page_content = prefix + doc.page_content

        out.append(doc)

    return out


def load_document(fpath: str) -> List[Document]:
    """
    Load one local document file.
    Supported: PDF, Markdown, TXT.
    """
    ext = os.path.splitext(fpath)[-1].lower()

    if ext == ".pdf":
        loader = PyPDFLoader(fpath)
        docs = loader.load()

    elif ext in [".md", ".markdown"]:
        repo_path = os.path.dirname(fpath)
        splits = PreProcMd(repo_path).run()
        target = os.path.abspath(fpath)

        docs = [
            d for d in splits
            if os.path.abspath(str(d.metadata.get("source", ""))) == target
        ]

    elif ext == ".txt":
        loader = TextLoader(fpath, encoding="utf-8")
        docs = loader.load()

    else:
        raise ValueError(f"Unsupported file type: {fpath}")

    print(f"Loaded {len(docs)} docs from {fpath}")
    return docs


def build_knowledge_base(
    embed_model: str,
    file_paths=None,
    docs=None,
):
    """
    Build the vector knowledge base from either:

    1. local files
    2. already-created LangChain Documents

    """

    all_docs = []

    # Normal docs mode
    if file_paths is not None:
        for path in file_paths:
            all_docs.extend(load_document(path))

    # Benchmark / pre-created Documents mode
    elif docs is not None:
        all_docs = docs

    else:
        raise ValueError(
            "You must provide either file_paths or docs."
        )

    all_docs = inject_md_context(all_docs)

    splits = conditional_split(
        all_docs,
        chunk_size=DEFAULT_CHUNK_SIZE,
        chunk_overlap=DEFAULT_CHUNK_OVERLAP,
    )

    print(f"Split into {len(splits)} chunks.")

    embeddings = OllamaEmbeddings(
        model=embed_model
    )

    vector_store = InMemoryVectorStore(
        embeddings
    )

    vector_store.add_documents(
        splits
    )

    print("Knowledge base ready.")

    return vector_store
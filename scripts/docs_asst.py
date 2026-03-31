import os
import sys
import argparse
from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_chroma import Chroma

# ==========================================
EMBED_MODEL = "mxbai-embed-large:latest"
#LLM_MODEL = "gpt-oss:20b"
LLM_MODEL = "llama3.2:3b"
# ==========================================


def load_document(file_path: str):
    ext = os.path.splitext(file_path)[-1].lower()

    if ext == ".pdf":
        loader = PyPDFLoader(file_path)
    elif ext in [".md", ".markdown", ".txt"]:
        loader = TextLoader(file_path, encoding="utf-8")
    else:
        raise ValueError(f"Unsupported file type: {file_path}")

    docs = loader.load()
    print(f"Loaded {len(docs)} docs from {file_path}")
    return docs

def build_knowledge_base(file_paths: str):
    # print(f"Loading PDF: {file_path}")
    # loader = PyPDFLoader(file_path)
    # docs = loader.load()
    # print(f"Loaded {len(docs)} pages.")
    all_docs = []
    for path in file_paths:
        all_docs.extend(load_document(path))

    # print(f"Total documents: {len(all_docs)}")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, add_start_index=True
    )
    splits = text_splitter.split_documents(all_docs)
    print(f"Split into {len(splits)} chunks.")

    embeddings = OllamaEmbeddings(model=EMBED_MODEL)
    vector_store = InMemoryVectorStore(embeddings)
    vector_store.add_documents(splits)

    print("Knowledge base ready")
    return vector_store


def answer_query(vector_store, query: str):
    print(f"\nQuery: {query}\n")

    # Retrieve relevant chunks
    results = vector_store.similarity_search_with_score(query, k=4)
    print(results)
    # Sort descending by similarity (higher --> similar)
    results = sorted(results, key=lambda x: x[1], reverse=True)
    top_doc, top_score = results[0]

    top_docs = [doc.page_content for doc, _ in results]

    # Combine retrieved context
    context = "\n\n".join(top_docs)

    # Build a simple prompt
    prompt = ChatPromptTemplate.from_template("""
You are a helpful faq/documentation assistant that answers questions using the provided document context.

<context>
{context}
</context>

Question: {query}

Answer clearly and concisely:
""")

    formatted_prompt = prompt.format(context=context, query=query)

    llm = ChatOllama(model=LLM_MODEL)
    response = llm.invoke(formatted_prompt)
    # print(f"{response.content}\n")
    print(f"Answer:\n{response.content}\n")
    print(f"Reference (score={top_score:.4f}):\n{top_doc.page_content.strip()[:800]}...\n")



def main():
    parser = argparse.ArgumentParser(
        description="Local faq chatbot w/ Ollama embeddings/llms + local docs"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default=".",
        help="Directory where your document files are located (default: current dir).",
    )
    parser.add_argument(
        "--persist_dir",
        type=str,
        default=None,
        help="Path to existing Chroma vector store)",
    )
    parser.add_argument(
        "files",
        nargs="*",
        help="List of document filenames (e.g., report.pdf notes.md policy.txt)",
    )

    args = parser.parse_args()

    if args.persist_dir:
        embeddings = OllamaEmbeddings(model=EMBED_MODEL)
        print(f"Loading existing vector store from {args.persist_dir}")
        vector_store = Chroma(
            collection_name="icer_docs_collection_full",
            embedding_function=embeddings,
            persist_directory=args.persist_dir,
        )
        print(vector_store._collection.get(include=["embeddings"]))
    else:
        # Combine the data_path and filenames
        # file_paths = [os.path.join(args.data_path, f) for f in args.files]
        if args.files:
            # Use only specified files
            file_paths = [os.path.join(args.data_path, f) for f in args.files]
        else:
            # Load all supported files from data_path
            supported_exts = (".pdf", ".md", ".markdown", ".txt")
            file_paths = [
                os.path.join(args.data_path, f)
                for f in os.listdir(args.data_path)
                if f.lower().endswith(supported_exts)
            ]

        # Verify files exist
        for path in file_paths:
            if not os.path.exists(path):
                raise FileNotFoundError(f"File not found: {path}")

        vector_store = build_knowledge_base(file_paths)

    print("\nAsk your questions (type 'exit' to quit):\n")
    while True:
        query = input("User: ")
        if query.strip().lower() in {"exit", "quit"}:
            print("Adios!")
            break
        answer_query(vector_store, query)


if __name__ == "__main__":
    main()

from pathlib import Path
from typing import List, Dict, Tuple, Any
import yaml
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter

def extract_front(text: str) -> Tuple[Dict[str, Any], str]:
    """
    Extract YAML front matter from a markdown-like text blob.

    Params:
        text: Full raw file text. If it starts with '---', this function tries to parse
              YAML until the next '\n---' delimiter.

    Returns:
            front_matter: Dict parsed from YAML.
            body: Remaining text after the front matter block.
    """
    if not text.startswith("---"):
        return {}, text
    end = text.find("\n---", 3)
    if end == -1:
        return {}, text
    yaml_text = text[3:end].strip()
    body = text[end + 4:].lstrip("\n")
    try:
        data = yaml.safe_load(yaml_text)
        if not isinstance(data, dict):
            data = {}
    except Exception:
        data = {}
    return data, body


##Preprocessing
class PreProcMd:
    def __init__(self, repo_path: str):
        """
        Initialize a markdown preprocessing pipeline.
        Params:
            repo_path: Directory containing markdown files to ingest.
        """
        self.repo_path = Path(repo_path)
        self.headers_to_split_on=[("#", "title"),("##", "section"),("###", "subsection"),]

##Load md
    def loaddoc(self) -> List[Document]:
        """
        Load markdown files from repo_path into LangChain Document objects.

        Each file becomes one Document before structural splitting.
        Metadata includes:
          - source
          - doc_type
          - YAML front matter fields ]
          - tags normalized to a list

        Returns:
            A list of Documents containing raw markdown body text and metadata.
        """
        if not self.repo_path.exists():
            raise FileNotFoundError(f"Path not found: {self.repo_path}")
        if not self.repo_path.is_dir():
            raise ValueError(f"Expected a directory path, got: {self.repo_path}")
        docs = []
        for i in list(self.repo_path.rglob("*.md")) + list(self.repo_path.rglob("*.markdown")):
            text = i.read_text(encoding="utf-8", errors="replace")
            front_matter, body = extract_front(text)
            metadata = { "source": str(i),"doc_type": "markdown",}
            metadata.update(front_matter)
            tags = metadata.get("tags", [])
            if isinstance(tags, str):
                tags = [tags]
            elif not isinstance(tags, list):
                tags = []
            metadata["tags"] = tags
            if body.strip() == "":
                continue 
            docs.append(Document(page_content=body,metadata=metadata))
        return docs

##Splitting
    def splitdoc(self, docs: List[Document]) -> List[Document]:
        """
        Structurally split markdown Documents by headers into smaller Documents.

        Uses MarkdownHeaderTextSplitter, preserving original headers if strip_headers=False.
        Each resulting chunk merges:
          - original doc metadata 
          - chunk metadata produced by the splitter 

        Params:
            docs: List of Documents to structurally split.

        Returns:
            A list of structurally split Documents.
        """
        splitter=MarkdownHeaderTextSplitter(headers_to_split_on=self.headers_to_split_on,strip_headers=False)
        chunks = []
        for i in docs:
            try:
                splits = splitter.split_text(i.page_content)
            except Exception:
                splits = [Document(page_content=i.page_content)]
            for chunk in splits:
                new_metadata = dict(i.metadata)
                new_metadata.update(chunk.metadata)
                chunk.metadata = new_metadata
                chunks.append(chunk)
        return chunks

##Pipeline
    def run(self) -> List[Document]:
        docs = self.loaddoc()
        splits = self.splitdoc(docs)
        return splits



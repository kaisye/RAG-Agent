from langchain_text_splitters import RecursiveCharacterTextSplitter


class RecursiveChunker:
    def __init__(self, chunk_size: int = 1024, chunk_overlap: int = 128):
        self.splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", " ", ""],
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )

    def split(self, text_blocks: list[dict], document_id: str) -> list[dict]:
        chunks: list[dict] = []
        for block in text_blocks:
            for sub in self.splitter.split_text(block["text"]):
                chunks.append({
                    "chunk_id": f"{document_id}_rc_{len(chunks):04d}",
                    "document_id": document_id,
                    "page": block["page"],
                    "text": sub,
                    "strategy": "recursive",
                })
        return chunks

# document_loader.py
import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

load_dotenv()

# Minimal Document class
class Document:
    def __init__(self, page_content, metadata=None):
        self.page_content = page_content
        self.metadata = metadata or {}

# Minimal text splitter
class SimpleTextSplitter:
    def __init__(self, chunk_size=1000, chunk_overlap=200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_documents(self, documents):
        chunks = []
        for doc in documents:
            text = doc.page_content
            start = 0
            while start < len(text):
                end = start + self.chunk_size
                chunk_text = text[start:end]
                chunks.append(Document(page_content=chunk_text, metadata=doc.metadata))
                start += self.chunk_size - self.chunk_overlap
        return chunks

class DocumentProcessor:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings(
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        self.text_splitter = SimpleTextSplitter()
        self.vector_store = []

    def load_documents_from_directory(self, directory_path):
        documents = []
        for filename in os.listdir(directory_path):
            filepath = os.path.join(directory_path, filename)
            if filename.endswith(".txt"):
                with open(filepath, "r") as f:
                    text = f.read()
                documents.append(Document(page_content=text, metadata={"source": filename}))
            elif filename.endswith(".pdf"):
                try:
                    import PyPDF2
                    reader = PyPDF2.PdfReader(filepath)
                    text = "\n".join(page.extract_text() or "" for page in reader.pages)
                    documents.append(Document(page_content=text, metadata={"source": filename}))
                except ImportError:
                    print("PyPDF2 not installed; skipping PDF:", filename)
        return documents

    def process_documents(self, documents):
        return self.text_splitter.split_documents(documents)

    def add_documents_to_vectorstore(self, directory_path):
        print("Loading documents...")
        documents = self.load_documents_from_directory(directory_path)
        print(f"Loaded {len(documents)} documents")

        print("Processing documents into chunks...")
        chunks = self.process_documents(documents)
        print(f"Created {len(chunks)} chunks")

        print("Embedding documents...")
        self.vector_store = [(self.embeddings.embed_query(chunk.page_content), chunk) for chunk in chunks]
        print("Done!")

    def retrieve(self, query, top_k=3):
        query_vector = self.embeddings.embed_query(query)
        scored_docs = []
        for vec, doc in self.vector_store:
            score = sum(q*v for q, v in zip(query_vector, vec))
            scored_docs.append((score, doc))
        scored_docs.sort(key=lambda x: x[0], reverse=True)
        return [doc for _, doc in scored_docs[:top_k]]
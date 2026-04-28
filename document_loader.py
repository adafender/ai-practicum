import os
import pickle
import faiss
import numpy as np
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

load_dotenv()

class Document:
    def __init__(self, page_content, metadata=None):
        self.page_content = page_content
        self.metadata = metadata or {}

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

        self.index = None
        self.documents = []

        self.index_path = "faiss.index"
        self.docs_path = "docs.pkl"

        self._load_index()

    def _load_index(self):
        if os.path.exists(self.index_path) and os.path.exists(self.docs_path):
            print("Loading existing FAISS index...")
            self.index = faiss.read_index(self.index_path)
            with open(self.docs_path, "rb") as f:
                self.documents = pickle.load(f)
            print("FAISS index loaded!")

    def _save_index(self):
        faiss.write_index(self.index, self.index_path)
        with open(self.docs_path, "wb") as f:
            pickle.dump(self.documents, f)

    def load_documents_from_directory(self, directory_path):
        documents = []
        for filename in os.listdir(directory_path):
            filepath = os.path.join(directory_path, filename)

            if filename.endswith(".txt"):
                with open(filepath, "r", encoding="utf-8") as f:
                    text = f.read()
                documents.append(Document(page_content=text, metadata={"source": filename}))

            elif filename.endswith(".pdf"):
                try:
                    import PyPDF2
                    reader = PyPDF2.PdfReader(filepath)

                    text = ""
                    for page in reader.pages:
                        extracted = page.extract_text()
                        if extracted:
                            text += extracted

                    if text.strip():
                        documents.append(Document(page_content=text, metadata={"source": filename}))
                    else:
                        print(f"Skipping empty PDF: {filename}")

                except Exception as e:
                    print(f"Error reading PDF {filename}: {e}")

        return documents

    def process_documents(self, documents):
        return self.text_splitter.split_documents(documents)

    def add_documents_to_vectorstore(self, directory_path):
        documents = self.load_documents_from_directory(directory_path)
        chunks = self.process_documents(documents)

        print(f"Embedding {len(chunks)} chunks...")

        vectors = [self.embeddings.embed_query(chunk.page_content) for chunk in chunks]
        vectors = np.array(vectors).astype("float32")

        if self.index is None:
            dimension = vectors.shape[1]
            self.index = faiss.IndexFlatL2(dimension)

        self.index.add(vectors)
        self.documents.extend(chunks)

        self._save_index()
        print("Saved FAISS index!")

    def retrieve(self, query, top_k=3):
        if self.index is None or len(self.documents) == 0:
            return []

        query_vector = np.array([self.embeddings.embed_query(query)]).astype("float32")

        distances, indices = self.index.search(query_vector, top_k)

        results = []
        for idx in indices[0]:
            if idx < len(self.documents):
                results.append(self.documents[idx])

        return results
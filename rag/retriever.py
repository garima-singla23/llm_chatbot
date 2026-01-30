import os
import numpy as np
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

VECTOR_PATH = "rag/vectorstore"

def build_vectorstore():
    docs = []
    for file in os.listdir("data"):
        if file.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join("data",file))
            docs.extend(loader.load())
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=50)
    split_docs = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")

    vectorstore = FAISS.from_documents(split_docs,embeddings)
    vectorstore.save_local(VECTOR_PATH)

def retrieve_docs(query, k=1):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vectorstore = FAISS.load_local(VECTOR_PATH, embeddings,allow_dangerous_deserialization=True)
    results = vectorstore.similarity_search(query, k=k)
    semantic_docs = vectorstore.similarity_search(query, k=3)
    if not semantic_docs:
        return []
    semantic_texts = [doc.page_content for doc in semantic_docs]
    vectorizer = TfidfVectorizer(stop_words="english")
    X = vectorizer.fit_transform(semantic_texts)
    query_vec = vectorizer.transform([query])
    similarities = cosine_similarity(X, query_vec).flatten()

    vectorizer = TfidfVectorizer(stop_words="english")
    X = vectorizer.fit_transform(semantic_texts)
    query_vec = vectorizer.transform([query])

    verified_docs = []
    for idx, score in enumerate(similarities):
        if score >= k:
            verified_docs.append(semantic_texts[idx])

    return verified_docs[:k]


                                
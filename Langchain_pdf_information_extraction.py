#!/usr/bin/env python
# coding: utf-8

"""
LangChain PDF Information Extraction Pipeline
Author: Anuraj Ramesh
Description: Load a PDF, split it into chunks, embed, store in a vector database, 
and retrieve relevant information using OpenAI embeddings and LLM compression retriever.
"""

import os
from dotenv import load_dotenv, find_dotenv

# Load environment variables
_ = load_dotenv(find_dotenv())
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

if OPENAI_API_KEY is None:
    raise ValueError("OpenAI API key not found in environment variables.")
    
# LangChain imports
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.llms import OpenAI

# -----------------------------
# Configurable Parameters
# -----------------------------
PDF_PATH = "docs/000_MaterialModelsforPolymersunderCrashLoadsExistingLS-DYNAModelsandPerspective.pdf"
CHROMA_PERSIST_DIR = "docs/chroma/"
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 150
TOP_K = 2
FETCH_K = 3
LLM_MODEL = "gpt-3.5-turbo-instruct"
TEMPERATURE = 0.0

# -----------------------------
# Step 1: Load PDF
# -----------------------------
def load_pdf(pdf_path, pages_limit=None):
    """Load PDF and optionally limit number of pages."""
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    if pages_limit:
        pages = pages[:pages_limit]
    return pages

# -----------------------------
# Step 2: Split Documents
# -----------------------------
def split_documents(pages, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    """Split PDF pages into smaller chunks for embeddings."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "(?<=\. )", " ", ""]
    )
    return splitter.split_documents(pages)

# -----------------------------
# Step 3: Create Vectorstore
# -----------------------------
def create_vectordb(splits, embedding_model, persist_dir=CHROMA_PERSIST_DIR):
    """Embed documents and store in Chroma vector database."""
    vectordb = Chroma.from_documents(
        documents=splits,
        embedding=embedding_model,
        persist_directory=persist_dir
    )
    vectordb.persist()
    return vectordb

# -----------------------------
# Step 4: Retrieve Relevant Documents
# -----------------------------
def retrieve_documents(question, vectordb, top_k=TOP_K, fetch_k=FETCH_K):
    """Retrieve relevant documents using Maximal Marginal Relevance."""
    return vectordb.max_marginal_relevance_search(question, k=top_k, fetch_k=fetch_k)

# -----------------------------
# Step 5: Compress and Summarize
# -----------------------------
def compress_documents(docs, llm_model=LLM_MODEL, temperature=TEMPERATURE):
    """Compress retrieved documents using LLMChainExtractor."""
    llm = OpenAI(temperature=temperature, model=llm_model)
    compressor = LLMChainExtractor.from_llm(llm)
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=vectordb.as_retriever()
    )
    compressed_docs = compression_retriever.get_relevant_documents(question)
    return compressed_docs

# -----------------------------
# Step 6: Pretty Print
# -----------------------------
def pretty_print_docs(docs):
    """Nicely format documents for display."""
    for i, d in enumerate(docs):
        print(f"\n{'-'*80}\nDocument {i+1}:\n{d.page_content}\n")

# -----------------------------
# Main Execution
# -----------------------------
if __name__ == "__main__":
    print("Loading PDF...")
    pages = load_pdf(PDF_PATH, pages_limit=11)

    print("Splitting documents...")
    splits = split_documents(pages)
    print(f"Total splits created: {len(splits)}")

    print("Creating vector database...")
    embedding = OpenAIEmbeddings()
    vectordb = create_vectordb(splits, embedding)

    question = "What are the characteristics of the model developed by Junginger?"
    print(f"\nRetrieving documents relevant to: {question}")
    docs = retrieve_documents(question, vectordb)

    print("\nCompressing and summarizing documents...")
    compressed_docs = compress_documents(docs)

    print("\nDisplaying retrieved documents:")
    pretty_print_docs(compressed_docs)

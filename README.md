# LangChain PDF Information Extraction Pipeline

This project demonstrates how to extract, embed, and retrieve information from PDFs using **LangChain**, **OpenAI embeddings**, and **Chroma vector database**. The pipeline is modular, professional, and easily extendable for research or business applications.

## Features

- Load PDF documents and optionally limit pages.
- Split documents into chunks for efficient embedding.
- Embed documents with OpenAI embeddings.
- Store and persist embeddings in Chroma vector database.
- Retrieve relevant documents using **Maximal Marginal Relevance (MMR)**.
- Compress and summarize results using **LLMChainExtractor**.
- Clean, professional code structure with functions, docstrings, and configurable parameters.

## Installation
pip install langchain openai chromadb python-dotenv


## Usage

1. Set your OpenAI API key in .env:

OPENAI_API_KEY=your_openai_api_key_here


2. Update the PDF_PATH in main.py if using a different PDF.

Run the pipeline:

python Langchain_pdf_information_extraction.py


3. View retrieved and compressed document excerpts in the console output.


## Example Output
--------------------------------------------------------------------------------
Document 1:
The characteristics of the model developed by Marschall include...


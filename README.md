# Advanced RAG System

An end-to-end Retrieval-Augmented Generation (RAG) application that answers questions about PDF documents using hybrid retrieval, a local large language model (LLM), and source-aware responses.

## Overview

The Advanced RAG System allows users to upload PDF documents, retrieve relevant passages using semantic and keyword search, and generate answers based on the retrieved content.

It combines FAISS vector search, BM25 keyword retrieval, Reciprocal Rank Fusion (RRF), Sentence Transformers embeddings, and Ollama-powered text generation in an interactive Streamlit application.

## Features

- **PDF ingestion:** Extract text from PDF files using PyMuPDF.
- **Overlapping text chunking:** Split extracted text into manageable passages.
- **Semantic search:** Generate embeddings using `sentence-transformers/all-MiniLM-L6-v2`.
- **Keyword search:** Retrieve passages using BM25.
- **Hybrid retrieval:** Combine semantic and keyword rankings using Reciprocal Rank Fusion.
- **Local LLM:** Generate answers using Ollama and `qwen2.5:3b`.
- **Source tracking:** Preserve PDF filenames and page numbers for retrieved passages.
- **Interactive UI:** Upload PDFs and ask questions through Streamlit.
- **Automated tests:** Test retrieval validation, text chunking, and PDF ingestion.

## System Architecture

1. Upload one or more PDF documents.
2. Extract text and preserve page-level metadata.
3. Split the text into overlapping chunks.
4. Generate dense embeddings using Sentence Transformers.
5. Build a FAISS vector index and a BM25 keyword index.
6. Retrieve relevant passages using both search methods.
7. Combine rankings using Reciprocal Rank Fusion (RRF).
8. Provide retrieved passages to the local LLM as context.
9. Display the generated answer and retrieved source passages in Streamlit.

## Technology Stack

- Python 3.11
- Streamlit
- PyMuPDF
- Sentence Transformers
- FAISS
- BM25
- Reciprocal Rank Fusion (RRF)
- Ollama
- Qwen 2.5 3B
- NumPy
- unittest

## Project Structure

```text
Advanced-RAG-System/
├── app.py
├── retrieval.py
├── ingestion.py
├── generation.py
├── evaluation.py
├── requirements.txt
├── README.md
├── .gitignore
├── data/
└── tests/
    ├── test_retrieval.py
    └── test_ingestion.py
```

## Prerequisites

- Windows 10 or Windows 11
- Python 3.11
- Git
- Ollama
- The `qwen2.5:3b` model
- An internet connection for installing dependencies and downloading the embedding model on first use

The computer must have enough available memory and processing resources to run the embedding model and local LLM.

## Installation

### 1. Clone the repository

Open PowerShell and run:

```powershell
git clone https://github.com/ahsanhabibabir04/Advanced-RAG-System.git
cd Advanced-RAG-System
```

### 2. Create a virtual environment

```powershell
py -3.11 -m venv .venv
```

Activate it:

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### 3. Install Python dependencies

```powershell
python -m pip install -r requirements.txt
```

### 4. Prepare Ollama

Install Ollama if it is not already installed.

Download the required model:

```powershell
ollama pull qwen2.5:3b
```

Verify that the model is available:

```powershell
ollama list
```

Ensure the Ollama service is running before using the application.

### 5. Run the application

From the project directory, execute:

```powershell
python -m streamlit run app.py
```

Open the local URL shown in the terminal. By default, it is:

http://localhost:8501

## How to Use

1. Open the Streamlit application.
2. Upload one or more text-based PDF documents.
3. Click **Process PDFs**.
4. Wait for PDF extraction and hybrid search indexing to finish.
5. Enter a question about the uploaded documents.
6. Click **Get Answer**.
7. Read the generated response.
8. Expand **Retrieved sources** to inspect the retrieved passages, PDF filenames, and page numbers.

## Run Automated Tests

Run the complete test suite from the project directory:

```powershell
python -m unittest discover -s tests -v
```

The current test suite contains 12 tests covering text tokenization, chunk splitting, retrieval input validation, PDF text extraction, page metadata, and error handling.

## Current Limitations

- Scanned and image-only PDFs may require OCR, which is not currently implemented.
- Answer quality depends on the document content, retrieved passages, and LLM.
- Retrieved source passages provide traceability but do not guarantee that every generated claim is fully supported.
- The current application is designed to run locally and requires Ollama.
- Retrieval quality has not yet been evaluated using a dedicated benchmark dataset.

## Future Improvements

- Add OCR support for scanned PDFs.
- Evaluate retrieval performance using Recall@K, Mean Reciprocal Rank (MRR), and nDCG.
- Add a cross-encoder reranker.
- Improve conversational memory and document management.
- Add configurable retrieval weights and model selection.
- Expand automated tests for generation and end-to-end application behavior.

## Disclaimer

This project is intended for educational and portfolio purposes. Generated answers may be incomplete or incorrect. Always verify important information against the original documents.

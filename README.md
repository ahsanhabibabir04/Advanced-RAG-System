# Advanced RAG System

An end-to-end Retrieval-Augmented Generation (RAG) application that enables users to ask questions about PDF documents and receive context-grounded answers using hybrid information retrieval and a locally running Large Language Model (LLM).

The system integrates FAISS-based semantic search, BM25 keyword retrieval, Reciprocal Rank Fusion (RRF), and Ollama-powered answer generation to retrieve relevant document passages and generate answers with source references.

## Overview

The application processes PDF documents, extracts and chunks their text, builds a hybrid retrieval index, and uses retrieved passages as context for answer generation. The Streamlit interface provides an interactive workflow for document upload, processing, question answering, and source inspection.

## Key Features

- **PDF Ingestion:** Extract text from multiple PDF documents using PyMuPDF.
- **Text Chunking:** Split document text into overlapping chunks while preserving source filenames and page numbers.
- **Semantic Retrieval:** Generate document embeddings using Sentence Transformers.
- **Keyword Retrieval:** Retrieve relevant passages using BM25.
- **Hybrid Search:** Combine semantic and keyword rankings using Reciprocal Rank Fusion.
- **LLM Integration:** Generate context-grounded answers using the Qwen 2.5 3B model through Ollama.
- **Source Attribution:** Display retrieved passages with PDF filenames and page numbers.
- **Interactive Interface:** Upload documents and ask questions through Streamlit.
- **Retrieval Evaluation:** Implement Precision@K, Recall@K, Reciprocal Rank, and Mean Reciprocal Rank.
- **Automated Testing:** Validate ingestion, retrieval, and evaluation components using Python's `unittest` framework.

## System Architecture

```text
PDF Documents
      |
      v
Text Extraction (PyMuPDF)
      |
      v
Text Chunking and Source Metadata
      |
      v
Embedding Generation
(Sentence Transformers)
      |
      v
Hybrid Retrieval
      |
      +-------------------+
      |                   |
      v                   v
FAISS Semantic Search   BM25 Keyword Search
      |                   |
      +-------------------+
                |
                v
     Reciprocal Rank Fusion
                |
                v
       Retrieved Passages
                |
                v
      Ollama + Qwen 2.5 3B
                |
                v
       Context-Grounded Answer
                |
                v
          Streamlit UI
```

## Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Web Interface | Streamlit |
| PDF Processing | PyMuPDF |
| Embedding Model | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector Search | FAISS |
| Keyword Retrieval | BM25 |
| Rank Fusion | Reciprocal Rank Fusion (RRF) |
| Language Model | `qwen2.5:3b` |
| Local Inference | Ollama |
| Numerical Computing | NumPy |
| Testing | `unittest` |
| Version Control | Git and GitHub |

## How It Works

1. **Document Ingestion:** Extract text from uploaded PDF documents and preserve source filenames and page numbers.
2. **Text Chunking:** Split extracted text into overlapping chunks to improve retrieval granularity.
3. **Embedding Generation:** Convert document chunks into normalized vector representations using Sentence Transformers.
4. **Index Construction:** Store embeddings in a FAISS index and create a BM25 keyword-search index.
5. **Hybrid Retrieval:** Retrieve candidate passages using semantic similarity and keyword relevance.
6. **Rank Fusion:** Combine the two rankings using Reciprocal Rank Fusion.
7. **Answer Generation:** Provide the retrieved passages and user question to the Qwen model through Ollama.
8. **Source Inspection:** Display the generated answer and supporting passages with source filenames and page numbers.

## Project Structure

```text
Advanced-RAG-System/
|
|-- app.py
|-- ingestion.py
|-- retrieval.py
|-- generation.py
|-- evaluation.py
|-- requirements.txt
|-- README.md
|-- .gitignore
|
|-- data/
|
|-- tests/
    |-- test_ingestion.py
    |-- test_retrieval.py
    |-- test_evaluation.py
```

| File | Description |
|---|---|
| `app.py` | Streamlit application and user interface |
| `ingestion.py` | PDF text extraction and chunk creation |
| `retrieval.py` | FAISS indexing, BM25 search, and hybrid retrieval |
| `generation.py` | Context-grounded answer generation using Ollama |
| `evaluation.py` | Information retrieval evaluation metrics |
| `requirements.txt` | Python dependency list |
| `tests/` | Automated tests for project components |

## Installation and Setup

### Prerequisites

- Python 3.11
- Git
- Ollama
- The `qwen2.5:3b` model

### 1. Clone the Repository

```powershell
git clone https://github.com/ahsanhabibabir04/Advanced-RAG-System.git
cd Advanced-RAG-System
```

### 2. Create a Virtual Environment

```powershell
python -m venv .venv
```

Activate the environment in Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install Dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Install and Prepare Ollama

Install Ollama from the official website:

https://ollama.com/

Download the language model:

```powershell
ollama pull qwen2.5:3b
```

Ensure that Ollama is running and the model is available before generating answers.

### 5. Run the Application

From the project directory, execute:

```powershell
streamlit run app.py
```

Open the local URL displayed in the terminal, typically:

http://localhost:8501

## Usage

1. Upload one or more text-based PDF documents.
2. Select **Process PDFs** to extract document text and build the hybrid retrieval index.
3. Enter a question about the uploaded documents.
4. Select **Get Answer** to retrieve relevant passages and generate a response.
5. Inspect the retrieved source passages and their associated page numbers.

**Note:** Scanned PDFs without an extractable text layer are not supported by the current implementation.

## Testing

Run the complete automated test suite:

```powershell
python -m unittest discover -s tests -v
```

The tests cover:

- Text tokenization and overlapping chunk generation
- Input validation and error handling
- PDF text extraction and source metadata
- Empty retrieval results and invalid queries
- Reciprocal Rank, Precision@K, Recall@K, and MRR calculations

**Latest local test result:** 19 tests passed.

## Retrieval Evaluation

The project implements the following information retrieval metrics:

| Metric | Description |
|---|---|
| Precision@K | Proportion of relevant results among the top K retrieved passages, using K as the denominator |
| Recall@K | Proportion of known relevant passages retrieved in the top K |
| Reciprocal Rank (RR) | Reciprocal of the rank of the first relevant result |
| Mean Reciprocal Rank (MRR) | Average reciprocal rank across multiple queries |

These metrics are implemented in `evaluation.py` and covered by automated tests. Meaningful benchmark results require a labeled evaluation dataset containing queries and their relevant document chunks.

**Evaluation status:** Metric implementations and unit tests are complete. A labeled retrieval benchmark and measured evaluation results have not yet been established.

## Current Limitations

- Answer generation requires a locally available Ollama runtime and the configured Qwen model.
- Scanned PDFs requiring optical character recognition are not supported.
- Retrieval evaluation requires labeled relevance judgments for meaningful benchmark measurements.
- Answer quality depends on document extraction, retrieval relevance, and the language model's output.
- The application is a portfolio project and is not yet a production-ready document intelligence system.

## Future Improvements

- Add OCR support for scanned documents.
- Persist document embeddings and retrieval indexes between sessions.
- Make chunk size, overlap, and retrieval weights configurable.
- Integrate a dedicated passage reranking model.
- Create a labeled benchmark dataset and report measured retrieval metrics.
- Improve error handling and document management.
- Deploy the application using an appropriate hosted LLM inference service.

## Learning Outcomes

This project demonstrates practical experience with:

- Retrieval-Augmented Generation architecture
- Semantic and lexical information retrieval
- Vector indexing and similarity search
- Hybrid retrieval and rank fusion
- PDF processing and source attribution
- Local LLM integration
- Information retrieval evaluation metrics
- Modular Python development and automated testing
- Building interactive AI applications with Streamlit

## Author

**M.A. Ahsan Habib Abir**

GitHub: [ahsanhabibabir04](https://github.com/ahsanhabibabir04)

Repository: [Advanced-RAG-System](https://github.com/ahsanhabibabir04/Advanced-RAG-System)

## Disclaimer

This project is developed for educational and portfolio purposes. Generated responses may be incomplete or inaccurate and should be verified against the original source documents. The application has not been validated for production or safety-critical use.

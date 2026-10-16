# PDF RAG Pipeline

### Document Question Answering with Semantic Search and Source Citations

A Python command-line application that answers questions using the contents of local PDF documents. It combines Retrieval-Augmented Generation (RAG), local sentence embeddings, a persistent Chroma vector database, and a Groq-hosted language model.

## Overview

Instead of sending an entire PDF to the LLM, the pipeline extracts text, divides it into overlapping chunks, and retrieves relevant passages for each question. The LLM receives those excerpts and is instructed to produce a grounded answer with filename and page references, or acknowledge missing information.

## Features

- **Multiple-PDF ingestion:** Recursively discover PDFs inside `data/pdfs/`.
- **Page-aware extraction:** Preserve source filenames and page metadata using PyMuPDF.
- **Tokenizer-aware chunking:** Split text into overlapping chunks targeting 200 tokens with 40-token overlap.
- **Local embeddings:** Encode passages and questions with `sentence-transformers/all-MiniLM-L6-v2`.
- **Persistent vector storage:** Store text, metadata, and normalized embeddings in ChromaDB.
- **Semantic retrieval:** Rank passages using cosine similarity, with configurable result count and minimum score.
- **Source-referenced answers:** Generate answers through Groq and display filenames, page numbers, and retrieval scores.
- **Interactive CLI:** Ask multiple independent questions in one session.
- **Repeatable indexing:** Use stable document IDs derived from relative PDF paths and replace a document's chunks when reindexing.
- **Error handling:** Report indexing errors, missing API configuration, and questions exceeding the embedding model's input limit.

## Tech Stack

| Technology | Role |
| --- | --- |
| Python | Command-line orchestration and processing |
| PyMuPDF | PDF text extraction |
| LangChain Text Splitters | Tokenizer-aware text splitting |
| Sentence Transformers | Local embedding generation |
| ChromaDB | Persistent vector database and similarity search |
| LangChain Groq | LLM integration |
| Groq-hosted `openai/gpt-oss-20b` | Default answer-generation model |
| python-dotenv | Environment configuration |

## Pipeline Workflow

### Indexing

1. Discover PDF files in `data/pdfs/`, including subdirectories.
2. Derive a document ID from each PDF's relative path.
3. Extract readable page text and preserve page indices.
4. Split text into overlapping chunks and check their token lengths.
5. Generate normalized sentence embeddings.
6. Store chunks, embeddings, and source metadata in a cosine Chroma collection.

### Question Answering

1. Accept a question from the terminal.
2. Embed the question using the same embedding model.
3. Search chunks belonging to PDFs successfully indexed during the current run.
4. Filter matches using the minimum cosine similarity score.
5. Build a numbered context within a 12,000-character budget.
6. Ask the LLM to answer from the retrieved excerpts.
7. Print the answer and source references with human-readable page numbers.

## Project Files

| Path | Purpose |
| --- | --- |
| `rag_pipeline.py` | PDF discovery, document IDs, indexing orchestration, question loop, and output |
| `rag_service.py` | Model loading, chunking, embedding, Chroma indexing, and retrieval |
| `llm_service.py` | LLM configuration, context construction, and answer generation |
| `requirements.txt` | Python dependencies |
| `.env` | Local Groq credentials and model configuration |
| `.gitignore` | Exclude credentials, environment files, and generated data |
| `data/pdfs/` | Input PDFs |
| `data/chroma/` | Generated persistent vector database |

## Getting Started

### Prerequisites

- A Python environment compatible with the dependencies below.
- A Groq API key with access to the configured model.
- Internet access for the first embedding-model download and LLM requests.
- Text-based PDFs containing the information you want to query.

The following commands use **Windows Command Prompt**. Start inside the project folder.

### 1. Create and activate a virtual environment

```bat
py -m venv .venv
.venv\Scripts\activate
```

### 2. Install dependencies

Use the following contents for `requirements.txt`:

```text
pymupdf
langchain-text-splitters
sentence-transformers
chromadb
langchain-groq
python-dotenv
```

Install them:

```bat
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 3. Configure the LLM

Create `.env` beside the Python modules:

```env
GROQ_API_KEY=your_actual_groq_api_key
GROQ_MODEL=openai/gpt-oss-20b
```

Obtain your key from [Groq Console](https://console.groq.com/keys). Keep it private and restart the pipeline after changing configuration. Model access and provider limits depend on your Groq account.

### 4. Add PDF documents

```bat
mkdir data\pdfs
```

Copy PDFs into that directory. Subdirectories are supported. Prefer explanatory notes or reference material: a syllabus checklist may mention a topic without containing an answer about it.

### 5. Check syntax and run

```bat
python -m py_compile rag_pipeline.py rag_service.py llm_service.py
python rag_pipeline.py
```

The first run may download the embedding model. The application indexes PDFs before displaying the question prompt.

### 6. Ask questions

Example input, when your PDFs cover the topic:

```text
Question: How are containers different from virtual machines?
```

Answers appear as Markdown text in the terminal, followed by retrieved source filenames, page numbers, and similarity scores. Their wording depends on the actual document contents.

Enter `exit` or `quit` to finish. You can also press `Ctrl+C`.

## Configuration

| Setting | Location | Default | Meaning |
| --- | --- | --- | --- |
| `GROQ_API_KEY` | `.env` | Required for answers | Groq API credential |
| `GROQ_MODEL` | `.env` | `openai/gpt-oss-20b` | Answer-generation model |
| `TOP_K` | `rag_pipeline.py` | `5` | Maximum retrieved passages |
| `MIN_SCORE` | `rag_pipeline.py` | `0.2` | Minimum cosine similarity |
| `MODEL_NAME` | `rag_service.py` | `sentence-transformers/all-MiniLM-L6-v2` | Local embedding model |
| `COLLECTION_NAME` | `rag_service.py` | `study_documents_minilm_v1` | Chroma collection inherited from the original project |
| Chunk size / overlap | `rag_service.py` | `200` / `40` tokens | Text splitting settings |
| `MAX_CONTEXT_CHARS` | `llm_service.py` | `12000` | Character budget for numbered excerpts |

The default threshold is a starting point, not a calibrated confidence score. Evaluate it with relevant and unrelated questions. If changing the embedding model or chunking configuration, update the collection metadata/profile and use a new collection name before reindexing.

## Indexing Behavior

- PDFs are reindexed at every startup; persistent storage does not currently skip unchanged documents.
- Successfully indexed documents are the only ones searched during that run.
- Removed PDFs can leave old chunks in Chroma, but those chunks are excluded because their document IDs are not in the current search list.
- Stable IDs prevent duplicate chunks for repeated indexing of the same relative file path.
- Reindexing replaces the old document index; a failed rebuild may require another run before the document is searchable.

## Limitations

- No OCR: image-only PDFs produce no readable text, and image-only pages in mixed PDFs are skipped.
- No conversation memory: every question is processed independently.
- No web search or general-knowledge fallback: answers are instructed to rely on retrieved PDF excerpts.
- LLM-generated citations are not automatically verified for factual support.
- Cosine similarity measures retrieval similarity, not answer correctness.
- Context is bounded by characters rather than exact LLM tokens.
- Tables, complex layouts, and equations may not extract cleanly.
- Questions exceeding the embedding model's token limit are rejected.
- No measured accuracy, latency, or scalability benchmark is claimed.

## Data and Credentials

PDFs and embeddings are stored locally. For answer generation, the question and selected PDF excerpts are sent to Groq. Only process documents you are permitted to send to that provider.

Recommended `.gitignore`:

```gitignore
.env
.venv/
__pycache__/
*.pyc
data/
```

Keep `.env` and generated databases out of Git. A committed `.env.example` may contain placeholder values, never a real API key.

## Troubleshooting

| Problem | What to check |
| --- | --- |
| `ModuleNotFoundError` | Activate `.venv` and install `requirements.txt` with that environment's Python. |
| No PDFs found | Place PDFs in `data/pdfs/`, relative to `rag_pipeline.py`. |
| No readable text | Use a text-based PDF or apply OCR externally. |
| No relevant passages | Check the PDF actually contains the answer and inspect retrieval settings. |
| Missing API key | Set `GROQ_API_KEY` in `.env` beside `llm_service.py`. |
| LLM request failure | Check network access, API credentials, model availability, and provider quota. |
| Collection mismatch | Use a fresh collection with matching embedding and chunking settings. |

## Future Improvements

- Skip unchanged PDFs using content hashes.
- Remove stale chunks when PDFs are deleted.
- Add hybrid keyword/vector search and reranking.
- Support OCR for scanned documents.
- Evaluate retrieval and citation quality with a reference question set.
- Add command-line options for indexing, searching, and configuration.

## Author

**Mukul Dev Singh** — [GitHub](https://github.com/mukul190303)

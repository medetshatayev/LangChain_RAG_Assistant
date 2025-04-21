# Quest Analytics RAG Assistant

An AI-powered Retrieval-Augmented Generation (RAG) assistant that helps researchers analyze scientific papers efficiently. This tool can read, understand, and summarize vast amounts of data in real-time.

## Features

- Document loading from various sources (PDF, text files)
- Text splitting techniques for different document formats (including LaTeX)
- Document embedding using sentence transformers
- Vector database storage with ChromaDB
- Retrieval system to fetch relevant document segments
- QA Bot with Gradio interface for easy interaction

## Project Structure

The project is organized into separate modules for each task:

1. `document_loading.py` - Loading documents using LangChain
2. `text_splitting.py` - Splitting text (especially LaTeX) into chunks
3. `embeddings.py` - Creating vector embeddings for text
4. `vector_database.py` - Creating and searching ChromaDB vector databases
5. `retriever.py` - Developing a retriever for fetching document segments
6. `qa_bot.py` - QA Bot with Gradio interface for document interaction
7. `main.py` - Central script that runs all tasks sequentially

## Installation

1. Clone this repository
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

3. Set up environment variables:
   - `HF_API_TOKEN`: Your Hugging Face API token

## Usage

### Running All Tasks

To run all tasks sequentially:

```bash
python main.py
```

### Running Individual Tasks

You can also run each task separately:

```bash
# Task 1: Document Loading
python document_loading.py

# Task 2: Text Splitting
python text_splitting.py

# Task 3: Document Embedding
python embeddings.py

# Task 4: Vector Database
python vector_database.py

# Task 5: Retriever
python retriever.py

# Task 6: QA Bot
python qa_bot.py
```

## Note

The PDF and text files should be placed in the "/Users/{user}/Downloads/" directory or update the paths in the code:

## License

This project is licensed under the MIT License. 
import os
import sys

# Import all task modules
try:
    from document_loading import load_pdf_document
    from text_splitting import split_latex_text
    from embeddings import embed_text
    from vector_database import create_vectordb
    from retriever import create_retriever
    from qa_bot import create_qa_interface

except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)

def main():
    print("Quest Analytics RAG Assistant")
    print("-" * 30)
    
    # Task 1: Load PDF document
    pdf_path = "/Users/mshatayev/Downloads/A_Comprehensive_Review_of_Low_Rank_Adaptation.pdf"
    print("\nTask 1: Document Loading")
    if os.path.exists(pdf_path):
        documents = load_pdf_document(pdf_path)
        print(f"Document loaded successfully with {len(documents)} pages/chunks")
    else:
        print(f"PDF file not found at path: {pdf_path}")
    
    # Task 2: Split LaTeX text
    print("\nTask 2: Text Splitting")
    latex_text = r"""
    \documentclass{article}

    \begin{document}

    \maketitle

    \section{Introduction}

    Large language models (LLMs) are a type of machine learning model that can be trained on vast amounts of text data to generate human-like language. In recent years, LLMs have made significant advances in various natural language processing tasks, including language translation, text generation, and sentiment analysis.

    \subsection{History of LLMs}

    The earliest LLMs were developed in the 1980s and 1990s, but they were limited by the amount of data that could be processed and the computational power available at the time. In the past decade, however, advances in hardware and software have made it possible to train LLMs on massive datasets, leading to significant improvements in performance.

    \subsection{Applications of LLMs}

    LLMs have many applications in the industry, including chatbots, content creation, and virtual assistants. They can also be used in academia for research in linguistics, psychology, and computational linguistics.

    \end{document}
    """
    splits = split_latex_text(latex_text)
    print(f"LaTeX text split into {len(splits)} chunks")
    
    # Task 3: Embed text
    print("\nTask 3: Document Embedding")
    query = "How are you?"
    embeddings = embed_text(query)
    if embeddings:
        print(f"Embedding successful for query: '{query}'")
        print(f"Total embedding dimensions: {len(embeddings)}")
        print(f"First 5 embedding values: {embeddings[:5]}")
    else:
        print("Embedding skipped or failed (check credentials in .env and task3_embeddings.py output).")
    
    # Task 4: Create vector database
    print("\nTask 4: Vector Database Creation")
    policies_path = "/Users/mshatayev/Downloads/new-Policies.txt"
    if os.path.exists(policies_path):
        query = "Smoking policy"
        vectordb = create_vectordb(policies_path, query)
        if hasattr(vectordb, '_collection'):
            print(f"Vector database created successfully with {vectordb._collection.count()} documents")
        else:
            print("Vector database created successfully")
    else:
        print(f"Policy file not found at path: {policies_path}")
    
    # Task 5: Develop a retriever
    print("\nTask 5: Retriever Development")
    if os.path.exists(policies_path):
        query = "Email policy"
        retriever = create_retriever(policies_path, query)
        print("Retriever created successfully")
    
    # Task 6: Launch QA Bot
    print("\nTask 6: QA Bot")
    print("Launching QA Bot interface...")
    demo = create_qa_interface()
    demo.launch()

if __name__ == "__main__":
    main()
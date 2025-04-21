import os
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

def create_vectordb(file_path, query):
    # Load the text document
    loader = TextLoader(file_path)
    documents = loader.load()

    # Split the document
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    splits = text_splitter.split_documents(documents)
    
    # Create embedding model
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Create ChromaDB and add documents
    vectordb = Chroma.from_documents(documents=splits, embedding=embedding_model)
    
    # Perform similarity search
    results = vectordb.similarity_search(query, k=5)
    
    print(f"Top 5 results for query: '{query}'")
    for i, doc in enumerate(results):
        print(f"Result {i+1}: {doc.page_content[:100]}...\n")
    
    return vectordb

if __name__ == "__main__":
    # Example usage
    policies_path = "/Users/mshatayev/Downloads/new-Policies.txt"
    
    if os.path.exists(policies_path):
        query = "Smoking policy"
        vectordb = create_vectordb(policies_path, query)
        print(f"Vector database created successfully with {vectordb._collection.count()} documents")
    else:
        print(f"Policy file not found at path: {policies_path}") 
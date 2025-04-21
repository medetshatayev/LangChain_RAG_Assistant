import os
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

def create_retriever(file_path, query):
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
    
    # Create retriever
    retriever = vectordb.as_retriever(search_kwargs={"k": 2})
    
    # Get results from retriever
    results = retriever.get_relevant_documents(query)
    
    print(f"Top 2 results for query: '{query}'")
    for i, doc in enumerate(results):
        print(f"Result {i+1}: {doc.page_content[:100]}...\n")
    
    return retriever

if __name__ == "__main__":
    policies_path = "/Users/mshatayev/Downloads/new-Policies.txt"
    
    if os.path.exists(policies_path):
        query = "Email policy"
        retriever = create_retriever(policies_path, query)
        print("Retriever created successfully")
    else:
        print(f"Policy file not found at path: {policies_path}") 
import os
from langchain_community.document_loaders import PyPDFLoader

def load_pdf_document(pdf_path):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    content = ""
    for doc in documents:
        content += doc.page_content
    
    print("First 1000 characters of the document:")
    print(content[:1000])
    
    return documents

if __name__ == "__main__":
    pdf_path = "/Users/mshatayev/Downloads/A_Comprehensive_Review_of_Low_Rank_Adaptation.pdf"
    
    if os.path.exists(pdf_path):
        documents = load_pdf_document(pdf_path)
        print(f"Document loaded successfully with {len(documents)} pages/chunks")
    else:
        print(f"PDF file not found at path: {pdf_path}") 
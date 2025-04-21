import os
import gradio as gr
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def process_pdf(pdf_path):
    # Load PDF document
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    
    # Get first 1000 characters for display
    content = ""
    for doc in documents:
        content += doc.page_content
    first_1000 = content[:1000]
    
    # Split the document
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    splits = text_splitter.split_documents(documents)
    
    # Create embedding model
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Create vectorstore
    vectordb = Chroma.from_documents(documents=splits, embedding=embedding_model)
    
    # Initialize retriever
    retriever = vectordb.as_retriever()
    
    # Get Hugging Face API token from environment variable
    hf_token = os.environ.get("HF_API_TOKEN")
    
    # Initialize LLM
    if not hf_token:
        raise ValueError("HF_API_TOKEN not found in environment variables. Please set it.")

    try:
        print("Using HuggingFace Endpoint wrapped by ChatHuggingFace")
        # 1. Create the endpoint connection
        endpoint_llm = HuggingFaceEndpoint(
            repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
            huggingfacehub_api_token=hf_token,
            task="text-generation",
            temperature=0.1,
            max_new_tokens=512,
            top_p=0.95
        )
        # 2. Wrap it with ChatHuggingFace
        llm = ChatHuggingFace(llm=endpoint_llm)
    except Exception as e:
        raise RuntimeError(f"Error initializing HuggingFace LLM: {str(e)}") from e
    
    # Create QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )
    
    return qa_chain, first_1000

def answer_query(pdf_file, query):
    if pdf_file is None:
        return "Please upload a PDF file first."
    
    try:
        qa_chain, first_1000 = process_pdf(pdf_file)
        # Use the newer invoke method instead of __call__
        result = qa_chain.invoke({"query": query})
        
        answer = result["result"]
        sources = [doc.page_content[:200] + "..." for doc in result["source_documents"]]
        
        response = f"Document Preview (first 1000 chars):\n{first_1000}\n\n"
        response += f"Answer: {answer}\n\nSources:\n"
        for i, source in enumerate(sources[:2]):  # Only show first 2 sources for brevity
            response += f"Source {i+1}: {source}\n\n"
        
        return response
    except Exception as e:
        # Return the actual error message instead of a generic one
        return f"Error processing your query: {str(e)}"

def create_qa_interface():
    with gr.Blocks(title="Quest Analytics QA Bot") as demo:
        gr.Markdown("# Quest Analytics QA Bot")
        gr.Markdown("Upload a PDF and ask questions about it.")
        
        # Add a note about model access requirements
        gr.Markdown("""
        > **Note**: This application uses the Mixtral-8x7B-Instruct-v0.1 model, which requires users 
        > to agree to share contact information. Make sure your Hugging Face account 
        > (associated with the API token) has accepted the model terms.
        """)
        
        with gr.Row():
            with gr.Column():
                pdf_input = gr.File(label="Upload PDF", file_types=[".pdf"])
                query_input = gr.Textbox(label="Your Question", value="What this paper is talking about?")
                submit_btn = gr.Button("Ask Question")
            
            with gr.Column():
                output = gr.Textbox(label="Response", lines=20)
        
        submit_btn.click(
            fn=answer_query,
            inputs=[pdf_input, query_input],
            outputs=output
        )
    
    return demo

if __name__ == "__main__":
    # Check if the token is available
    hf_token = os.environ.get("HF_API_TOKEN")
    if hf_token:
        print("Hugging Face API token found in environment variables.")
        print("IMPORTANT: Make sure your HF account has accepted the terms for Mixtral-8x7B-Instruct-v0.1")
        print("Visit https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1 and accept the model terms if you haven't already.")
    else:
        # Instead of just printing a warning, raise an error if the token is missing at startup.
        # Alternatively, we could let process_pdf handle it, but failing early might be better.
        raise ValueError("HF_API_TOKEN not found in environment variables. The application cannot start without it.")
    
    print("Starting QA Bot...")
    
    # Create and launch the QA bot interface
    demo = create_qa_interface()
    demo.launch() 
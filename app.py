import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import GoogleGenerativeAIEmbeddings
import streamlit as st
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    st.error("Google API Key not found. Please set it in the .env file.")
genai.configure(api_key=api_key)


# Read all PDFs and return extracted text
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text() or ""  # Handle cases where text might be None
    return text


# Split text into manageable chunks
def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=500)
    chunks = splitter.split_text(text)
    return chunks  # list of strings


# Get embeddings for each chunk and save as FAISS index
def get_vector_store(chunks):
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vector_store = FAISS.from_texts(chunks, embedding=embeddings)
        vector_store.save_local("faiss_index")
        st.success("Vector store created successfully!")
    except Exception as e:
        st.error(f"Error creating vector store: {e}")
        raise


# Create a conversational chain
def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context. If the answer is not in
    the provided context, just say, "Answer is not available in the context." Don't provide a wrong answer.
    
    Context:\n{context}?\n
    Question:\n{question}\n
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-1.5-pro", client=genai, temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(llm=model, chain_type="stuff", prompt=prompt)
    return chain


# Process user input and query FAISS index
def user_input(user_question):
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        docs = vector_store.similarity_search(user_question)
        chain = get_conversational_chain()
        response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
        return response.get("output_text", "No response generated.")
    except Exception as e:
        st.error(f"Error processing question: {e}")
        return "Error occurred while processing your question."


# Clear chat history
def clear_chat_history():
    st.session_state.messages = [
        {"role": "assistant", "content": "Upload some PDFs and ask me a question."}
    ]


# Main app
def main():
    st.set_page_config(page_title="Gemini PDF Chatbot", page_icon="🤖")

    # Sidebar for uploading PDF files
    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader(
            "Upload your PDF Files and Click on Submit & Process", accept_multiple_files=True
        )
        if pdf_docs:
            total_size = sum(pdf.size for pdf in pdf_docs)
            if total_size > 200 * 1024 * 1024:  # 200MB limit
                st.error("Total file size exceeds 200MB. Please upload smaller files.")
            elif st.button("Submit & Process"):
                with st.spinner("Processing..."):
                    raw_text = get_pdf_text(pdf_docs)
                    if raw_text.strip():
                        text_chunks = get_text_chunks(raw_text)
                        get_vector_store(text_chunks)
                    else:
                        st.error("No readable text found in the uploaded PDFs.")

    # Main content area for chat
    st.title("Chat with PDF files using Gemini 🤖")
    st.sidebar.button("Clear Chat History", on_click=clear_chat_history)

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Upload some PDFs and ask me a question."}
        ]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    if prompt := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = user_input(prompt)
                st.write(response)

        if response:
            st.session_state.messages.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    main()

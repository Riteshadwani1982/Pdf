import os
import streamlit as st
import google.generativeai as genai

from dotenv import load_dotenv
from PyPDF2 import PdfReader

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate


# Load API Key
load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    st.error("Google API key not found")

genai.configure(api_key=api_key)


# Extract text from PDFs
def get_pdf_text(pdf_docs):

    text = ""

    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)

        for page in pdf_reader.pages:
            page_text = page.extract_text()

            if page_text:
                text += page_text

    return text


# Split text
def get_text_chunks(text):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=200
    )

    return splitter.split_text(text)


# Create Vector Store
def get_vector_store(text_chunks):

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)

    vector_store.save_local("faiss_index")

    st.success("PDF processed successfully")


# QA Chain
def get_conversational_chain():

    prompt_template = """
    Answer the question using the provided context.

    If the answer is not present in the context say:
    "Answer is not available in the context."

    Context:
    {context}

    Question:
    {question}

    Answer:
    """

    model = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.3
    )

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    chain = load_qa_chain(
        llm=model,
        chain_type="stuff",
        prompt=prompt
    )

    return chain


# User query
def user_input(question):

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_store = FAISS.load_local(
        "faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )

    docs = vector_store.similarity_search(question)

    chain = get_conversational_chain()

    response = chain(
        {"input_documents": docs, "question": question},
        return_only_outputs=True
    )

    return response["output_text"]


# Clear chat
def clear_chat():

    st.session_state.messages = [
        {"role": "assistant", "content": "Upload PDFs and ask questions."}
    ]


# Streamlit app
def main():

    st.set_page_config(page_title="Gemini PDF Chatbot", page_icon="🤖")

    st.title("Chat with PDFs using Gemini 🤖")

    with st.sidebar:

        st.header("Upload PDFs")

        pdf_docs = st.file_uploader(
            "Upload PDF files",
            accept_multiple_files=True
        )

        if pdf_docs and st.button("Submit & Process"):

            with st.spinner("Processing..."):

                raw_text = get_pdf_text(pdf_docs)

                if raw_text.strip():

                    text_chunks = get_text_chunks(raw_text)

                    get_vector_store(text_chunks)

                else:
                    st.error("No readable text found in PDFs.")

        st.button("Clear Chat History", on_click=clear_chat)

    if "messages" not in st.session_state:

        st.session_state.messages = [
            {"role": "assistant", "content": "Upload PDFs and ask questions."}
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

        st.session_state.messages.append(
            {"role": "assistant", "content": response}
        )


if __name__ == "__main__":
    main()
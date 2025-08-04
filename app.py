import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain_text_splitters.character import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import HumanMessage, SystemMessage
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_core.documents import Document

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key)

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Function to extract text along with page number references
def get_text_from_pdf(uploaded_pdf):
    docs = []
    for pdf in uploaded_pdf:
        reader = PdfReader(pdf)
        number_of_pages = len(reader.pages)
        for i in range(number_of_pages):
            page = reader.pages[i]
            text = page.extract_text()
            if text:
                docs.append(Document(page_content=text, metadata={"page_number": i+1}))
    return docs

# Split text into chunks
def get_text_chunks(documents):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_documents(documents)
    return chunks

# Vector Store using FAISS
def get_vector_store(text_chunks):
    embedding = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    vector_store = FAISS.from_documents(documents=text_chunks, embedding=embedding)
    return vector_store

# RAG Chain Setup
def get_conversational_rag_chain(vector_store):
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=False,  
        output_key="answer"
    )
    return qa_chain

def main():
    st.set_page_config(page_title="Chat with PDF", page_icon=":books:")
    st.header("Chat with PDF :books:")

    if "qa_chain" not in st.session_state:
        st.session_state.qa_chain = None

    with st.sidebar:
        st.subheader("Your Documents")
        uploaded_pdf = st.file_uploader("Upload your PDFs here...", accept_multiple_files=True)

        if st.button("Process PDFs"):
            if uploaded_pdf:
                with st.spinner("Processing..."):
                    documents = get_text_from_pdf(uploaded_pdf)
                    text_chunks = get_text_chunks(documents)
                    vector_store = get_vector_store(text_chunks)
                    st.session_state.qa_chain = get_conversational_rag_chain(vector_store)
                    st.success("PDFs processed successfully! You can now ask questions.")
            else:
                st.warning("Please upload at least one PDF.")

    if st.session_state.qa_chain:
        user_query = st.text_input("Ask a question about your document:")

        if user_query:
            with st.spinner("Fetching answer..."):
                result = st.session_state.qa_chain({"question": user_query})
                answer = result["answer"]

                st.markdown(f"### Answer:\n{answer}")

        if memory.chat_memory.messages:
            st.markdown("### Conversation History")
            for message in memory.chat_memory.messages:
                if isinstance(message, HumanMessage):
                    st.markdown(f"**You:** {message.content}")
                else:
                    st.markdown(f"**Bot:** {message.content}")

if __name__ == '__main__':
    main()

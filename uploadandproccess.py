import streamlit as st
import os
import tempfile
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, CSVLoader, WebBaseLoader
from functions import reset_files
def upload_and_process_files(file_type):
    temp_filepath = None
    
    if file_type == "URL":
        URL = st.text_input("Enter website URL ") 
        if URL:
            st.session_state.URL = True
    else:
        uploaded_files = st.file_uploader("Upload File", accept_multiple_files=True, on_change=reset_files, type=["pdf", "csv"])
        if uploaded_files:
            st.session_state.uploaded_file = True
        for uploaded_file in uploaded_files:
            if uploaded_file is not None:
                # Write to temporary file
                temp_dir = tempfile.TemporaryDirectory()
                file = uploaded_file
                temp_filepath = os.path.join(temp_dir.name, file.name)
                with open(temp_filepath, 'wb') as f:
                    f.write(file.getvalue())
    
    if (st.session_state.uploaded_file or st.session_state.URL) and (st.session_state.processed_files == False):
        print("processing new files")
        with st.spinner("processing document..."):
            # Extract embeddings
            embeddings = OpenAIEmbeddings()
        
            # Choosing loader
            if file_type == "PDF":
                loader = PyPDFLoader(file_path=temp_filepath)
            elif file_type == "CSV":
                loader = CSVLoader(file_path=temp_filepath)
            else:
                loader = WebBaseLoader(URL)
            
            documents = loader.load_and_split()
            
            # Vector store
            print("creating new vector store")
            st.session_state.vectorstore = Chroma.from_documents(documents, embeddings, collection_name="abc")
            # Clear documents
            documents = None
            st.session_state.processed_files = True
            st.write(" Document loaded")
            
    elif (not st.session_state.uploaded_file) and (not st.session_state.URL):
        st.warning("Please upload at least one document before submitting a question.")
        st.stop()
    
    return temp_filepath
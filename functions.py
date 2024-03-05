
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader ,CSVLoader,WebBaseLoader
def reset_vector():
        print("clearing vector store")
        if st.session_state.vectorstore:
            collection = st.session_state.vectorstore
            collection.delete_collection()
        st.session_state.processed_files = False
        st.session_state.uploaded_file = False
        st.session_state.URL = False
    
def reset_files():
    print("detected file change")
    st.session_state.processed_files = False
    st.session_state.uploaded_file = False
    st.session_state.URL = False


                                
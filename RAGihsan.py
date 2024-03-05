import os
import warnings
from dotenv import load_dotenv

import streamlit as st

from functions import reset_files, reset_vector
from answerresponse import perform_question_answering
from uploadandproccess import upload_and_process_files


warnings.filterwarnings("ignore",category=UserWarning)
warnings.filterwarnings("ignore",category=DeprecationWarning)

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.environ.get('api_key')


# Initialize session_state
if 'processed_files' not in st.session_state:
    st.session_state.processed_files = False
# Initialize vectorstore in session_state
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
if 'loadedfiles' not in st.session_state:
    st.session_state.loadedfiles = []
if 'uploaded_file' not in st.session_state:
    st.session_state.uploaded_file = False
if 'URL' not in st.session_state:    
    st.session_state.URL = False           

URL =None
uploaded_files =None

#streamlit titles
st.set_page_config(page_title="RAG use case")
st.title("RAG CHAIN")
st.header("OpenAI LLM Chatbot")

with st.sidebar:
    st.header("Choose file Type")
    file_type = st.radio("Choose file Type ", ["PDF","CSV","URL"], index=0,key="filetype",on_change=reset_files,
                         horizontal=True,label_visibility="collapsed")
    st.caption("Reset the vector store")
    clear_button= st.button("Clear DB")


if clear_button:
    reset_vector()
       
#file uploader    
st.markdown("Upload files and get answers based on the content ")  
st.header(f"Upload {file_type} ")


#process file
upload_and_process_files(file_type)

#question

st.header("Ask a question")
question = st.text_area("Enter your questions here")  
q_submit = st.button("submit")

with st.sidebar:
    score = st.slider('Adjust required relevancy score threshold from response ', 0.0, 1.0, 0.6)
    model_name = st.radio("Choose GPT model ", ["gpt-4","gpt-3.5-turbo-0125"], index=0,key="modelname")
    if st.session_state.processed_files == True:
        st.write("file processed")

#chain    
if q_submit: 
    with st.spinner("loading..."):
        # Use the existing vectorstore for retriever initialiclear_buttonation
        perform_question_answering(question, model_name, score)
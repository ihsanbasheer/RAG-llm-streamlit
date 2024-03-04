import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.chroma import Chroma
from langchain.llms.openai import OpenAIChat
from langchain_openai import ChatOpenAI
import warnings 
from langchain_community.document_loaders import PyPDFLoader ,CSVLoader,WebBaseLoader
import os
import tempfile 

warnings.filterwarnings("ignore",category=UserWarning)
warnings.filterwarnings("ignore",category=DeprecationWarning)

api_key = "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"
os.environ["OPENAI_API_KEY"] = "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"

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

if file_type == "URL":
    URL = st.text_input("Enter website URL ") 
    if URL:
        st.session_state.URL = True
else:
    uploaded_files = None
    uploaded_files = st.file_uploader("Upload File",accept_multiple_files= True ,on_change=reset_files ,type = ["pdf","csv"])
    if uploaded_files:
        st.session_state.uploaded_file =True
    for uploaded_file in uploaded_files:
        if uploaded_file is not None:
            # Write to temporary file
            temp_dir = tempfile.TemporaryDirectory()
            file = uploaded_file
            #print(f"""Converting: {file}""")
            temp_filepath = os.path.join(temp_dir.name, file.name)
            with open(temp_filepath, 'wb') as f:
                f.write(file.getvalue())
  
st.write("---")


# embeddings and loader

if (st.session_state.uploaded_file or st.session_state.URL) and (st.session_state.processed_files == False):
    print("processing new files")
    with st.spinner("processing document..."):
        
        # extract embeddings
        embeddings = OpenAIEmbeddings()
        
        #choosing loader
        if file_type == "PDF":
            loader = PyPDFLoader(file_path= temp_filepath)
            
        elif file_type == "CSV":
            loader = CSVLoader(file_path= temp_filepath)
            
        else:
            loader = WebBaseLoader(URL)
            
                    

    documents = loader.load_and_split()
    
    # vector store
    print("creating new vector store")
    st.session_state.vectorstore = Chroma.from_documents(documents, embeddings, collection_name ="abc" )
    # clear documents
    documents = None
    st.session_state.processed_files = True
    st.write(" Document loaded")
    

        
          
elif (not st.session_state.uploaded_file) and (not st.session_state.URL):
    st.warning("Please upload at least one document before submitting a question.")
    st.stop()


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
        # Search the DB.
        if not st.session_state.vectorstore:
            st.warning("Please upload document again.")
            st.session_state.processed_files = False
            
        else:
            print("using vectorstore")
            results = st.session_state.vectorstore.similarity_search_with_relevance_scores(question, k=3)  
            if len(results) == 0 or results[0][1] < score:
                print(f"Unable to find matching results.")
                st.write(f"Unable to find matching results, relevancy score = {results[0][1]}")
            else:
                PROMPT_TEMPLATE = """
                You are a bot that analyzes documents and answers question based on the content.
                If the information is not in documents you will answer I dont have the required information.
                
                {context}
                
                ---
                
                Answer the question based on the above context: {question}
                """

                context_text = "\n\n--\n\n".join([doc.page_content for doc, score in results])
                prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
                prompt = prompt_template.format(context=context_text, question=question)

                
                #prompting
                model = ChatOpenAI(model = model_name)
                response_text = model.invoke(prompt)
                
                #sources
                sources = [doc[0].metadata for doc in results]
                f_sources = [{'page': source.get('page', 0), 'source': os.path.basename(source['source'])} for source in sources]
                formatted_response = f"\n\n\nResponse: {response_text},\n\n\nRelevancy score = {results[0][1]}"
                
                #Respone
                
                st.subheader('Answer:')
                st.write(formatted_response)
                with st.sidebar:
                    st.write(f"Sources : {f_sources}")     
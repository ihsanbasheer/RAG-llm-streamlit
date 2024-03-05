import streamlit as st
import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

def perform_question_answering(question, model_name, score):
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
            model = ChatOpenAI(model=model_name)
            response_text = model.invoke(prompt)

            #sources
            sources = [doc[0].metadata for doc in results]
            f_sources = [{'page': source.get('page', 0), 'source': os.path.basename(source['source'])} for source in sources]
            formatted_response = f"\n\n\nResponse: {response_text},\n\n\nRelevancy score = {results[0][1]}"

            #Response
            st.subheader('Answer:')
            st.write(formatted_response)
            with st.sidebar:
                st.write(f"Sources : {f_sources}")
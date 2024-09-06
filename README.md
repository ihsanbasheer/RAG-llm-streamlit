# RAG Implementation

The Retrieval Augmented Generation (RAG) is a powerful tool for document retrieval, summarization, and interactive question-answering. This project utilizes LangChain, Streamlit, and Chroma to provide a seamless web application for users to perform these tasks. With RAG, you can easily upload multiple different type of documents/web URLS, generate vector embeddings for text within these documents, and perform conversational interactions with the documents. The chat history is also remembered for a more interactive experience.

# Features 

* Streamlit Web App: The project is built using Streamlit, providing an intuitive and interactive web interface for users.

* Document Uploader: Users can upload multiple files of types such as CSV ,TXT, PDF or even web URLS, which are then processed for further analysis.

* Document Splitting: The uploaded documents  are split into smaller text chunks, ensuring compatibility with models with token limits.

* Vector Embeddings: The text chunks are converted into vector embeddings, making it easier to perform retrieval and question-answering tasks.

* Interactive Conversations: Users can engage in interactive conversations with the documents, asking questions and receiving answers. 

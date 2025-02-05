import os

import streamlit as st
from langchain.chains import RetrievalQA
from langchain.chains import RetrievalQAWithSourcesChain
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.callbacks.base import BaseCallbackHandler
from langchain.vectorstores.neo4j_vector import Neo4jVector
from streamlit.logger import get_logger
from chains import (
    load_embedding_model,
    load_llm,
)

# load api key lib
from dotenv import load_dotenv

load_dotenv(".env")


url = os.getenv("NEO4J_URI")
username = os.getenv("NEO4J_USERNAME")
password = os.getenv("NEO4J_PASSWORD")
ollama_base_url = os.getenv("OLLAMA_BASE_URL")
embedding_model_name = os.getenv("EMBEDDING_MODEL")
llm_name = os.getenv("LLM")
# Remapping for Langchain Neo4j integration
os.environ["NEO4J_URL"] = url

logger = get_logger(__name__)


embeddings, dimension = load_embedding_model(
    embedding_model_name, config={"ollama_base_url": ollama_base_url}, logger=logger
)


class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)


llm = load_llm(llm_name, logger=logger, config={"ollama_base_url": ollama_base_url})


def main():
    st.header("📄Ask Magicbeans")
    #st.header("📄Chat with your pdf file")

    # upload a your pdf file
    pdf = st.file_uploader("Upload your PDFs", type="pdf", accept_multiple_files=True)

    
    if len(pdf) > 0 :
        for file in pdf:        
            pdf_reader = PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
            # langchain_textspliter
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, chunk_overlap=200, length_function=len
            )
            chunks = text_splitter.split_text(text=text)
            # Store the chunks part in db (vector)
            vectorstore = Neo4jVector.from_texts(
                chunks,
                url=url,
                username=username,
                password=password,
                embedding=embeddings,
                index_name="pdf_bot",
                node_label="PdfBotChunk",
                pre_delete_collection=False,  # Fucking KEEP the DATA!!!!!! 
                #pre_delete_collection=True,  # Delete existing PDF data
            )
            qa = RetrievalQA.from_chain_type(
                llm=llm, chain_type="stuff", retriever=vectorstore.as_retriever()
            )
         # Accept user questions/query
        query = st.text_input("Ask questions about your PDF files")
        if query:
            stream_handler = StreamHandler(st.empty())
            qa.run(query, callbacks=[stream_handler])

    if len(pdf) == 0 :
        st.subheader("No PDF file uploaded.Searching for answers...") 
        store = Neo4jVector.from_existing_index(
            index_name="pdf_bot",   
            url=url,    
            username=username,  
            password=password,  
            node_label="PdfBotChunk",   
            embedding=embeddings     
            #pre_delete_collection=False    # Keep the data
        )
        qa = RetrievalQA.from_chain_type(
            llm=llm, chain_type="stuff", retriever=store.as_retriever()
            )
        # Accept user questions/query
        query = st.text_input("Ask questions about your PDF files")

        if query:
            stream_handler = StreamHandler(st.empty())
            qa.run(query, callbacks=[stream_handler])

if __name__ == "__main__":
    main()

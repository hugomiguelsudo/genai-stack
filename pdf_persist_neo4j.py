import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.neo4j_vector import Neo4jVector
from dotenv import load_dotenv
from streamlit.logger import get_logger
from chains import (
    load_embedding_model,
    load_llm
)

class PDFtoNeo4j:
    def __init__(self, pdf_dir, neo4j_uri, neo4j_username, neo4j_password, embedding_model, logger):
        # Constructor for the class, initializes variables
        self.pdf_dir = pdf_dir
        self.neo4j_uri = neo4j_uri
        self.neo4j_username = neo4j_username
        self.neo4j_password = neo4j_password
        self.embedding_model = embedding_model
        self.logger = logger
        self._setup_environment()

    def _setup_environment(self):
        # Sets up the environment variable for Neo4J URL
        os.environ["NEO4J_URL"] = self.neo4j_uri

    def process_pdfs(self):
        # Processes each PDF file in the specified directory
        for filename in os.listdir(self.pdf_dir):
            if filename.endswith('.pdf'):
                with open(os.path.join(self.pdf_dir, filename), 'rb') as pdf_file:
                    self._process_pdf(pdf_file)

    def _process_pdf(self, pdf_file):
        # Reads and processes a single PDF file
        pdf_reader = PdfReader(pdf_file)

        # Extracts text from each page and concatenates it
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        # Splits the text into chunks using RecursiveCharacterTextSplitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200, length_function=len
        )
        chunks = text_splitter.split_text(text=text)

        # Stores the chunks in Neo4j database
        vectorstore = Neo4jVector.from_texts(
            chunks,
            url=self.neo4j_uri,
            username=self.neo4j_username,
            password=self.neo4j_password,
            embedding=self.embedding_model,
            index_name="pdf_bot",
            node_label="PdfBotChunk",
            pre_delete_collection=False, #É false não?
        )

def main():
    # Loads environment variables and initializes the PDF to Neo4j processor
    load_dotenv(".env")

    pdf_dir = "/Users/hugomiguel/Downloads/PDFs\ sharepoint\ Magicbeans\ 23\ Jan\ 2014"
    neo4j_uri = os.getenv("NEO4J_URI")
    neo4j_username = os.getenv("NEO4J_USERNAME")
    neo4j_password = os.getenv("NEO4J_PASSWORD")
    embedding_model_name = os.getenv("EMBEDDING_MODEL")

    logger = get_logger(__name__)
    embeddings, dimension = load_embedding_model(
        embedding_model_name, config={"ollama_base_url": os.getenv("OLLAMA_BASE_URL")}, logger=logger
    )

    pdf_processor = PDFtoNeo4j(pdf_dir, neo4j_uri, neo4j_username, neo4j_password, embeddings, logger)
    pdf_processor.process_pdfs()

if __name__ == "__main__":
    main()
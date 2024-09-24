import getpass
from dotenv import load_dotenv
import streamlit as st
import os
from PyPDF2 import PdfReader
from langchain.schema import Document
from langchain.text_splitter import CharacterTextSplitter
import getpass
import cassio
from ragstack_knowledge_graph.cassandra_graph_store import CassandraGraphStore
from langchain_experimental.graph_transformers import LLMGraphTransformer
from ragstack_knowledge_graph.cassandra_graph_store import CassandraGraphStore
from langchain_openai import OpenAIEmbeddings
from langchain.document_loaders import PDFPlumberLoader
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_openai import ChatOpenAI
from langchain_openai import AzureOpenAIEmbeddings
from langchain.vectorstores import AstraDB
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.document_loaders import PDFPlumberLoader
from langchain.document_loaders import PyPDFLoader
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_openai import ChatOpenAI
from langchain_openai import AzureChatOpenAI
from ragstack_colbert import CassandraDatabase, ColbertEmbeddingModel, ColbertVectorStore

# Set environment variables from st.secrets
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
os.environ["ASTRA_DB_DATABASE_ID"] = st.secrets["ASTRA_DB_DATABASE_ID"]
os.environ["ASTRA_DB_APPLICATION_TOKEN"] = st.secrets["ASTRA_DB_APPLICATION_TOKEN"]
os.environ["ASTRA_DB_ENDPOINT"] = st.secrets["ASTRA_DB_ENDPOINT"]

# Initialize components
#llm = ChatOpenAI(temperature=0, model_name="gpt-4")
AZURE_OPENAI_API_VERSION= '2023-06-01-preview'
llm = AzureChatOpenAI(
    openai_api_version='2023-06-01-preview',
    #azure_deployment="abptest"
    azure_deployment="OpenAIKey"
)

cassio.init(auto=True)
#text_embeddings = OpenAIEmbeddings()
text_embeddings = AzureOpenAIEmbeddings(
    azure_deployment="Embedding",
    openai_api_version="2023-05-15",
)
graph_store = CassandraGraphStore(text_embeddings=text_embeddings)
llm_transformer = LLMGraphTransformer(llm=llm)


vstore = AstraDB(
    embedding=text_embeddings,
    collection_name="mrrbot",
    api_endpoint=os.environ["ASTRA_DB_ENDPOINT"],
    token=os.environ["ASTRA_DB_APPLICATION_TOKEN"],
)

def save_uploaded_file(uploaded_file):
    temp_dir = "tempDir"
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    file_path = os.path.join(temp_dir, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path

def pdf_to_text_files(pdf_path, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    reader = PdfReader(pdf_path)
    num_pages = len(reader.pages)
    for i in range(num_pages):
        page = reader.pages[i]
        text = page.extract_text()
        with open(os.path.join(output_dir, f"page_{i + 1}.txt"), "w", encoding="utf-8") as f:
            f.write(text)

def process_with_langchain(output_dir):
    documents = []
    for filename in sorted(os.listdir(output_dir)):
        if filename.endswith(".txt"):
            with open(os.path.join(output_dir, filename), "r", encoding="utf-8") as f:
                text = f.read()
                doc = Document(page_content=text, metadata={"filename": filename})
                documents.append(doc)
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = []
    for doc in documents:
        chunks.extend(text_splitter.split_text(doc.page_content))
    return chunks

def read_text_files(directory):
    texts = []
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            with open(os.path.join(directory, filename), 'r', encoding='utf-8') as file:
                texts.append(file.read())
    return texts

st.title("PDF Processor")

uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
output_dir = "output_text_files"

if uploaded_file is not None:
    st.success("File uploaded successfully!")
    pdf_path = save_uploaded_file(uploaded_file)

    if st.button("Load Data into Graph Store"):
        pdf_to_text_files(pdf_path, output_dir)
        chunks = process_with_langchain(output_dir)
        documents = [Document(page_content=chunk) for chunk in chunks]
        graph_documents = llm_transformer.convert_to_graph_documents(documents)
        graph_store.add_graph_documents(graph_documents)
        st.success("Data loaded into Graph Store successfully!")

    if st.button("Load Data into Vector Store (LangChain)"):
        pdf_to_text_files(pdf_path, output_dir)
        chunks = process_with_langchain(output_dir)
        documents = [Document(page_content=chunk) for chunk in chunks]
        inserted_ids_from_pdf = vstore.add_documents(documents)
        st.success(f"Inserted {len(inserted_ids_from_pdf)} documents into Vector Store (LangChain) successfully!")

    if st.button("Load Data into Vector Store (ColBERT)"):
        pdf_to_text_files(pdf_path, output_dir)
        texts = read_text_files(output_dir)
        database = CassandraDatabase.from_astra(
            astra_token=os.environ["ASTRA_DB_APPLICATION_TOKEN"],
            database_id=os.environ["ASTRA_DB_DATABASE_ID"],
            keyspace='default_keyspace'
        )
        embedding_model = ColbertEmbeddingModel()
        vector_store = ColbertVectorStore(database=database, embedding_model=embedding_model)
        results = vector_store.add_texts(texts=texts, doc_id="livs")
        st.success("Data loaded into Vector Store (ColBERT) successfully!")

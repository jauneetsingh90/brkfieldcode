import streamlit as st
from operator import itemgetter
from ragstack_knowledge_graph.traverse import Node
from ragstack_knowledge_graph import extract_entities
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import AzureOpenAIEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from ragstack_knowledge_graph.cassandra_graph_store import CassandraGraphStore
from langchain_community.vectorstores import AstraDB
import cassio
from langchain_openai import AzureChatOpenAI
import os
from ragstack_colbert import CassandraDatabase, ColbertEmbeddingModel, ColbertVectorStore
from ragstack_langchain.colbert import ColbertVectorStore as LangchainColbertVectorStore


# Load environment variables from st.secrets
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
os.environ["ASTRA_DB_DATABASE_ID"] = st.secrets["ASTRA_DB_DATABASE_ID"]
os.environ["ASTRA_DB_APPLICATION_TOKEN"] = st.secrets["ASTRA_DB_APPLICATION_TOKEN"]
os.environ["ASTRA_DB_ENDPOINT"] = st.secrets["ASTRA_DB_ENDPOINT"]

os.environ["LANGCHAIN_PROJECT"] = "test-ls-lc"
os.environ["LANGCHAIN_TRACING_V2"] = "true"

# Initialize components
#cassio.init(auto=True)

AZURE_OPENAI_API_VERSION= '2023-06-01-preview'
llm = AzureChatOpenAI(
    openai_api_version='2023-06-01-preview',
    azure_deployment="OpenAIKey"
)

cassio.init(auto=True)
#text_embeddings = OpenAIEmbeddings()
text_embeddings = AzureOpenAIEmbeddings(
    azure_deployment="Embedding",
    openai_api_version="2023-05-15",
)
#llm = ChatOpenAI(model_name="gpt-4")
#text_embeddings = OpenAIEmbeddings()
graph_store = CassandraGraphStore(text_embeddings=text_embeddings)
vstore = AstraDB(
    embedding=text_embeddings,
    collection_name="fhome",
    api_endpoint=os.environ["ASTRA_DB_ENDPOINT"],
    token=os.environ["ASTRA_DB_APPLICATION_TOKEN"],
)
database = CassandraDatabase.from_astra(
    astra_token=os.environ["ASTRA_DB_APPLICATION_TOKEN"],
    database_id=os.environ["ASTRA_DB_DATABASE_ID"],
    keyspace='default_keyspace'
)
embedding_model = ColbertEmbeddingModel()

vector_store =LangchainColbertVectorStore(
    database=database,
    embedding_model=embedding_model,
)
#vector_store = ColbertVectorStore(database=database, embedding_model=embedding_model)


def vector_search_prompt(user_query):
    return f"Search for any context related to the reports, We can scan through the documents\n\nQuery: \"{user_query}\""

def combined_search_prompt(user_query, graph_triples, vector_results, colbert_results):
    return (
        f"The user asked about reports loaded, their scope, what is information in reports etc Below is the retrieved information from the knowledge graph, vector search results, and Colbert search results. Use this to answer the user's question.\n\n"
        f"Original Question: {user_query}\n\n"
        f"Knowledge Graph Triples:\n{graph_triples}\n\n"
        f"Vector Search Results:\n{vector_results}\n\n"
        f"Colbert Search Results:\n{colbert_results}\n\n"
        f"Response:"
    )

def _combine_relations(relations):
    return "\n".join(map(repr, relations))

ANSWER_PROMPT = (
    "This question has been asked to retrieve information from a knowledge graph The below question will get triplets from context got from database.Use the information in the triples to answer the original question.\n\n"
    "Original Question: {{question}}\n\n"
    "Knowledge Graph Triples:\n{{context}}\n\n"
    "Response:"
)

graph_chain = (
    {"question": RunnablePassthrough()}
    | RunnablePassthrough.assign(entities=extract_entities(llm))
    | RunnablePassthrough.assign(triples=itemgetter("entities") | graph_store.as_runnable())
    | RunnablePassthrough.assign(context=itemgetter("triples") | RunnableLambda(_combine_relations))
    | ChatPromptTemplate.from_messages([ANSWER_PROMPT])
    | llm
)

# Define functions for retrieving data
def query_graph(question):
    result = graph_chain.invoke({"question": question})
    return result

def query_vector_store(question):
    prompt = vector_search_prompt(question)
    results = vstore.similarity_search(prompt, k=3)
    return results

def query_colbert_vector_store(question):
    prompt = vector_search_prompt(question)
    results = vector_store.similarity_search(prompt, k=3)
    return results

def query_combined(question):
    graph_answer = query_graph(question)
    graph_answer_str = str(graph_answer)[:500]  # Convert to string and limit the length
    
    vector_results = query_vector_store(question)
    result_strings = [res.page_content[:500] for res in vector_results]  # Limit the length of each result
    text_result = "\n\n ".join(result_strings[:2])  # Use only the top 2 results to limit length

    colbert_results = query_colbert_vector_store(question)
    colbert_result_strings = [res.page_content[:500] for res in colbert_results]  # Limit the length of each result
    colbert_text_result = "\n\n ".join(colbert_result_strings[:2])  # Use only the top 2 results to limit length

    combined_prompt = combined_search_prompt(question, graph_answer_str, text_result, colbert_text_result)
    combined_chain = (
        {"context": RunnablePassthrough(), "question": RunnablePassthrough()}
        | ChatPromptTemplate.from_messages([combined_prompt])
        | llm
        | StrOutputParser()
    )
    rag_output = combined_chain.invoke({"question": question, "context": text_result + "\n\n" + colbert_text_result})
    return rag_output

# Streamlit app
st.title("MR Query Support Bot")

question = st.text_input("Enter your question:")

if st.button("Query Knowledge Graph"):
    if question:
        answer = query_graph(question)
        st.markdown(f"**Answer from Knowledge Graph:** {answer}")
    else:
        st.warning("Please enter a question.")

if st.button("Query Vector Store"):
    if question:
        results = query_vector_store(question)
        result_strings = [res.page_content[:500] for res in results]
        text_result = "\n\n ".join(result_strings[:2])
        st.markdown(f"**Answer from Vector Store:**\n{text_result}")
    else:
        st.warning("Please enter a question.")

if st.button("Query Colbert Store"):
    if question:
        results = query_colbert_vector_store(question)
        result_strings = [res.page_content[:500] for res in results]
        text_result = "\n\n ".join(result_strings[:2])
        st.markdown(f"**Answer from Colbert Store:**\n{text_result}")
    else:
        st.warning("Please enter a question.")

if st.button("Query Combined"):
    if question:
        answer = query_combined(question)
        st.markdown(f"**Answer from Combined Method:** {answer}")
    else:
        st.warning("Please enter a question.")
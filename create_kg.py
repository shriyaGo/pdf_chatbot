from dotenv import load_dotenv
import os
from langchain_community.graphs import Neo4jGraph
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_openai import ChatOpenAI;
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Neo4jVector

load_dotenv() #load environment vars

NEO4J_URI = os.environ["NEO4J_URI"]
NEO4J_USERNAME = os.environ["NEO4J_USERNAME"]
NEO4J_PASSWORD = os.environ["NEO4J_PASSWORD"]
NEO4J_DATABASE = os.environ["NEO4J_DATABASE"]

#knowledge graph object
kg = Neo4jGraph(
    url=NEO4J_URI,
    username=NEO4J_USERNAME,
    password=NEO4J_PASSWORD,
    database=NEO4J_DATABASE,
)


llm = ChatOpenAI(api_key=os.environ['OPENAI_API_KEY'])

llm_transformer = LLMGraphTransformer(llm=llm)

INDEX_NAME = 'VECTOR_SEARCH_INDEX'
ENTITY_INDEX_NAME = 'ENTITY_VECTOR_SEARCH'

def createHybridIndex():
  vector_index = Neo4jVector.from_existing_graph(
    OpenAIEmbeddings(), 
    search_type="hybrid", # hybrid search means the search will be vector based as well as keyword based
    node_label="Document", # Node label to create index on
    index_name=INDEX_NAME, #index name
    text_node_properties=["text"],  # which property of node is used to create index on
    embedding_node_property="embedding",# embedding will be stored in this propert
)
  return vector_index

def add_all_docs_to_graph(docs):
    graph_documents = llm_transformer.convert_to_graph_documents(docs)
    return kg.add_graph_documents(
    graph_documents,
    include_source=True,
    baseEntityLabel=True,
)

def createEntityIndex():
  kg.query(
   f"CREATE FULLTEXT INDEX {ENTITY_INDEX_NAME} IF NOT EXISTS FOR (e:__Entity__) ON EACH [e.id]")

def get_text_from_files(filePath):
    text = PyPDFLoader(filePath).load()
    text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=5000,
    chunk_overlap=20,
    length_function=len,
    is_separator_regex=False,
    )
    docs = text_splitter.split_documents(text)
    add_all_docs_to_graph(docs)
    createHybridIndex()
    createEntityIndex()






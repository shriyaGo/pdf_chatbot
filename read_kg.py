from langchain_community.graphs import Neo4jGraph
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import os
from langchain_community.graphs import Neo4jGraph
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_openai import ChatOpenAI;
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Neo4jVector
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import Tuple, List
from langchain.prompts import PromptTemplate

from langchain_core.runnables import (
    RunnableBranch,
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)
from langchain_community.vectorstores import Neo4jVector
from langchain_openai import OpenAIEmbeddings
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.prompts import ChatPromptTemplate
from langchain_community.vectorstores.neo4j_vector import remove_lucene_chars
from langchain_core.messages import AIMessage, HumanMessage
import warnings
warnings.filterwarnings("ignore")
INDEX_NAME = 'VECTOR_SEARCH_INDEX'
ENTITY_INDEX_NAME = 'ENTITY_VECTOR_SEARCH'



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




# This class used to instruct LLM to provide answer in a specific format. This is needed when we want the pass 
# the output of this llm to some other downstream
class NamesEntitiy(BaseModel):

    names: List[str] = Field(
        ...,
        description="All the person, character,concept, creature, Ghost, school house, Train, Vehicle organization, or business entities that "
        "appear in the text",
    )

def get_hybrid_index():
  return Neo4jVector.from_existing_index(embedding=OpenAIEmbeddings(), index_name=INDEX_NAME, search_type="hybrid", keyword_index_name='keyword' )




def get_chain_to_extract_entity(question):
    # this will extract  entity from the qiven question start
    entityPrompt = ChatPromptTemplate.from_messages(
        [
            (
             "system",
                """ you are extracting Entity Information Source, Factor, Contact, Person and Concept""",
            ),
            (
                "human",
               "return entity for question: {question}"

            ),
        ]
        )

    return  entityPrompt | llm.with_structured_output(NamesEntitiy)

def generate_full_text_query(input: str) -> str:
    """
    Generate a full-text search query for a given input string.

    This function constructs a query string suitable for a full-text search.
    It processes the input string by splitting it into words and appending a
    similarity threshold (~2 changed characters) to each word, then combines 
    them using the AND operator. Useful for mapping entities from user questions
    to database values, and allows for some misspelings.
    """
    full_text_query = ""
    words = [el for el in remove_lucene_chars(input).split() if el]
    for word in words[:-1]:
        full_text_query += f" {word}~2 AND"
    full_text_query += f" {words[-1]}~2"
    return full_text_query.strip()

def retrive_all_node_for_entity_in_question(question, entity_chain) -> str:
    print('===enter====')
    """
    Collects the neighborhood of entities mentioned
    in the question
    """
    result = ""
    entities = entity_chain.invoke({"question": question});
    result = (search_on_node(entities.names),checking(entities.names))
    print('exit')
    return result


def search_on_node(entities):
    result = ''
    for entity in entities:
        response = kg.query("""
                         CALL db.index.fulltext.queryNodes('ENTITY_VECTOR_SEARCH', $query, {limit:10})
YIELD node
CALL {
    WITH node
    MATCH (node)-[r:!MENTIONS]->(ne)
    RETURN DISTINCT node.id + ' - ' + type(r) + ' -> ' + ne.id AS output
    UNION
    WITH node
    MATCH (node)<-[r:!MENTIONS]-(ne)
    RETURN DISTINCT node.id + ' - ' + type(r) + ' -> ' + ne.id AS output
}
RETURN output LIMIT 10   
                            """,
                            {"query": generate_full_text_query(entity)} )
        result += "\n".join([el['output'] for el in response])
    return result;


     
def getunstructuredData(question):
    
    index =  Neo4jVector.from_existing_index(embedding=OpenAIEmbeddings(), index_name='VECTOR_SEARCH_INDEX' )
    result = index.similarity_search(question, k=2)
    response = []
    for el in result:
        response.append(el.page_content)
    return response

def checking(entities) -> str:
    """
    Collects the neighborhood of entities mentioned
    in the question
    """
    result = ""
    for entity in entities:
        response = kg.query(
            """CALL db.index.fulltext.queryNodes("keyword", $query, {limit:2})
YIELD node, score
CALL {
    WITH node
    MATCH (node)-[r:MENTIONS]->(neighbor)
    RETURN node.id + ' - ' + type(r) + ' -> ' + neighbor.id AS output
}
RETURN output LIMIT 10
            """,
            {"query": generate_full_text_query(entity)},
        )
        result += "\n".join([el['output'] for el in response])
    return result



def perform_search_on_kw_and_vector(question):
    entity_extraction_chain = get_chain_to_extract_entity(question)
    def inner(question):
      (structured_data,unstrucured_data)  = retrive_all_node_for_entity_in_question(question, entity_extraction_chain)
      unstrucured_data_context= "".join(getunstructuredData(question))
      final_data = f"""Structured data:
         {structured_data}

       Unstructured data:
      {unstrucured_data}


     More Context:
{unstrucured_data_context}

    """
     # print(final_data)
      return final_data
    return inner


def get_answer(question):
    _search_query =  RunnableLambda(lambda x: x["question"])

    template = """You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise. 

Question: {question} 

Context: {context} 

Answer:

"""  
    prompt = PromptTemplate.from_template(template)
    chain = (
    RunnableParallel(
        {
            "context": _search_query | perform_search_on_kw_and_vector(question),
            "question": RunnablePassthrough(),
        }
    )|prompt| llm
    | StrOutputParser()
    )
    return chain.invoke(input={'question':question})
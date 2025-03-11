import os
import time
from dotenv import load_dotenv
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_neo4j import Neo4jGraph
from langchain_community.vectorstores import Neo4jVector
from langchain_community.document_loaders import TextLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_experimental.llms.ollama_functions import OllamaFunctions
from langchain_community.vectorstores.neo4j_vector import remove_lucene_chars
import psutil
from pydantic import BaseModel, Field
from neo4j import GraphDatabase
from JSONReader import load_data


# Load environment variables
load_dotenv()

documents = load_data('data/corpus.json')
CHUNKS_SIZE = 100
CHUNKS_OVERLAP = 24

graph = Neo4jGraph()

def chunk():
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNKS_SIZE,
        chunk_overlap=CHUNKS_OVERLAP,
        length_function=len,
        is_separator_regex=False,
    )
    chunked_documents = []
    chunk_ids = []
    id_counter = 1

    for document in documents:
        chunks = text_splitter.split_text(document.text)
        for chunk in chunks:
            chunked_documents.append(
                Document(page_content=chunk, metadata=document.metadata)
            )
            chunk_ids.append(str(id_counter))  
            id_counter += 1
    return chunked_documents, chunk_ids  

def get_embedding_function():
    return HuggingFaceEmbeddings(model_name="BAAI/bge-m3")

def track_usage():
    """Returns current CPU and RAM usage"""
    process = psutil.Process()
    return {
        "cpu_percent": process.cpu_percent(interval=1),
        "memory_mb": process.memory_info().rss / (1024 * 1024)  # Convert bytes to MB
    }


def convert_to_graph(chunked_documents):
    start_time = time.time()
    usage_before = track_usage()
    llm = ChatOllama(model="mistral", temperature=0)
    llm_transformer = LLMGraphTransformer(llm=llm)
    graph_documents = llm_transformer.convert_to_graph_documents(chunked_documents)
    graph.add_graph_documents(
        graph_documents,
        baseEntityLabel=True,
        include_source=True
    )

    end_time = time.time()
    usage_after = track_usage()
    
    # Calculate metrics
    elapsed_time = end_time - start_time
    cpu_usage = usage_after["cpu_percent"] - usage_before["cpu_percent"]
    memory_usage = usage_after["memory_mb"] - usage_before["memory_mb"]

    # Print Metrics
    print(f"Execution Time: {elapsed_time:.2f} sec")
    print(f"CPU Usage: {cpu_usage:.2f}%")
    print(f"Memory Usage: {memory_usage:.2f} MB")

    vector_index = Neo4jVector.from_existing_graph(
        get_embedding_function(),
        search_type="hybrid",
        node_label="Document",
        text_node_properties=["text"],
        embedding_node_property="embedding"
    )
    return vector_index.as_retriever()

driver = GraphDatabase.driver(
    uri=os.environ["NEO4J_URI"],
    auth=(os.environ["NEO4J_USERNAME"], os.environ["NEO4J_PASSWORD"])
)
def clear_database():
    with GraphDatabase.driver(
        uri=os.environ["NEO4J_URI"],
        auth=(os.environ["NEO4J_USERNAME"], os.environ["NEO4J_PASSWORD"])
    ) as temp_driver:
        with temp_driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n;")
            print("Database cleared successfully.")

def create_fulltext_index(tx):
    tx.run('''CREATE FULLTEXT INDEX `fulltext_entity_id` FOR (n:__Entity__) ON EACH [n.id];''')

def create_index():
    with driver.session() as session:
        session.execute_write(create_fulltext_index)
        print("Fulltext index created successfully.")
try:
    create_index()
except:
    pass

driver.close()

class Entities(BaseModel):
    names: list[str] = Field(..., description="List of entities in the text.")

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are extracting organization and person entities from the text."),
    ("human", "Use the given format to extract information from the following input: {question}")
])

entity_chain = ChatOllama(model="mistral", temperature=0).with_structured_output(Entities)

def generate_full_text_query(input: str) -> str:
    words = [el for el in remove_lucene_chars(input).split() if el]
    return " AND ".join([f"{word}~2" for word in words]) if words else ""

def graph_retriever(question: str) -> str:
    result = ""
    entities = entity_chain.invoke(question)
    for entity in entities.names:
        response = graph.query(
            """
            CALL db.index.fulltext.queryNodes('fulltext_entity_id', $query, {limit:2})
            YIELD node,score
            CALL {
              WITH node
              MATCH (node)-[r:!MENTIONS]->(neighbor)
              RETURN node.id + ' - ' + type(r) + ' -> ' + neighbor.id AS output
              UNION ALL
              WITH node
              MATCH (node)<-[r:!MENTIONS]-(neighbor)
              RETURN neighbor.id + ' - ' + type(r) + ' -> ' + node.id AS output
            }
            RETURN output LIMIT 50
            """,
            {"query": entity},
        )
        result += "\n".join([el['output'] for el in response])
    return result

def full_retriever(question: str):
    graph_data = graph_retriever(question)
    vector_data = [el.page_content for el in vector_retriever.invoke(question)]
    return f"""Graph data:\n{graph_data}\nvector data:\n{"#Document ".join(vector_data)}"""

template = """Answer the question based only on the following context:\n{context}\n\nQuestion: {question}\nUse natural language and be concise.\nAnswer:"""
prompt = ChatPromptTemplate.from_template(template)

chain = ({
        "context": full_retriever,
        "question": RunnablePassthrough(),
    } | prompt | ChatOllama(model="mistral", temperature=0) | StrOutputParser())

if __name__ == "__main__":
    #clear_database()

    docs, _ = chunk()
    vector_retriever = convert_to_graph(docs)
    print(chain.invoke(input="What does John work as?"))

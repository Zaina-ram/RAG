import json
import os
import threading
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
os.environ.clear()
load_dotenv()


# Load documents from JSON
documents = load_data('data/corpus.json')
CHUNK_SIZE = 2048
CHUNK_OVERLAP = 24

performance_data = []
monitoring = True

graph = Neo4jGraph()

def chunk_documents():
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        is_separator_regex=False,
    )
    chunked_docs = []
    for document in documents:
        chunks = text_splitter.split_text(document.text)
        for chunk in chunks:
            chunked_docs.append(Document(page_content=chunk, metadata=document.metadata))
    return chunked_docs  

def get_embedding_function():
    return HuggingFaceEmbeddings(model_name="BAAI/bge-m3")

def monitor_performance():
    """Continuously logs CPU and RAM usage while the script runs."""
    while monitoring:
        cpu_usage = psutil.cpu_percent(interval=5)  # Sampling every 0.5 sec
        ram_usage = psutil.virtual_memory().percent
        performance_data.append({
            "cpu": cpu_usage,
            "ram": ram_usage
        })

def build_graph():
    docs = chunk_documents()
    
    monitor_thread = threading.Thread(target=monitor_performance, daemon=True)
    monitor_thread.start()
    start_time = time.time()
    #usage_before = track_usage()
    
    llm = ChatOllama(model="mistral", temperature=0)
    llm_transformer = LLMGraphTransformer(llm=llm)
    graph_docs = llm_transformer.convert_to_graph_documents(docs)
    graph.add_graph_documents(graph_docs, baseEntityLabel=True, include_source=True)
    
    global monitoring
    monitoring = False
    monitor_thread.join()
    end_time = time.time()
    execution_time = end_time - start_time

    #usage_after = track_usage()
    
    with open("eval/knowledge_graph_CPU_RAM.json", 'w') as json_file:
        json.dump(performance_data, json_file, indent=4)

    print(f"Execution Time: {execution_time:.2f} sec")
    #print(f"CPU Usage: {usage_after['cpu_percent'] - usage_before['cpu_percent']:.2f}%")
    #print(f"Memory Usage: {usage_after['memory_mb'] - usage_before['memory_mb']:.2f} MB")

def setup_vector_retriever():
    return Neo4jVector.from_existing_graph(
        get_embedding_function(),
        search_type="hybrid",
        node_label="Document",
        text_node_properties=["text"],
        embedding_node_property="embedding"
    ).as_retriever()

driver = GraphDatabase.driver(
    uri=os.environ["NEO4J_URI"],
    auth=(os.environ["NEO4J_USERNAME"], os.environ["NEO4J_PASSWORD"])
)

def clear_database():
    with driver.session() as session:
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

entity_chain = ChatOllama(model="mistral", temperature=0).with_structured_output(Entities)

def graph_retriever(question: str) -> str:
    result = ""
    entities = entity_chain.invoke(question)
    for entity in entities.names:
        query = """
        CALL db.index.fulltext.queryNodes('fulltext_entity_id', $query, {limit:2})
        YIELD node, score
        MATCH (node)-[r]->(neighbor)
        RETURN node.id AS source, type(r) AS relation, neighbor.id AS target
        LIMIT 50
        """
        response = graph.query(query, {"query": entity})
        for row in response:
            result += f"{row['source']} - {row['relation']} -> {row['target']}\n"
    print(result)
    return result if result else "No relevant graph data found."

vector_retriever = setup_vector_retriever()

def full_retriever(question: str):
    graph_data = graph_retriever(question)
    vector_data = [el.page_content for el in vector_retriever.invoke(question)]
    return f"""Graph data:\n{graph_data}\nVector data:\n{'#Document '.join(vector_data)}"""

template = """Answer the question based only on the following context:\n{context}\n\nQuestion: {question}\nUse natural language and be concise.\nAnswer:"""
prompt = ChatPromptTemplate.from_template(template)

chain = ({
    "context": full_retriever,
    "question": RunnablePassthrough(),
} | prompt | ChatOllama(model="mistral", temperature=0) | StrOutputParser())

if __name__ == "__main__":
    #clear_database()
    build_graph()
    
    #question = "What does John work as?"
    #print(chain.invoke(input=question))

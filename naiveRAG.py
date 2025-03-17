import json
from langchain.schema import Document
from JSONReader import load_data
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM

documents = load_data('data/corpus.json')
CHROMA_PATH = "chroma"

CHUNKS_SIZE = 1024
CHUNKS_OVERLAP = 24

save_file = []  # List to store results

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

def embedd_and_store(chunked_documents, chunk_ids):
    db = Chroma(
        collection_name="documents", 
        embedding_function=get_embedding_function(), 
        persist_directory=CHROMA_PATH
    )
    db.add_documents(documents=chunked_documents, ids=chunk_ids)

def get_embedding_function():
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")    
    return embeddings

def retrieve_context(query, top_k=3):
    db = Chroma(
        collection_name="documents",
        embedding_function=get_embedding_function(),
        persist_directory=CHROMA_PATH
    )
    retrieval_list = []
    results = db.similarity_search(query, k=top_k)

    for doc in results:
        retrieval_list.append({
            "title": doc.metadata.get("title", "Unknown Title"),
            "url": doc.metadata.get("url", "Unknown URL"),
            "source": doc.metadata.get("source", "Unknown Source"),
            "published_at": doc.metadata.get("published_at", "Unknown Date"),
            "fact": doc.page_content
        })

    return retrieval_list

def generate_response(query):
    evidence_list = retrieve_context(query)

    context = "\n".join([doc["fact"] for doc in evidence_list])
    prompt = f"prefix = Below is a question followed by some context from different sources. Please answer the question based on the context. The answer to the question is a word or entity ONLY. If the provided information is insufficient to answer the question, respond 'Insufficient Information' only. For yes/no answers, answer only with yes or no without further explaination. \n\n{context}\n\nQuestion: {query}\nAnswer:"

    llm = OllamaLLM(model="mistral", temperature=0)
    response = llm.invoke(prompt)

    return response, evidence_list

def generate_and_save_results():
    i = 0
    with open('eval_test.json') as data_file:
        data = json.load(data_file)

    for item in data:
        print("Processing item:", item)
        query = item['query']
        question_type = item['question_type']
        answer, evidence_list = generate_response(query)

        save = {
            "query": query,
            "answer": answer.strip(),
            "question_type": question_type,
            "evidence_list": evidence_list
        }

        save_file.append(save)

    with open("naiveRAG_results.json", 'w') as json_file:
        json.dump(save_file, json_file, indent=4)

generate_and_save_results()
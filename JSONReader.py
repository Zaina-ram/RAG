from typing import List
from llama_index import Document
import json

# Source: https://github.com/yixuantt/MultiHop-RAG/blob/main/util.py#L81

def load_data(input_file: str) -> List[Document]:
        """Load data from the input file."""
        documents = []
        with open(input_file, 'r') as file:
            load_data = json.load(file)
        for data in load_data:
            metadata = {"title": data['title'],"url": data['url'], "published_at": data['published_at'],"source":data['source']}
            documents.append(Document(text=data['body'], metadata=metadata))
        
        #print(len(documents))
        return documents
    
#load_data("data/corpus.json")
#!/usr/bin/env python
# coding: utf-8

# ติดตั้ง Library
import subprocess

subprocess.run(['pip', 'install', '--upgrade', 'imageio'])
subprocess.run(['pip', 'install', 'pillow==10.3.0'])
subprocess.run(['pip', 'install', '-q', 'llama-index', 'google-generativeai', 'llama-index-llms-gemini'])
subprocess.run(['pip', 'install', '-q', 'llama-index-embeddings-huggingface'])
subprocess.run(['pip', 'install', '-q', 'datasets', 'deeplake', 'llama-index-vector-stores-deeplake'])

# การตั้งค่า LLM
from google.colab import userdata
from llama_index.llms.gemini import Gemini
import google.generativeai as genai

key = userdata.get('GEMINI_API_KEY')
llm = Gemini(api_key=key, model="models/gemini-1.5-pro-latest")
print(llm)

res = llm.complete('อธิบายเกี่ยวกับระยอง')
print(res.text)

# การทำ Embedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
x = embed_model._get_text_embedding('อธิบายเกี่ยวกับระยอง')
print(type(x))
print(len(x))

# การ Splitter text document
from llama_index.core import Document, VectorStoreIndex, StorageContext
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.deeplake import DeepLakeVectorStore
from datasets import load_dataset
import pandas as pd
import json

# โหลดข้อมูล JSON
json_file_path = '/content/data.json'
with open(json_file_path, 'r', encoding='utf-8') as file:
    data = json.load(file)

# แปลงข้อมูล JSON เป็น DataFrame
json_data = data['data']
df = pd.DataFrame(json_data)

# ฟังก์ชันเพื่อให้แน่ใจว่าค่าทั้งหมดเป็นสตริง
def ensure_string(value):
    if isinstance(value, list):
        return ' '.join([json.dumps(item) if isinstance(item, dict) else str(item) for item in value])
    elif isinstance(value, dict):
        return json.dumps(value)
    else:
        return str(value)

# ใช้ฟังก์ชันกับคอลัมน์ 'question' และ 'answer'
df['question'] = df['question'].apply(ensure_string)
df['answer'] = df['answer'].apply(ensure_string)

# รวม 'question' และ 'answer' เป็นหนึ่ง text field และสร้าง Document objects
documents = [Document(text=row['question'] + " " + row['answer']) for index, row in df.iterrows()]

# แสดงเนื้อหาของ Document objects
for doc in documents[:5]:
    print(doc.text)

print(len(documents))
print(documents[0])

# การแปลงเอกสารเป็น nodes
parser = SentenceSplitter()
nodes = parser.get_nodes_from_documents(documents)

print(len(nodes))
print(nodes[0])
print(nodes[1])

# การ import to vector store
vector_store = DeepLakeVectorStore(dataset_path='./deeplake')
storage_context = StorageContext.from_defaults(vector_store=vector_store)

index = VectorStoreIndex(
    nodes,
    embed_model=embed_model,
    storage_context=storage_context,
    show_progress=True
)

query_engine = index.as_query_engine(llm=llm)

res = query_engine.query("วิชาที่ต้องเรียนปี1เทอม1เรียนอะไรบ้าง")
print(res.source_nodes[0].text)

chat_engine = index.as_query_engine(llm=llm)

res = chat_engine.query('วิชาที่ต้องเรียน ปี 1 เทอม 1 มีวิชาอะไรบ้าง ครับ/คะ ?')
print(res)
print(res.source_nodes[1].text)





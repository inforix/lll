from langchain.embeddings import OpenAIEmbeddings
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index import LangchainEmbedding

# Load embedding
def load_embedding(embedding_source:str = "huggingface"):
  if embedding_source == "openai":
    return OpenAIEmbeddings()
  else:
    llama_model_path = "../models/all-mpnet-base-v2"
    embed_model = HuggingFaceEmbeddings(model_name=llama_model_path)
    return embed_model
  
embedding = load_embedding()
query_result = embedding.embed_query("上海海事大学")
len(query_result)

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import DirectoryLoader
from typing import List
from llama_index.readers.schema.base import Document

def load_documents():
  loader = DirectoryLoader("../data/", "**/*.txt")
  documents = loader.load()
  text_splitter = RecursiveCharacterTextSplitter(        
   chunk_size = 1000,
   chunk_overlap  = 20,
  )
  texts = text_splitter.split_documents(documents)
  return texts

docs = load_documents()
print(len(docs))

from langchain.vectorstores import FAISS, Chroma

index = FAISS.from_documents(docs, embedding)

def get_similiar_docs(query, k=3, score=False):
  if score:
    similar_docs = index.similarity_search_with_score(query, k=k)
  else:
    similar_docs = index.similarity_search(query, k=k)
  
  #print(similar_docs)
  return similar_docs


similar_docs = get_similiar_docs("领导干部离沪外出请假报告相关的规章制度有哪些？", score=True)
print(similar_docs)

from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
import os

os.environ['OPENAI_API_KEY'] = "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"
os.environ['HTTPS_PROXY']="http://10.81.38.5:8443"
# model_name = "text-davinci-003"
model_name = "gpt-3.5-turbo"
# model_name = "gpt-4"

llm = OpenAI(model_name=model_name)

chain = load_qa_chain(llm, chain_type="stuff")

def get_answer(query):
  similar_docs = get_similiar_docs(query)
  answer = chain.run(input_documents=similar_docs, question=query)
  return answer

print(get_answer("上海海事大学有多少毕业生？"))
print(get_answer("上海海事大学有马克思主义学院吗？"))
print(get_answer("通知公告的主管部门是？"))
print(get_answer("离沪外出请假报告相关的规章制度有哪些？"))

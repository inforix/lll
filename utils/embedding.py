# Load embedding
from langchain.embeddings import OpenAIEmbeddings
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.embeddings.base import Embeddings
from pydantic import BaseModel, Extra, Field
from typing import Any, List
from gensim.models import KeyedVectors

# Load embedding
def load_embedding(embedding_source:str = "huggingface"):
  if embedding_source == "openai":
    return OpenAIEmbeddings()
  elif embedding_source == "huggingface":
    #llama_model_path = "../models/all-mpnet-base-v2"
    llama_model_path = "../models/text2vec-large-chinese"
    llama_model_path = "../models/chinese-roberta-wwm-ext-large"
    embed_model = HuggingFaceEmbeddings(model_name=llama_model_path)
    return embed_model

class TencentEmbeddings(BaseModel, Embeddings):
  wv:KeyedVectors
  
  def __init__(self, path:str, **kwargs: Any):
    self.wv.load_word2vec_format(path, binary=True)
  
  class Config:
    extra = Extra.forbid
  
  def embed_query(self, text:str) -> List[float]:
    text = text.replace("\n", "").replace(" ", "")
    embeddings = [self.wv[t] for t in text]
    return embeddings
  
  def embed_query(self, texts: List[str]) -> List[List[float]]:
    texts = list(map(lambda x: x.replace("\n", "").replace(" ", ""), texts))
    embeddings = [self.embed_query(x) for x in texts]
    return embeddings
    
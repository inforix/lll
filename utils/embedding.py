from langchain.embeddings import OpenAIEmbeddings
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index import LangchainEmbedding
from typing import Any, Dict, List, Optional
from langchain.embeddings.base import Embeddings
import numpy as np

def read_vectors(path):  # read top n word vectors, i.e. top is 10000
  lines_num, dim = 0, 0
  vectors = {}
  iw = []
  wi = {}
  with open(path, encoding='utf-8', errors='ignore') as f:
      first_line = True
      for line in f:
          if first_line:
              first_line = False
              dim = int(line.rstrip().split()[1])
              continue
          lines_num += 1
          tokens = line.rstrip().split(' ')
          vectors[tokens[0]] = np.asarray([float(x) for x in tokens[1:]])
          iw.append(tokens[0])
  for i, w in enumerate(iw):
      wi[w] = i
  return vectors, iw, wi, dim

def normalize(matrix):
  norm = np.sqrt(np.sum(matrix * matrix, axis=1))
  matrix = matrix / norm[:, np.newaxis]
  return matrix
  
class CustomEmbeddings(Embeddings):
  
  def __init__(self, path, **kwargs: Any):
    vectors, iw,  wi, dim = read_vectors(path)
    self.vectors = vectors
    self.iw = iw
    self.wi = wi
    self.dim = dim

  def embed_documents(self, texts: List[str]) -> List[List[float]]:
    embeddings = [self.embed_query(text) for text in texts]
    return embeddings

  def embed_query(self, text: str) -> List[float]:
    text = text.replace("\n", " ")
    embedding = [self.vectors[x] for x in text]
    return embedding

vectors, iw, wi, dim = read_vectors("../models/sgns.renmin.word")

# Turn vectors into numpy format and normalize them
matrix = np.zeros(shape=(len(iw), dim), dtype=np.float32)
for i, word in enumerate(iw):
		matrix[i, :] = vectors[word]
matrix = normalize(matrix)

#matrix[wi["上"]]
vectors['上']

# Load embedding
def load_embedding(embedding_source:str = "huggingface"):
  if embedding_source == "openai":
    return OpenAIEmbeddings()
  else:
    llama_model_path = "../models/all-mpnet-base-v2"
    embed_model = LangchainEmbedding(HuggingFaceEmbeddings(model_name=llama_model_path))
    return embed_model
  

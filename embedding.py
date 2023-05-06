from langchain.embeddings import OpenAIEmbeddings
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index import LangchainEmbedding

# Load embedding
def load_embedding(embedding_source:str = "huggingface"):
  if embedding_source == "openai":
    return OpenAIEmbeddings()
  else:
    llama_model_path = "../models/all-mpnet-base-v2"
    embed_model = LangchainEmbedding(HuggingFaceEmbeddings(model_name=llama_model_path))
    return embed_model
  

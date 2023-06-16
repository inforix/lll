# Load embedding
from langchain.embeddings import OpenAIEmbeddings
from langchain.embeddings.huggingface import HuggingFaceEmbeddings

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

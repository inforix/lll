from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain.document_loaders import DirectoryLoader
from typing import List
from langchain.docstore.document import Document

def load_documents():
  loader = DirectoryLoader("./data/", "**/*.txt")
  documents = loader.load()
  text_splitter = CharacterTextSplitter(        
   chunk_size = 1000,
   chunk_overlap  = 60,
  )
  texts = text_splitter.split_documents(documents)
  return texts
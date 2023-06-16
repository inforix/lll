import torch
from utils.embedding import load_embedding
from utils.documents import load_documents
from utils.vendors import ChromaVectorStore

from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain.document_loaders import DirectoryLoader

from langchain.chains.question_answering import load_qa_chain
from langchain.chains import RetrievalQA
from utils.model import load_chinese_vicuna_model, load_model, load_alpaca_model
from utils.mossllm import MOSSLLM
from utils.chatglm import ChatGLM
from utils.customllm import CustomVicunaLLM, CustomLLM
from langchain import HuggingFacePipeline
from langchain.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.embeddings.huggingface import HuggingFaceEmbeddings

prompt_template = (
  "Below is an instruction that describes a task. "
  "Write a response in chinese that appropriately completes the request.\n\n"
  "### Instruction:\n{context}\n{question}\n\n### Response: ")
prompt_template = """
Answer the question in chinese based on the context below. Keep the answer short and concise. Respond "Unsure about answer" if not sure about the answer.

### Context: {context}

### Question: {question}

### Answer: """

class QA:
  def __init__(self, embedding_model_path: str, model_path:str, lora_path: str, model_type:str = "alpaca", device:str = "cuda") -> None:
    self.device = device
    if device == "cuda" and not torch.cuda.is_available():
      self.device = "cpu"
    
      
    self.embedding_model_path = embedding_model_path
    self.model_path = model_path
    self.lora_path = lora_path
    self.model_type = model_type
    
    self.embedding = HuggingFaceEmbeddings(model_name=self.embedding_model_path)
    self.load_model()
    self.add_documents()

  def load_model(self):
    if self.model_type == "chatglm":
      self.llm = ChatGLM()
      self.llm.load_model()
    elif self.model_type == "moss":
      self.llm = MOSSLLM()
      self.llm.load_model()
    elif self.model_type == "vicuna":
      model, tokenizer = load_chinese_vicuna_model(local=True, device=self.device)
      self.llm = CustomVicunaLLM(model, tokenizer, self.device)
    elif self.model_type == "alpaca":
      model, tokenizer = load_model(self.model_path, self.lora_path, self.device)
      model.type
      self.llm = CustomLLM(model, tokenizer, self.device)
      # llm = HuggingFacePipeline.from_model_id(model_id = "../models/chinese-alpaca-7b",
      #                                         task="text-generation",
      #                                         model_kwargs={
      #                                                       "torch_dtype" : load_dtype,
      #                                                       "low_cpu_mem_usage" : True,
      #                                                       "temperature": 0.1,
      #                                                       "max_length": 4000,
      #                                                       "device_map": "auto",
      #                                                       "repetition_penalty":2.0}
      #                                         )
  
  def add_documents(self, path: str = "./data/", pattern: str = "**/*.txt"):
    loader = DirectoryLoader(path, pattern)
    documents = loader.load()
    text_splitter = CharacterTextSplitter(        
      chunk_size = 2000,
      chunk_overlap  = 100,
    )
    documents = text_splitter.split_documents(documents)
    #self.docsearch = Chroma.from_documents(documents=documents, embedding=self.embedding)
    self.store = ChromaVectorStore(documents, self.embedding)
    self.load_chain()
  
  def load_chain(self):
    PROMPT = PromptTemplate(
            template=prompt_template, input_variables=["context", "question"]
        )
    
    # self.qa = RetrievalQA.from_chain_type(
    #   llm = self.llm,
    #   chain_type="stuff",
    #   retriever=self.docsearch.as_retriever(search_kwargs={"k":1}),
    #   chain_type_kwargs={"prompt": PROMPT}
    # )
    self.qa = load_qa_chain(self.llm, chain_type="stuff", prompt=PROMPT)
    
  def query(self, query:str) -> str:
    if len(query.strip()) == 0:
      return ""
    
    similar_docs = self.store.get_similiar_docs(query, k=3)
    
    return self.qa.run(input_documents=similar_docs, question=query)
from typing import List, Optional, Mapping, Any
import torch
import logging

from langchain import PromptTemplate, LLMChain, OpenAI
from langchain.document_loaders import TextLoader, DirectoryLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS, Chroma
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain.embeddings import OpenAIEmbeddings
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.indexes import VectorstoreIndexCreator
from llama_index.embeddings.base import BaseEmbedding
from llama_index.readers.schema.base import Document
from llama_index import SimpleDirectoryReader, LangchainEmbedding, GPTListIndex,GPTSimpleVectorIndex, PromptHelper, LLMPredictor, ServiceContext

from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer, LlamaForCausalLM
import transformers

from peft import PeftModel
from PyPDF2 import PdfReader
from customllm import CustomVicunaLLM, CustomLLM

import os

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

os.environ['OPENAI_API_KEY'] = "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"

# disable GPU if needed
if os.environ.get('DISABLE_GPU') == "1":
  torch.cuda.is_available = lambda: False

if torch.cuda.is_available():
  device = torch.device(0)
else:
  device = torch.device('cpu')

assert (
    "LlamaTokenizer" in transformers._import_structure["models.llama"]
), "LLaMA is now in HuggingFace's main branch.\nPlease reinstall it: pip uninstall transformers && pip install git+https://github.com/huggingface/transformers.git"


def create_llama_prompt():
  # The prompt template below is taken from llama.cpp
  # and is slightly different from the one used in training.
  # But we find it gives better results
  prompt_input = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n\n{instruction}\n\n### Response:\n\n"
  )

  prompt = PromptTemplate(template=prompt_input, input_variables=["instruction"])
  
  return prompt

generation_config = dict(
  temperature=0.2,
  top_k=40,
  top_p=0.9,
  do_sample=True,
  num_beams=1,
  repetition_penalty=1.3,
  max_new_tokens=400
)

base_model = "../models/chinese-llama-7b-hf-merged"
base_model = "../models/chatglm-6b"
base_model = "../models/llama-7b-hf"
#model_path = "../models/moss-moon-003-sft-int4"
lora_model_path = "../models/chinese-alpaca-lora-7b"
#lora_model_path = "../models/Chinese-Vicuna-lora-7b-belle-and-guanaco"

def load_model(model_path: str, lora_path:str = None, device = torch.device("cpu")):
  assert model_path is not None

  load_type = torch.float16
  tokenizer_model_path = model_path
  if lora_path is not None:
    tokenizer_model_path = lora_path
  tokenizer = LlamaTokenizer.from_pretrained(tokenizer_model_path, trust_remote_code=True)
  base_model = AutoModelForCausalLM.from_pretrained(
    model_path,
    load_in_8bit=False,
    #load_in_8bit_fp32_cpu_offload=True,
    low_cpu_mem_usage = True,
    torch_dtype = load_type,
    trust_remote_code=True
  )

  model_vocab_size = base_model.get_input_embeddings().weight.size(0)
  tokenizer_vocab_size = len(tokenizer)
  print(f"Vocab of the base model: {model_vocab_size}")
  print(f"Vocab of the tokenizer: {tokenizer_vocab_size}")
  if model_vocab_size < tokenizer_vocab_size:
    print("Resize model embeddings to fit tokenizer")
    base_model.resize_token_embeddings(tokenizer_vocab_size)

  if lora_path is None:
    model = base_model
  else:
    model = PeftModel.from_pretrained(base_model, lora_path, torch_dtype = load_type)

  if device == torch.device("cpu"):
    model.float()
  model.to(device)
  model.eval()
 
  return model, tokenizer


def load_vicuna_model(device):
  llama_model_path = "../models/llama-7b-hf"
  lora_model_path = "../models/Chinese-Vicuna-lora-7b-belle-and-guanaco"
  
  # load tokenizer
  tokenizer = LlamaTokenizer.from_pretrained(llama_model_path, trust_remote_code=True)
  
  # load model
  lora_bin_path = os.path.join(lora_model_path, "adapter_model.bin")
  print(lora_bin_path)
  if not os.path.exists(lora_bin_path):
    pytorch_bin_path = os.path.join(lora_model_path, "pytorch_model.bin")
    print(pytorch_bin_path)
    if os.path.exists(pytorch_bin_path):
      os.rename(pytorch_bin_path, lora_bin_path)
    else:
      assert ('Checkpoint is not Found!')
      
  if torch.cuda.is_available():
    model = LlamaForCausalLM.from_pretrained(llama_model_path,
                                             load_in_8bit=True,
                                             torch_dtype = torch.float16,
                                             device_map={"":0},)
    model = PeftModel.from_pretrained(model, lora_model_path, torch_dtype = torch.float16, device_map={"":0})
  else:
    model = LlamaForCausalLM.from_pretrained(llama_model_path,
                                             device_map={"":device},
                                             low_cpu_mem_usage=True)
    model = PeftModel.from_pretrained(model, lora_model_path, device_map={"":device})
    #model.half()
  
  print(model.dtype)
  model.eval()

  return model, tokenizer


      
# define prompt helper
# set maximum input size
max_input_size = 1024
# set number of output tokens
num_outputs = 256
# set maximum chunk overlap
max_chunk_overlap = 20
prompt_helper = PromptHelper(max_input_size, num_outputs, max_chunk_overlap, chunk_size_limit=2000)

def load_service_meta():
  #model, tokenizer = load_model(base_model, lora_model_path, device)
  model, tokenizer = load_vicuna_model(device)
  
  llama_model_path = "../models/all-mpnet-base-v2"
  embed_model = LangchainEmbedding(HuggingFaceEmbeddings(model_name=llama_model_path))
  # define LLM
  llm_predictor = LLMPredictor(llm=CustomLLM(model, tokenizer, device))
  service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, embed_model= embed_model, prompt_helper=prompt_helper)
  return service_context

def load_service_openai():
  # define LLM
  llm_predictor = LLMPredictor(llm=OpenAI(temperature=0, model_name="gpt-3.5-turbo", max_tokens=num_outputs))
  service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, prompt_helper=prompt_helper)
  
  return service_context


def load_documents() -> List[Document]:
  loader = DirectoryLoader("../data/", "**/*.txt")
  documents = loader.load()
  text_splitter = CharacterTextSplitter(        
   separator = "\n\n\n",
   chunk_size = 8000,
   chunk_overlap  = 0,
   length_function = len,
  )
  #text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
  texts = text_splitter.split_documents(documents)
  return texts

def ask(index, question):
  print("Human: ", question)
  response = index.query(question,response_mode="default", mode="default")
  print("AI: ", response)

  
def gpt_list_index():
  documents = SimpleDirectoryReader("../data").load_data()

  service_context = load_service_meta()

  index = GPTSimpleVectorIndex.from_documents(documents, service_context=service_context)

  return index

  
def main():
  index = gpt_list_index()
  # Query and print response
  ask(index, "上海高级国际航运学院是什么时候成立的？")
  ask(index, "学校有几个博士点？")
  ask(index, "学校有多少个硕士点")
  ask(index, "学校有马克思主义学院吗？")

  while True:
    question = input("Human: ")
    print("\n")
    if question == "quit":
      break
    
    ask(index, question)

#interactivate2()

def qa():
  model, tokenizer = load_model(base_model, lora_model_path, device)
  texts = load_documents()
  #embeddings = OpenAIEmbeddings()
  embeddings = HuggingFaceEmbeddings(model_name="../models/all-mpnet-base-v2")
  db = Chroma.from_documents(texts, embeddings)
  retriever = db.as_retriever(search_type="similarity", search_kwargs={"k":1})
  qa = RetrievalQA.from_chain_type(llm=CustomLLM(model, tokenizer, device), chain_type="stuff", retriever=retriever)
  
  chat_history = []
  while True:
    query = input("Human: ")
    result = qa(query)
    print("AI: ", result["answer"])

def vectors():
  model, tokenizer = load_model(base_model, lora_model_path)
  # define LLM
  llm=CustomLLM(model, tokenizer)
  
  loader = DirectoryLoader("../data/", "**/*.txt")
  index = VectorstoreIndexCreator(
    text_splitter=CharacterTextSplitter(chunk_size=1000, chunk_overlap=0),
    #embedding=OpenAIEmbeddings(),
    vectorstore_cls=Chroma
  ).from_loaders([loader])
  while True:
    question = input("Human: ")
    print("\n")
    response = index.query(question, llm=llm)
    print("AI: ", response)
  
def load_pdf(file):
  reader = PdfReader(file)
  raw_text = ''
  for i,page in enumerate(reader.pages):
    text = page.extract_text()
    if text:
      raw_text += text
  
  text_splitter = CharacterTextSplitter(        
    separator = "\n",
    chunk_size = 1000,
    chunk_overlap  = 200,
    length_function = len,
  )
  texts = text_splitter.split_text(raw_text)
  
  embeddings = LangchainEmbedding(HuggingFaceEmbeddings(model_name = base_model))
  docsearch = FAISS.from_texts(texts, embeddings)
  return docsearch


def construct_index(directory_path):
  # set maximum input size
  max_input_size = 4096
  # set number of output tokens
  num_outputs = 256
  # set maximum chunk overlap
  max_chunk_overlap = 20
  # set chunk size limit
  chunk_size_limit = 600

  prompt_helper = PromptHelper(max_input_size, num_outputs, max_chunk_overlap, chunk_size_limit=chunk_size_limit)

  # define LLM
  llm_predictor = LLMPredictor(llm=OpenAI(temperature=0, model_name="text-davinci-002", max_tokens=num_outputs))
  
  documents = SimpleDirectoryReader(directory_path).load_data()
  
  service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, prompt_helper=prompt_helper)
  index = GPTSimpleVectorIndex.from_documents(documents, service_context=service_context)
  
  index.save_to_disk('index.json')
  
  return index

def ask_bot(input_index:str = 'index.json'):
  index = GPTSimpleVectorIndex.load_from_disk(input_index)
  while True:
    query = input('What do you want to ask the bot?   \n')
    response = index.query(query, response_mode="compact")
    print ("\nBot says: \n\n" + response.response + "\n\n\n")
    
# query ="请问如何申请VPN"
# docsearch = load_pdf(pdf_path)
# docs = docsearch.similarity_search(query)

# llm_chain.run(input_documents=docs, question=query)

# input_text = "请写一首古诗"
# inputs = tokenizer(input_text, return_tensors="pt")  #add_special_tokens=False ?
# generation_output = model.generate(
# 		input_ids = inputs["input_ids"].to(device), 
# 		attention_mask = inputs['attention_mask'].to(device),
# 		eos_token_id=tokenizer.eos_token_id,
# 		pad_token_id=tokenizer.pad_token_id,
# 		**generation_config
# )
# s = generation_output[0]
# output = tokenizer.decode(s,skip_special_tokens=True)
# response = output.split("### Response:")[1].strip()
# print("Response: ",response)
# print("\n")

if __name__ == "__main__":
  #qa()
  main()
  #vectors()
  
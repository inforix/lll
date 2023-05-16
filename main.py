from typing import List, Optional, Mapping, Any
import torch
import logging
import asyncio

from langchain import PromptTemplate, LLMChain, OpenAI
from langchain.document_loaders import TextLoader, DirectoryLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS, Chroma
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import ConversationalRetrievalChain, RetrievalQA, VectorDBQA
from langchain.embeddings import OpenAIEmbeddings
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.indexes import VectorstoreIndexCreator
from langchain.chat_models import ChatOpenAI
from llama_index.readers.schema.base import Document
from llama_index import SimpleDirectoryReader, LangchainEmbedding, GPTListIndex, PromptHelper, LLMPredictor, ServiceContext, GPTVectorStoreIndex
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.chains.conversational_retrieval.prompts import CONDENSE_QUESTION_PROMPT

from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer, LlamaForCausalLM
import transformers

from peft import PeftModel
from PyPDF2 import PdfReader
from utils.customllm import CustomVicunaLLM, CustomLLM
from utils.embedding import load_embedding
from utils.model import load_model, load_moss_moon, load_vicuna_model

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
moss_moon_model = "../models/moss-moon-003"
lora_model_path = "../models/chinese-alpaca-lora-7b"
#lora_model_path = "../models/Chinese-Vicuna-lora-7b-belle-and-guanaco"


      
# define prompt helper
# set maximum input size
max_input_size = 2000
# set number of output tokens
num_outputs = 256
# set maximum chunk overlap
max_chunk_overlap = 20
prompt_helper = PromptHelper(max_input_size, num_outputs, max_chunk_overlap, chunk_size_limit=2000)


  
def load_service_meta():
  #model, tokenizer = load_model(base_model, lora_model_path, device)
  model, tokenizer = load_vicuna_model(device)
  
  embed_model = load_embedding("huggingface")
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
   chunk_size = 1000,
   chunk_overlap  = 0,
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

  index = GPTListIndex.from_documents(documents, service_context=service_context)

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
async def async_generate(chain, question, chat_history):
  result = await chain.arun({"question": question, "chat_history": chat_history})
  return result

def run_async_chain(chain, question, chat_history):
  loop = asyncio.new_event_loop()
  asyncio.set_event_loop(loop)
  result = {}
  try:
      answer = loop.run_until_complete(async_generate(chain, question, chat_history))
  finally:
      loop.close()
  result["answer"] = answer
  return result
  
def qa(chat_mode = False, vectordb: str = "chroma", embedding_source: str = "huggingface"):
  # load prompt
  with open("prompts/question_prompt.txt", "r") as f:
    template_quest = f.read()
  with open("prompts/chat_reduce_prompt.txt", "r") as f:
    chat_reduce_template = f.read()
  with open("prompts/combine_prompt.txt", "r") as f:
    template = f.read()
  with open("prompts/chat_combine_prompt.txt", "r") as f:
    chat_combine_template = f.read()
    
  c_prompt = PromptTemplate(input_variables=["summaries", "question"], template=template,
                            template_format="jinja2")

  q_prompt = PromptTemplate(input_variables=["context", "question"], template=template_quest, template_format="jinja2")
    
  texts = load_documents()
  
  embedding = load_embedding(embedding_source)
    
  if vectordb == "faiss":
    docsearch = FAISS.from_documents(texts, embedding)
    retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k":2})
  elif vectordb == "chroma":
    docsearch = Chroma.from_documents(texts, embedding)
    retriever = docsearch.as_retriever()
  
  if chat_mode:
    llm = ChatOpenAI()
    messages_combine = [
        SystemMessagePromptTemplate.from_template(chat_combine_template),
        HumanMessagePromptTemplate.from_template("{question}")
    ]
    p_chat_combine = ChatPromptTemplate.from_messages(messages_combine)
    messages_reduce = [
        SystemMessagePromptTemplate.from_template(chat_reduce_template),
        HumanMessagePromptTemplate.from_template("{question}")
    ]
    p_chat_reduce = ChatPromptTemplate.from_messages(messages_reduce)
    
    question_generator = LLMChain(llm=llm, prompt=CONDENSE_QUESTION_PROMPT)
    doc_chain = load_qa_chain(llm, chain_type="map_reduce", combine_prompt=p_chat_combine)
    chain = ConversationalRetrievalChain(
        retriever=docsearch.as_retriever(k=2),
        question_generator=question_generator,
        combine_docs_chain=doc_chain,
    )
    chat_history = []
    #chain = RetrievalQA.from_chain_type(llm=llm, chain_type="map_reduce", retriever=retriever, return_source_documents=True)
    result = run_async_chain(chain, "学校有几个博士点？", chat_history)
    print("AI: ", result["answer"])
    result = run_async_chain(chain, "上海高级国际航运学院是什么时候成立的？", chat_history)
    print("AI: ", result["answer"])
    result = run_async_chain(chain, "学校有多少个硕士点", chat_history)
    print("AI: ", result["answer"])
    result = run_async_chain(chain, "学校有马克思主义学院吗？", chat_history)
    print("AI: ", result["answer"])
    while True:
      query = input("Human: ")
      if query == "quit":
        break
      #result = ask_chain(chain, query)
      result = run_async_chain(chain, query, chat_history)
      print("AI: ", result["answer"])
  else:
    model, tokenizer = load_model(base_model, lora_model_path, device)
    llm = CustomLLM(model, tokenizer, device)
    qa_chain = load_qa_chain(llm=llm, chain_type="map_reduce", combine_prompt=c_prompt, question_prompt=q_prompt)
    chain = RetrievalQA(combine_documents_chain=qa_chain, retriever=retriever, return_source_documents=True)
    result = ask_chain(chain, "上海海事大学有几个博士点？")
    print("AI: ", result["answer"])
    result = ask_chain(chain, "上海高级国际航运学院是什么时候成立的？")
    print("AI: ", result["answer"])
    result = ask_chain(chain, "上海海事大学有多少个硕士点")
    print("AI: ", result["answer"])
    result = ask_chain(chain, "上海海事大学有马克思主义学院吗？")
    print("AI: ", result["answer"])
    while True:
      query = input("Human: ")
      if query == "quit":
        break
      #result = ask_chain(chain, query)
      result = ask_chain(chain, query)
      print("AI: ", result["answer"])
  

def ask_chain(chain, query):
  result = chain({"query": query})
  if "result" in result:
    result["answer"] = result["result"]
  
  result["answer"] = result["answer"].replace("\\n", "\n")
  try:
    result["answer"] = result["answer"].split("SOURCES:")[0]
  except:
    pass
  
  return result

def vectors():
  model, tokenizer = load_model(base_model, lora_model_path)
  # define LLM
  llm=CustomLLM(model, tokenizer, device)
  
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
  index = GPTVectorStoreIndex.from_documents(documents, service_context=service_context)
  
  index.save_to_disk('index.json')
  
  return index

def ask_bot(input_index:str = 'index.json'):
  index = GPTVectorStoreIndex.load_from_disk(input_index)
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
  qa()
  #main()
  #vectors()
  
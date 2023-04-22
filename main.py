from typing import List, Optional, Mapping, Any
import torch

from langchain.llms import HuggingFacePipeline
from langchain.llms.base import LLM
from langchain import PromptTemplate, LLMChain, OpenAI
from langchain.document_loaders import TextLoader, DirectoryLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS, Chroma
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.embeddings import OpenAIEmbeddings
from langchain.indexes import VectorstoreIndexCreator


from llama_index import SimpleDirectoryReader, LangchainEmbedding, GPTListIndex,GPTSimpleVectorIndex, PromptHelper
from llama_index import LLMPredictor, ServiceContext

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, AutoModelForSeq2SeqLM, Pipeline
from transformers import LlamaTokenizer, LlamaForCausalLM

from peft import PeftModel
from PyPDF2 import PdfReader

import os

os.environ['OPENAI_API_KEY'] = "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"

torch.cuda.is_available = lambda: False
if torch.cuda.is_available():
  device = torch.device(0)
else:
  device = torch.device('cpu')


# The prompt template below is taken from llama.cpp
 # and is slightly different from the one used in training.
 # But we find it gives better results
prompt_input = (
  "Below is an instruction that describes a task. "
  "Write a response that appropriately completes the request.\n\n"
  "### Instruction:\n\n{instruction}\n\n### Response:\n\n"
)

prompt = PromptTemplate(template=prompt_input, input_variables=["instruction"])

sample_data = ["为什么要减少污染，保护环境？"]

generation_config = dict(
  temperature=0.2,
  top_k=40,
  top_p=0.9,
  do_sample=True,
  num_beams=1,
  repetition_penalty=1.3,
  max_new_tokens=400
)

def generate_prompt(instruction, input=None):
  if input:
    instruction = instruction + '\n' + input
  return prompt_input.format_map({'instruction': instruction})
  
  
#model_path = "../models/llama-7b-hf"
model_path = "../models/chinese-llama-7b-hf-merged"
#model_path = "../models/moss-moon-003-base"
lora_model_path = "../models/chinese-alpaca-lora-7b"

def load_model(model_path: str, lora_path:str = None):
  assert model_path is not None

  load_type = torch.float16
  tokenizer_model_path = model_path
  if lora_path is not None:
    tokenizer_model_path = lora_path
  tokenizer = AutoTokenizer.from_pretrained(tokenizer_model_path, trust_remote_code=True)
  base_model = AutoModelForCausalLM.from_pretrained(
    model_path,
    load_in_8bit=False,
    #load_in_8bit_fp32_cpu_offload=True,
    device_map = "auto",
    low_cpu_mem_usage = True,
    torch_dtype = load_type,
    trust_remote_code=True
  )

  model_vocab_size = base_model.get_input_embeddings().weight.size(0)
  tokenizer_vocab_size = len(tokenizer)
  print(f"Vocab of the base model: {model_vocab_size}")
  print(f"Vocab of the tokenizer: {tokenizer_vocab_size}")
  if model_vocab_size <= tokenizer_vocab_size:
    print("Resize model embeddings to fit tokenizer")
    base_model.resize_token_embeddings(tokenizer_vocab_size)

  if lora_path is None:
    model = base_model
  else:
    model = PeftModel.from_pretrained(base_model, lora_model_path, torch_dtype = load_type)

  model.float()
  model.to(device)
  model.eval()
 
  return model, tokenizer





class CustomLLM(LLM):
  pipeline:Pipeline = None
  def __init__(self, model, tokenizer, device: torch.device = "cpu"):
    super().__init__()
    self.pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer, device=device, )

  def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
    prompt_length = len(prompt)
    response = self.pipeline(prompt, max_new_tokens=num_outputs)[0]["generated_text"]

    # only return newly generated tokens
    return response[prompt_length:]

  @property
  def _identifying_params(self) -> Mapping[str, Any]:
      return {"name_of_model": self.model_name}

  @property
  def _llm_type(self) -> str:
      return "custom"
      
# define prompt helper
# set maximum input size
max_input_size = 4096
# set number of output tokens
num_outputs = 1000
# set maximum chunk overlap
max_chunk_overlap = 20
prompt_helper = PromptHelper(max_input_size, num_outputs, max_chunk_overlap, chunk_size_limit=2000)

def load_service_meta():
  model, tokenizer = load_model(model_path, None)
  # define LLM
  llm_predictor = LLMPredictor(llm=CustomLLM(model, tokenizer))
  service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, prompt_helper=prompt_helper)
  return service_context

def load_service_openai():
  # define LLM
  llm_predictor = LLMPredictor(llm=OpenAI(temperature=0, model_name="gpt-3.5-turbo", max_tokens=num_outputs))
  service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, prompt_helper=prompt_helper)
  
  return service_context



def interactivate():
  model, tokenizer = load_model(model_path, None)
  
  pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_length = 2000
  )
  local_llm = HuggingFacePipeline(pipeline=pipe)
  llm_chain = LLMChain(prompt=prompt, llm=local_llm)
  while True:
    question = input("Human: ")
    print("\n")
    print(llm_chain.run(question))

def load_documents():
  loader = DirectoryLoader("../data/", "**/*.txt")
  documents = loader.load()
  text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
  texts = text_splitter.split_documents(documents)
  return texts

def ask(index, question):
  response = index.query(question, response_mode="compact")
  print("AI: ", response)
  
def local_llm():
  documents = SimpleDirectoryReader("../data").load_data()

  service_context = load_service_meta()

  index = GPTListIndex.from_documents(documents, service_context=service_context)

  # Query and print response
  question = "上海高级国际航运学院是什么时候成立的？"
  ask(index, question)

  while True:
    question = input("Human: ")
    print("\n")
    ask(index, question)

#interactivate2()

def qa():
  texts = load_documents()
  embeddings = OpenAIEmbeddings()
  db = Chroma.from_documents(texts, embeddings)
  retriever = db.as_retriever(search_type="similarity", search_kwargs={"k":1})
  qa = ConversationalRetrievalChain.from_llm(OpenAI(), retriever)
  
  chat_history = []
  while True:
    query = input("Human: ")
    result = qa({"question": query, "chat_history": chat_history})
    print("AI: ", result["answer"])

def vectors():
  model, tokenizer = load_model(model_path, None)
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
    response = index.query(question, llm=llm, response_mode="compact")
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
  
  embeddings = LangchainEmbedding(HuggingFaceEmbeddings(model_name = model_path))
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
  
  service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor)
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
  local_llm()
  #vectors()
  
import torch
import logging

from langchain.chains.question_answering import load_qa_chain
from langchain import PromptTemplate
from langchain.docstore.document import Document
from langchain.llms import OpenAI

from langchain_wenxin import Wenxin

from utils.chatglm import ChatGLM
from utils.mossllm import MOSSLLM
from utils.customllm import CustomLLM, CustomHFLLM
from utils.cpmbeellm import CPMBeeLLM
from utils.baichuan import BaichuanChat
from langchain.llms import Tongyi


def load_llm(model_name:str = "alpaca", device:str="cuda"):
  
  if model_name == "chatglm":
    llm = ChatGLM("../models/chatglm2-6b", device=device)
  elif model_name == "moss":
    llm = MOSSLLM("../models/moss-moon-003-sft", device=device)
  elif model_name == "openai":
    llm = OpenAI()
  elif model_name == "baichuan":
    llm = BaichuanChat("../models/Baichuan-13B-Chat", device=device)
  elif model_name == "cpmbee":
    llm = CPMBeeLLM("../models/cpm-bee-10b", device=device)
  elif model_name == "xgen":
    llm = CustomLLM(device = device, model_name=model_name, eos_token_id=50256)
  elif model_name == "wenxin":
    llm = Wenxin(model="ernie-bot")
  elif model_name == "tongyi":
    llm = Tongyi()
  else:  
    llm = CustomLLM(device = device, model_name=model_name)

  return llm
import torch
from transformers import Pipeline, pipeline, GenerationConfig, AutoModel, AutoTokenizer
from langchain.llms.base import LLM
from langchain.llms.utils import enforce_stop_tokens
from typing import List, Optional, Mapping, Any
import sys
import logging

class ChatGLM(LLM):
  max_token: int = 10000
  temperature: float = 0.1
  top_p = .9
  history = []
  tokenizer: object = None
  model: object = None
  
  def __init__(self):
    super().__init__()
    
  @property
  def _llm_type(self) -> str:
    return "ChatGLM"
  
  def _call(self, prompt:str, stop: Optional[List[str]] = None) -> str:
    response, _ = self.model.chat(
      self.tokenizer,
      prompt,
      history = self.history,
      max_length = self.max_token,
      temperature = self.temperature
    )
    
    if stop is not None:
      response = enforce_stop_tokens(response, stop)
    
    self.history = self.history + [[None, response]]
    
    return response

  def load_model(self, model_name_or_path:str = "../models/chatglm-6b"):
    self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)

    if torch.cuda.is_available():
      self.model = AutoModel.from_pretrained(model_name_or_path, trust_remote_code=True).quantize(4).half().cuda()
    else:
      self.model = AutoModel.from_pretrained(model_name_or_path, trust_remote_code=True).float()
      
    self.model = self.model.eval()

if __name__ == "__main__":
  torch.cuda.is_available = lambda: False
  
  # from model import load_vicuna_model
  # device = "cpu"
  
  # model, tokenizer = load_vicuna_model(device)
  # llm = CustomVicunaLLM(model, tokenizer, device)
  # question:str = "请问什么时候放假"
  # ret = llm.generate(question)
  
  chatglm = ChatGLM()
  chatglm.load_model()
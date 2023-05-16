import torch
from transformers import Pipeline, pipeline, GenerationConfig, AutoModel, AutoTokenizer
from langchain.llms.base import LLM
from langchain.llms.utils import enforce_stop_tokens
from typing import List, Optional, Mapping, Any
import sys
import logging

class CustomLLM(LLM):
  pipeline:Pipeline = None
  max_new_tokens:int = 256
  logger:Any 
  
  def __init__(self, model, tokenizer, device: torch.device = torch.device("cpu"), max_new_tokens:int = 1000 ):
    super().__init__()
    self.logger = logging.getLogger(self.__class__.__name__)
    self.logger.setLevel(logging.DEBUG)
    self.pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer, device=device, trust_remote_code=True, torch_dtype = torch.float16, repetition_penalty=1.0)
    self.max_new_tokens = max_new_tokens

  def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
    prompt_length = len(prompt)
    response = self.pipeline(prompt, max_new_tokens=self.max_new_tokens)
    self.logger.debug("=================== RESPONSE =================== \n")
    self.logger.debug(response)
    generated = response[0]["generated_text"]
    self.logger.debug(f"prompt length:\t{prompt_length}")
    self.logger.debug(generated[prompt_length:])
    # only return newly generated tokens
    return generated[prompt_length:]

  @property
  def _identifying_params(self) -> Mapping[str, Any]:
      return {"name_of_model": self.model_name}

  @property
  def _llm_type(self) -> str:
      return "custom"
    

class CustomVicunaLLM(LLM):
  pipeline:Pipeline = None
  generation_config: GenerationConfig = None
  model: torch.nn.Module = None
  tokenizer: torch.nn.Module = None
  repetition_penalty: float = 1.0
  use_typewriter: bool = True
  device: torch.device = torch.device("cpu")
  
  def __init__(self, model, tokenizer, device: torch.device = torch.device("cpu"), temperature=0.1, top_p=0.75, top_k=40, num_beams=4, max_new_tokens=128, min_new_tokens=1, repetition_penalty=2.0, use_typewriter=True, **kwargs ):
    super().__init__()
    #self.pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer, device=device, trust_remote_code=True, torch_dtype = torch.float16)
    self.generation_config = GenerationConfig(
      temperature=temperature,
      top_p=top_p,
      top_k=top_k,
      num_beams=num_beams,
      eos_token_id=tokenizer.eos_token_id,
      pad_token_id=tokenizer.pad_token_id,
      max_new_tokens=max_new_tokens, # max_length=max_new_tokens+input_sequence
      min_new_tokens=min_new_tokens, # min_length=min_new_tokens+input_sequence
      **kwargs,
    )

    self.model = torch.compile(model) if torch.__version__ >= "2" and sys.platform != "win32" else model
    
    self.tokenizer = tokenizer
    self.repetition_penalty = repetition_penalty
    self.use_typewriter = use_typewriter
    self.device = device

  def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
    inputs = self.tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(self.device)
    
    with torch.no_grad():
      if self.use_typewriter:
        for generation_output in self.model.stream_generate(
          input_ids=input_ids,
          generation_config=self.generation_config,
          return_dict_in_generate=True,
          output_scores=False,
          repetition_penalty=float(self.repetition_penalty),
        ):
          outputs = self.tokenizer.batch_decode(generation_output)
          show_text = "\n--------------------------------------------\n".join(
              [output.split("### Response:")[1].strip().replace('�','')+" ▌" for output in outputs]
          )
          # if show_text== '':
          #     yield last_show_text
          # else:
          yield show_text
        yield outputs[0].split("### Response:")[1].strip().replace('�','')
      else:
        generation_output = self.model.generate(
          input_ids=input_ids,
          generation_config=self.generation_config,
          return_dict_in_generate=True,
          output_scores=False,
          repetition_penalty=1.3,
        )
        output = generation_output.sequences[0]
        output = self.tokenizer.decode(output).split("### Response:")[1].strip()
        print(output)
        yield output
    

  @property
  def _identifying_params(self) -> Mapping[str, Any]:
      return {"name_of_model": self.model_name}

  @property
  def _llm_type(self) -> str:
      return "custom"

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
  
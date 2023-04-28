import torch
from transformers import Pipeline, pipeline, GenerationConfig
from langchain.llms.base import LLM
from typing import List, Optional, Mapping, Any
import sys

class CustomLLM(LLM):
  pipeline:Pipeline = None
  max_new_tokens:int = 1000
  
  def __init__(self, model, tokenizer, device: torch.device = torch.device("cpu"), max_new_tokens:int = 1000 ):
    super().__init__()
    self.pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer, device=device, trust_remote_code=True, torch_dtype = torch.float16)
    self.max_new_tokens = max_new_tokens

  def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
    prompt_length = len(prompt)
    response = self.pipeline(prompt, max_new_tokens=self.max_new_tokens)
    print("=================== RESPONSE =================== \n")
    #print(response)
    generated = response[0]["generated_text"]
    print("prompt length:\t", prompt_length)
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
  toknizer: torch.nn.Module = None
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
    
    self.toknizer = tokenizer
    self.repetition_penalty = repetition_penalty
    self.use_typewriter = use_typewriter
    self.device = device

  def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
    inputs = self.toknizer(prompt, return_tensors="pt")
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
      
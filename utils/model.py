import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer, LlamaForCausalLM
from peft import PeftModel
import os
import sys
import warnings

def load_model(model_path: str, lora_path:str = None, device = "cpu"):
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

  if device == "cpu":
    model.float()
  model.to(device)
  model.eval()
 
  return model, tokenizer

def load_alpaca_plus_model(device, local:bool = True):
  model_path = "../models/chinese-alpaca-plus-7b"

  tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
  model = AutoModelForCausalLM.from_pretrained(model_path,low_cpu_mem_usage=True,
                                              device_map="auto",
                                              trust_remote_code=True)
  model.to(device)
  if device == "cuda":
    model.half()
  
  return model, tokenizer
  
def load_alpaca_model(device, local:bool = True):
  base_model = "../models/llama-7b-hf" if local else "decapoda-research/llama-7b-hf"
  lora_model_path = "../models/chinese-alpaca-lora-7b" if local else "ziqingyang/chinese-alpaca-lora-7b"
  # base_model = "../models/llama-30b-hf"
  # lora_model_path = "../models/chinese-alpaca-lora-33b"
  # base_model = "../models/chinese-alpaca-7b-merged"
  # lora_model_path = None
  
  model, tokenizer = load_model(base_model, lora_model_path, device)
  return model, tokenizer

def load_alpaca2_model(device, local:bool = True):
  base_model = "../models/chinese-alpaca-2-13b-hf" if local else "ziqingyang/chinese-alpaca-2-7b"
  lora_model_path = None
  
  model, tokenizer = load_model(base_model, lora_model_path, device)
  return model, tokenizer

def load_wizard_model(device, local:bool = True):
  base_model = "../models/wizardLM-7B-HF" if local else "microsoft/wizardlm-base"
  
  model, tokenizer = load_model(base_model, None, device)
  
  return model, tokenizer


def load_vicuna_model(device, local:bool = True):
  model_path = "../models/vicuna-7b" if local else "decapoda-research/llama-7b-hf"
  lora_model_path = None # "../models/vicuna-7b-delta-v1.1" if local else "lmsys/vicuna-7b-delta-v1.1"
  
  if device == "cpu":
    kwargs = {"torch_dtype": torch.float32}
  else:
    kwargs = {"torch_dtype": torch.float16}
    
  tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
  model = AutoModelForCausalLM.from_pretrained(model_path,low_cpu_mem_usage=True, **kwargs)
  
  if device == "cuda":
    model.to(device)
  
  return model, tokenizer

def load_chinese_vicuna_model(device:str = "cpu", local:bool = True):
  llama_model_path = "../models/llama-7b-hf" if local else "decapoda-research/llama-7b-hf"
  lora_model_path = "../models/Chinese-Vicuna-lora-7b-3epoch-belle-and-guanaco" if local else "Facico/Chinese-Vicuna-lora-7b-3epoch-belle-and-guanaco"
  
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
      warnings.warn(
            "The file name of the lora checkpoint'pytorch_model.bin' is replaced with 'adapter_model.bin'"
        )
    else:
      assert ('Checkpoint is not Found!')
      
  if torch.cuda.is_available():
    model = LlamaForCausalLM.from_pretrained(llama_model_path,
                                             #load_in_8bit=True,
                                             torch_dtype = torch.float16,
                                             device_map="auto",
                                             )
    model = PeftModel.from_pretrained(model, lora_model_path, torch_dtype = torch.float16, device_map="auto")
  else:
    model = LlamaForCausalLM.from_pretrained(llama_model_path,
                                             device_map={"": device},
                                             low_cpu_mem_usage=True)
    model = PeftModel.from_pretrained(model, lora_model_path, device_map={"":device})
  # if not LOAD_8BIT:
    #model.half()
  
  print(model.dtype)
  model.eval()
  if torch.__version__ >= "2" and sys.platform != "win32":
    model = torch.compile(model)

  return model, tokenizer

def load_moss_moon(local:bool=True):
  model_name_or_path = "../models/moss-moon-003-sft" if local else "fnlp/moss-moon-003-sft"
  tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
  if torch.cuda.is_available():
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, trust_remote_code=True).half().cuda()
  else:
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, trust_remote_code=True).float()

  return model, tokenizer

if __name__ == "__main__":
  torch.cuda.is_available = lambda: False
  load_moss_moon()
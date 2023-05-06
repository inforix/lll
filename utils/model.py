import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer, LlamaForCausalLM
from peft import PeftModel
import os

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

def load_moss_moon():
  tokenizer = AutoTokenizer.from_pretrained("../models/moss-moon-003-sft-int4", trust_remote_code=True)
  model = AutoModelForCausalLM.from_pretrained("../models/moss-moon-003-sft-int4", trust_remote_code=True).half().cuda()

  return model, tokenizer

if __name__ == "__main__":
  load_moss_moon()
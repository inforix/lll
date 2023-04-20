from langchain.llms import HuggingFacePipeline
from langchain import PromptTemplate, LLMChain
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, AutoModelForSeq2SeqLM
from transformers import LlamaTokenizer, LlamaForCausalLM
from peft import PeftModel

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
model_path = "../models/chinese-llama-13b-hf-merged"
lora_model_path = "../models/chinese-alpaca-lora-7b"

def load_model(model_path: str, lora_path:str = None):
  assert model_path is not None

  load_type = torch.float16
  tokenizer_model_path = model_path
  if lora_path is not None:
    tokenizer_model_path = lora_path
  tokenizer = LlamaTokenizer.from_pretrained(tokenizer_model_path)
  base_model = LlamaForCausalLM.from_pretrained(
    model_path,
    load_in_8bit=False,
    #load_in_8bit_fp32_cpu_offload=True,
    device_map = "auto",
    low_cpu_mem_usage = True,
    torch_dtype = load_type,
  )

  model_vocab_size = base_model.get_input_embeddings().weight.size(0)
  tokenizer_vocab_size = len(tokenizer)
  print(f"Vocab of the base model: {model_vocab_size}")
  print(f"Vocab of the tokenizer: {tokenizer_vocab_size}")
  if model_vocab_size != tokenizer_vocab_size:
    assert tokenizer_vocab_size > model_vocab_size
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

model, tokenizer = load_model(model_path, None)

pipe = pipeline(
  "text-generation",
  model=model,
  tokenizer=tokenizer,
  max_length = 2000
)

local_llm = HuggingFacePipeline(pipeline=pipe)
llm_chain = LLMChain(prompt=prompt, llm=local_llm)
question = "请评价一下“青山绿水就是金山银山”"
print(llm_chain.run(question))


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
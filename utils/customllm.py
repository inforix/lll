import torch
from transformers import Pipeline, pipeline, GenerationConfig, AutoModel, AutoTokenizer, LlamaTokenizer, LlamaForCausalLM
from langchain.llms.base import LLM
from langchain.llms.utils import enforce_stop_tokens
from typing import List, Optional, Mapping, Any
import sys
import logging

class CustomLLM(LLM):
  model_name:str = "CustomLLM"
  pipeline:Pipeline = None
  max_new_tokens:int = 1000
  logger:Any 
  
  def __init__(self, model, tokenizer, device: torch.device = torch.device("cpu"), max_new_tokens:int = 1000, model_name:str = "CustomLLM" ):
    super().__init__()
    self.logger = logging.getLogger(self.__class__.__name__)
    self.logger.setLevel(logging.DEBUG)
    if device == "cpu":
      model_kwargs = {"torch_dtype": torch.float32}
    else:
      model_kwargs = {"torch_dtype": torch.float16}
    self.pipeline = pipeline("text-generation", 
                             model=model, 
                             tokenizer=tokenizer, 
                             device=device, 
                             trust_remote_code=True, 
                             repetition_penalty=2.0,
                             model_kwargs=model_kwargs
                    )
    self.max_new_tokens = max_new_tokens
    self.model_name = model_name

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
  generation_config: GenerationConfig = None
  model: LlamaForCausalLM = None
  tokenizer: LlamaTokenizer = None
  repetition_penalty: float = 1.0
  use_typewriter: bool = True
  device: str = "cpu"
  
  def __init__(self, model, tokenizer, device: str = "cpu", temperature=0.1, top_p=0.75, top_k=40, num_beams=4, max_new_tokens=1024, min_new_tokens=1, repetition_penalty=2.0, use_typewriter=False, **kwargs ):
    super().__init__()
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
    if device == "cuda":
      input_ids = inputs["input_ids"].to(self.device)
    else:
      input_ids = inputs["input_ids"]
    
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
          repetition_penalty=self.repetition_penalty,
        )
        # output = generation_output.sequences[0]
        # output = self.tokenizer.decode(output).split("### Response:")[1].strip()
        output = generation_output[0][len(input_ids[0]):]
        output = self.tokenizer.decode(output, skip_special_tokens=True).strip()
        print(output)
        return(output)
    

  @property
  def _identifying_params(self) -> Mapping[str, Any]:
      return {"name_of_model": "vicuna"}

  @property
  def _llm_type(self) -> str:
      return "vicuna"


if __name__ == "__main__":
  torch.cuda.is_available = lambda: False
  
  from model import load_vicuna_model, load_chinese_vicuna_model
  from langchain.chains.question_answering import load_qa_chain

  device = "cpu"
  
  model, tokenizer = load_chinese_vicuna_model(device)
  llm = CustomVicunaLLM(model, tokenizer, device)
  question:str = """根据下文请回答问题，如果没有答案，直接返回“不知道”。```实行校院二级管理体制，现设有商船学院、交通运输学院、经济管理学院（上海高级国际航运学院、亚洲邮轮学院）、物流工程学院（中荷机电工程学院）、法学院、信息工程学院、外国语学院、海洋科学与工程学院、理学院、徐悲鸿艺术学院、马克思主义学院、物流科学与工程研究院、体育教学部、国际教育学院、继续教育学院、上海港湾学校等二级办学部门。在27000余名学生中，有全日制本科生近16000人，各类在校研究生近8000人，留学生900余人。在近1300名专任教师中，有教授近190名。学校致力于培养国家航运业所需要的各级各类专门人才，已向全国港航企事业单位及政府部门输送了逾19万毕业生，被誉为“高级航运人才的摇篮”。\n\n学校2013年成立中国（上海）自贸区供应链研究院和上海高级国际航运学院。中国（上海）自贸区供应链研究院将自贸区建设与供应链研究有机结合，以提升自贸区产业链建设水平，促进自贸区货物贸易向服务贸易的转型发展，同时推动政府监管职能的转变。上海高级国际航运学院采取国际上先进的商学院运作模式，与全球优秀教育机构资源共享，着力打造国内领先、国际知名的航运金融教育品牌，构筑具有影响力的航运高端人才输出基地。\n\n2008年，上海市教育委员会、上海市城乡建设和交通委员会、上海海事大学、虹口区人民政府等20多家单位共同发起成立上海国际航运研究中心。中心挂靠上海海事大学，是国际航运业发展的研究和咨询机构，为政府和国内外企业与航运机构等提供决策咨询和信息服务，是上海市教委首批建立的“高校知识服务平台”之一。 2014年，市教委将该平台挂牌为“上海市协同创新中心”。\n\n学校与境外100余所姐妹院校建立了校际交流与合作关系，开展教师交流、合作办学、合作科研、学生交换等。与联合国国际海事组织、波罗的海国际航运公会、挪威船级社等国际知名航运组织/机构建立了密切联系。自2010年起开设“国际班”，邀请美国、韩国、波兰、俄罗斯、德国等国家航海院校的学生来校学习“航海技术”“航运管理”等专业。2011年，经教育部批准，学校与加纳中西非地区海事大学合作举办“物流管理”本科教育项目，并开始在非洲招生，这是上海市地方高校第一个颁发中国高校本科文凭的海外办学项目。2012年，学校获教育部批准正式成为“接受中国政府奖学金来华留学生院校”。```问题：```上海海事大学有几个博士点？```"""
  inputs = tokenizer(question, return_tensor="pt")
  outputs = model.generate(input_ids=inputs["input_ids"])
  outputs = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
  print(outputs)

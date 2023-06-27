import torch
from utils.chatglm import ChatGLM
from utils.mossllm import MOSSLLM
from langchain.chains.question_answering import load_qa_chain
from langchain import PromptTemplate
from langchain.docstore.document import Document
from utils.customllm import CustomLLM, CustomHFLLM


class QuestionAnswerChain:
  _template_quest = """已知信息：
{context} 
根据上述已知信息，简洁和专业的来回答用户的问题。如果无法从中得到答案，请说 “根据已知信息无法回答该问题” 或 “没有提供足够的相关信息”，不允许在答案中添加编造成分，答案请使用中文。 问题是：{question}"""

  def __init__(self, model_name:str = "alpaca", device:str="cuda", question_template:PromptTemplate = None) -> None:
    self.model_name = model_name
    self.device = "cuda" if torch.cuda.is_available() else "cpu" if device is None or device == "cuda" else device
    self.question_template = self._template_quest if question_template is None else question_template

    self.load_llm()
  
  def load_llm(self):
  
    if self.model_name == "chatglm":
      llm = ChatGLM("../models/chatglm2-6b", device=self.device)
    elif self.model_name == "moss":
      llm = MOSSLLM("../models/moss-moon-003-sft", device=self.device)
    else:  
      llm = CustomLLM(device = self.device, model_name=self.model_name)

    print("llm loaded.")
    q_prompt = PromptTemplate(input_variables=["context", "question"], template=self.question_template)

    self.chain = load_qa_chain(llm, chain_type="stuff", prompt=q_prompt)
  
  def question_over_document(self, text:str, question:str) -> str:
    doc = Document(page_content=text, metadata={})
    docs = [doc]

    answer = self.chain.run(input_documents=docs, question=question)
    return answer
  
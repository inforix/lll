import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils.customllm import CustomLLM, CustomHFLLM
from utils.model import load_alpaca_model, load_chinese_vicuna_model, load_moss_moon, load_wizard_model
from utils.chatglm import ChatGLM
from utils.mossllm import MOSSLLM
from utils.QuestionAnswerChain import QuestionAnswerChain
from langchain.chains.question_answering import load_qa_chain
from langchain import PromptTemplate
from langchain.docstore.document import Document
from argparse import ArgumentParser
from questions import Texts, Questions
import logging

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

torch.cuda.is_available = lambda: False

device = "cuda" if torch.cuda.is_available() else "cpu"
    
parser = ArgumentParser()
parser.add_argument("-m", "--model", type=str, default="alpaca", help="Model to use")
args = parser.parse_args()



template_quest = """Below is a context and a question, please answer the question according by the context.if there is no answer in context, please answer 'No' directly.
Context:
```
{context}
```
Question:
```
{question}
```
### Response:"""

template_quest = """已知信息：
{context} 
根据上述已知信息，简洁和专业的来回答用户的问题。如果无法从中得到答案，请说 “根据已知信息无法回答该问题” 或 “没有提供足够的相关信息”，不允许在答案中添加编造成分，答案请使用中文。 问题是：{question}"""

# template_question = """根据接下来的文本内容回答后面的问题，如果不能从文本内容中找到答案就直接返回“不知道”，不要随意捏造文字。
# ### 文本内容:
# {context}
# ### 问题:
# {question}
# ### Response:
# """

# model, tokenizer = load_wizard_model(device) #
# model, tokenizer = load_alpaca_model(device)
# #model, tokenizer = load_chinese_vicuna_model(device)

#model.eval()
#print("model loaded!")

# if args.model == "chatglm":
#   llm = ChatGLM("../models/chatglm2-6b")
# elif args.model == "moss":
#   llm = MOSSLLM("../models/moss-moon-003-sft")
# else:  
#   llm = CustomHFLLM(device = device, model_name=args.model)

# q_prompt = PromptTemplate(input_variables=["context", "question"], template=template_quest)

# chain = load_qa_chain(llm, chain_type="stuff", prompt=q_prompt)


# def qa(context, question):
#   doc = Document(page_content=context, metadata={})
#   docs = [doc]

#   answer = chain.run(input_documents=docs, question=question)
#   print(answer)
  
chain = QuestionAnswerChain(args.model, device)
logger.info("chain loaded.")

for i in range(len(Texts)):
  outputs = chain.question_over_document(Texts[i], Questions[i])
  print(Questions[i], "\n", outputs, "\n\n")

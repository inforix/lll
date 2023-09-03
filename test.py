import os
import torch
import logging
from argparse import ArgumentParser
from dotenv import load_dotenv

from utils.QuestionAnswerChain import QuestionAnswerChain
from qadoc import QA

from questions import Texts, Questions

load_dotenv()

template_quest1 = """Below is a context and a question, please answer the question according by the context.if there is no answer in context, please answer 'No' directly.
Context:
```
{context}
```
Question:
```
{question}
```
### Response:"""

template_quest2 = """已知信息：
{context} 
根据上述已知信息，简洁和专业的来回答用户的问题。如果无法从中得到答案，请说 “根据已知信息无法回答该问题” 或 “没有提供足够的相关信息”，不允许在答案中添加编造成分，答案请使用中文。 问题是：{question}"""

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

torch.cuda.is_available = lambda: False

device = "cuda" if torch.cuda.is_available() else "cpu"
    
parser = ArgumentParser()
parser.add_argument("-m", "--model", type=str, default="alpaca", help="Model to use")
parser.add_argument("-t","--template", type=str, default=template_quest1, help="Template to use")
args = parser.parse_args()

# template_question = """根据接下来的文本内容回答后面的问题，如果不能从文本内容中找到答案就直接返回“不知道”，不要随意捏造文字。
# ### 文本内容:
# {context}
# ### 问题:
# {question}
# ### Response:
# """

#chain = QuestionAnswerChain(args.model, device, question_template=template_quest1)
#logger.info("chain loaded.")

qa = QA(embedding_source="dashscope", embedding_model_path="../models/m3e-large", model_type="wenxin", device=device)
for question in Questions:
  print(question)
  print(qa.query(question))
  print("\n")
  
# for i in range(len(Texts)):
#   outputs = chain.question_over_document(Texts[i], Questions[i])
#   print(Questions[i], "\n", outputs, "\n\n")

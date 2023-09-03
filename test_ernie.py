"""
This file is used to generate prompt
"""
import requests
import json
import sys
import os
from questions import Questions, Texts
from providers.baidu import BaiduErnie
from dotenv import load_dotenv

template_quest = """Below is a context and a question, please answer the question according by the context.if there is no answer in context, please answer 'No' directly.
Context:
```
{context}
```
Question:
```
{question}
```"""

prompts = [template_quest.format(context=Texts[i], question=Questions[i]) for i in range(len(Texts))]

load_dotenv()

apikey = os.environ["BAIDU_API_KEY"]
secretkey = os.environ["BAIDU_SECRET_KEY"]

ernie = BaiduErnie(apikey, secretkey)
result = ernie.inferer(prompts[0])
print(result)
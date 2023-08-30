"""
This file is used to generate prompt
"""
from questions import Questions, Texts

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

with open('prompt.txt', 'w+') as f:
  for i in range(len(Texts)):
    f.write(template_quest.format(context=Texts[i], question=Questions[i]) )
    f.write("\n\n\n")
  
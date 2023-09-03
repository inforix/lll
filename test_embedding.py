import torch

from dotenv import load_dotenv



torch.cuda.is_available = lambda: False

from qadoc import QA

load_dotenv()

qa = QA(embedding_model_path="../models/m3e-large",
        embedding_source="huggingface",
        model_type="wenxin",
        device="cpu"
        )

# print(qa.query("上海高级国际航运学院是哪一年成立的？")) #, "学校2013年成立中国（上海）自贸区供应链研究院和上海高级国际航运学院"))
# print(qa.query("上海海事大学有多少毕业生？")) #, "输送了逾19万毕业生"))
# print(qa.query("上海海事大学有几个博士点？")) ##, "4个一级学科博士点"))
# print(qa.query("上海海事大学有多少个硕士点？")) #, "17个一级学科硕士学位授权点"))
# print(qa.query("上海海事大学有马克思主义学院吗？")) #, "徐悲鸿艺术学院、马克思主义学院、"))
# print(qa.query("通知公告的主管部门是？")) #, "二、通知公告"))
# print(qa.query("离沪外出请假报告相关的规章制度有哪些？")) ##, "1.《上海海事大学领导干部离沪外出请假报告规定"))
# print(qa.query("信息化专项申报的联系方式是什么号码？")) ##, "38284493"))
# print(qa.query("2023年4月19日有什么活动？")) #, "4月19日"))


print(qa.check_similar("上海高级国际航运学院是哪一年成立的？", "学校2013年成立中国（上海）自贸区供应链研究院和上海高级国际航运学院"))
print(qa.check_similar("上海海事大学有多少毕业生？", "输送了逾19万毕业生"))
print(qa.check_similar("上海海事大学有几个博士点？", "4个一级学科博士点"))
print(qa.check_similar("上海海事大学有多少个硕士点？", "17个一级学科硕士学位授权点"))
print(qa.check_similar("上海海事大学有马克思主义学院吗？", "马克思主义学院"))
print(qa.check_similar("通知公告的主管部门是？", "二、通知公告"))
print(qa.check_similar("离沪外出请假报告相关的规章制度有哪些？", "1.《上海海事大学领导干部离沪外出请假报告规定"))
print(qa.check_similar("信息化专项申报的联系方式是什么号码？", "38284493"))
print(qa.check_similar("2023年4月19日有什么活动？", "4月19日"))
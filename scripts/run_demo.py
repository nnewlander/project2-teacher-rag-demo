from __future__ import annotations

from app.config import get_settings
from app.services.qa_service import QAService


def main() -> None:
    qa = QAService(get_settings())
    qa.init_kb()

    demo_queries = [
        "老师在作业批改台里找不到上节课的作业发布入口，是入口改版了吗？",
        "Python进阶 这节课里 字典操作 和 切片 总被学生混淆，课堂上怎么讲更顺？",
        "for循环 为什么这里会报错？",
    ]

    for q in demo_queries:
        resp = qa.ask(q)
        print("=" * 80)
        print("Q:", resp.query)
        print("mode:", resp.mode, "faq_id:", resp.faq_id)
        print("A:", resp.answer)
        print("contexts:", len(resp.contexts))


if __name__ == "__main__":
    main()


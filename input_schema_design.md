# 项目二输入数据格式设计（建议）

## 一、目标
为“核桃教师智能知识库问答系统（RAG）”提供可从原始态复现的输入数据格式。

## 二、输入源拆分
1. 文档类：课程资料、教师手册、平台说明、OCR 扫描资料  
2. 工单类：历史技术支持记录、多轮对话  
3. 代码类：课堂代码示例、题解、报错示例  
4. FAQ 类：高频标准问答  
5. 外部补充类：Python 官方文档、常用库说明、少量高质量教程  
6. 路由评测类：用于 BM25 / RAG / HyDE / 子查询 / 回溯检索 的验证查询

## 三、统一建库前推荐字段
- source_id
- source_type
- content_origin
- title
- raw_text
- product_module
- course_stage
- knowledge_tags
- issue_type
- version
- status
- updated_at
- noise_flags
- duplicate_group_id

## 四、设计原则
- 尽量保留 raw_text，不要一开始就改写成干净 QA
- 保留多轮工单上下文，后续再提炼问答对
- 保留 deprecated / draft / duplicate，用于版本治理和去重
- FAQ 不要只保留标准问法，原始问法也要保留
- 外部资料只做补充，不要和内部资料混成同权重

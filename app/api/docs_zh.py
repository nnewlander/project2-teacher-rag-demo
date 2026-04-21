from __future__ import annotations

from fastapi import APIRouter
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.responses import HTMLResponse


router = APIRouter()


@router.get("/docs", include_in_schema=False)
def docs_zh() -> HTMLResponse:
    """
    中文引导版 Docs：
    - 顶部提供中文“怎么用”的说明
    - 下方嵌入 Swagger UI（按钮本体仍可能为英文，这是 Swagger UI 默认行为）

    TODO:
    - 若需要“按钮级中文化”，可引入 swagger-ui 的 i18n 插件/自托管静态资源
    """

    swagger = get_swagger_ui_html(
        openapi_url="/openapi.json",
        title="教师智能知识库问答系统（RAG）- 接口文档",
        swagger_ui_parameters={
            "defaultModelsExpandDepth": 1,
            "docExpansion": "list",
            "displayRequestDuration": True,
        },
    )

    help_html = """
    <div style="font-family: -apple-system, Segoe UI, Arial, 'Microsoft YaHei', sans-serif;
                padding: 16px 16px 0 16px; line-height: 1.6;">
      <h2 style="margin: 0 0 8px 0;">怎么使用这个页面？</h2>
      <p style="margin: 0 0 10px 0;">
        这是系统自动生成的接口测试页。你可以在浏览器里直接调用 API，不需要写前端代码。
      </p>
      <ol style="margin: 0 0 10px 18px; padding: 0;">
        <li>找到 <b>POST /ask</b>（提问接口）</li>
        <li>点击 <b>Try it out</b>（试运行）</li>
        <li>在请求体里填入：<code>{"query":"你的问题"}</code></li>
        <li>点击 <b>Execute</b>（执行），在下方查看响应</li>
      </ol>
      <p style="margin: 0 0 10px 0;">
        响应字段说明：
        <code>mode</code> 为 <b>faq</b> 表示 FAQ 直达命中；为 <b>rag_mock</b> 表示走 RAG 兜底（当前为模拟生成）。
      </p>
      <hr style="border: 0; border-top: 1px solid #e5e7eb; margin: 12px 0 0 0;" />
    </div>
    """

    # swagger.body 是完整 HTML；我们把中文说明插入到 <body> 之后
    html = swagger.body.decode("utf-8")
    html = html.replace("<body>", "<body>" + help_html, 1)
    return HTMLResponse(html)


---
name: research_helper
description: 在用户要求读取网页、查看资料、基于 URL 总结内容或做轻量研究时，指导 Agent 使用 fetch 工具并区分工具事实与推断。
triggers:
  - 查资料
  - 资料
  - 网页
  - 链接
  - URL
  - research
  - fetch
tools:
  - mcp_fetch_url
  - web_fetch
---
# Research Helper

当用户要求读取网页、基于链接总结内容、查资料或做轻量研究时：

1. 如果用户给出明确 URL，优先使用 `mcp_fetch_url` 或 `web_fetch` 读取页面文本。
2. 如果用户没有给出 URL，不要假装已经查到资料；先说明需要链接，或只基于已有上下文回答。
3. 回答时区分“工具读取到的内容”和“基于内容的推断”。
4. 页面读取失败时，说明失败原因，并建议用户换链接或提供更多上下文。
5. 不要编造网页中不存在的数字、标题、作者或结论。
6. 对时效性问题保持谨慎，说明当前回答依据的是工具读取结果。

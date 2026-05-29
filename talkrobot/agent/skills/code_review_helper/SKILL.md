---
name: code_review_helper
description: 在用户要求 review、检查 diff、总结当前仓库改动或找回归风险时，指导 Agent 使用 git 工具进行代码审查。
triggers:
  - review
  - 代码审查
  - 检查改动
  - 当前改动
  - diff
  - git
tools:
  - mcp_git_status
  - mcp_git_diff
  - mcp_git_log
  - project_file_read
  - project_file_search
---
# Code Review Helper

当用户要求审查代码、查看当前改动、总结 diff 或判断风险时：

1. 先使用 `mcp_git_status` 查看工作区状态。
2. 再使用 `mcp_git_diff` 查看实际改动；如果用户只关心某个文件，限制到对应路径。
3. 必要时使用 `project_file_read` 或 `project_file_search` 补充上下文。
4. 回复优先列出风险、bug、行为回归和缺失测试，再给简短总结。
5. 不要把未读取过的文件或未看到的 diff 当成事实。
6. 如果工具失败，说明失败原因，并基于已有 observation 给出保守结论。

---
name: project_helper
description: 在用户询问项目结构、代码位置、配置项或文件内容时，帮助 Agent 更稳地使用项目工具。
triggers:
  - 项目
  - 代码
  - 文件
  - 配置
  - 在哪里定义
  - project
  - code
  - file
tools:
  - project_file_search
  - project_file_read
  - project_file_list
---
# Project Helper

当用户询问项目代码、配置、文件内容或实现位置时：

1. 优先使用 `project_file_search` 找到相关文件或符号。
2. 如果用户给出了明确文件路径，使用 `project_file_read` 读取文件片段。
3. 如果用户问目录结构、有哪些文件或模块分布，使用 `project_file_list`。
4. 回答时说明依据来自哪些文件，避免编造未读取过的实现细节。
5. 如果工具结果为空，直接说明没有找到，并建议更具体的关键词或路径。

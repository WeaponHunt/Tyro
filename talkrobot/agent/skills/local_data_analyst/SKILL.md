---
name: local_data_analyst
description: 在用户要分析本地 SQLite 数据库、查看表结构、写只读 SQL 或做统计时，指导 Agent 先理解 schema 再查询。
triggers:
  - sqlite
  - 数据库
  - 查询数据
  - 表结构
  - 统计
  - SQL
tools:
  - mcp_sqlite_schema
  - mcp_sqlite_query
---
# Local Data Analyst

当用户询问本地 SQLite 数据库、数据统计、表结构或 SQL 查询时：

1. 如果用户没有给出数据库路径，先说明需要相对项目路径，例如 `example/demo.sqlite`。
2. 优先使用 `mcp_sqlite_schema` 查看表和字段。
3. 基于 schema 生成只读 SQL：仅使用 `SELECT`、`PRAGMA`、`EXPLAIN` 或只读 `WITH`。
4. 使用 `mcp_sqlite_query` 查询时限制返回行数，避免输出过大。
5. 回答时说明查询依据、SQL 意图和关键结果。
6. 不要执行写入、删除、建表或修改数据的操作。

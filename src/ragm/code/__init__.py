"""代码库索引基础组件（v1：Python AST + SQLite FTS5）。

模块：
- scanner：目录扫描、排除规则、allowed_roots 越界保护、文件指纹；
- parser：stdlib ast 解析模块/类/函数/方法/嵌套符号与 import/reference；
- chunker：符号到检索块（超长按行切分，确定性 chunk_id）；
- store：SQLite（WAL + FTS5）持久化与检索；
- indexer：扫描 → 解析 → 切块 → 入库的增量索引器。

v1 仅支持 Python，检索为 FTS5 关键词匹配（中文效果有限），
反向 import 为文件级启发式，不承诺精确调用链。
"""

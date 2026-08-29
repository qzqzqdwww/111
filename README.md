# Feedback-Memory Code Review Agent

一个带**反馈记忆**的代码审查 Agent。审查完代码后，你可以通过"删除/提升严重级别"来教会它团队惯例——这些惯例会在后续相似审查中自动注入，且具备跨语言隔离和可验证的合规评分。

## 核心设计

### 两阶段审查

```
阶段 1  自由工具循环  →  模型调用 get_diff / read_file / grep_repo / run_linter
阶段 2  强制结构化输出 →  调用 emit_findings，返回 schema 约束的 JSON
```

**为什么分两阶段：** Anthropic 的 `tool_choice` 强制是获取结构化 JSON 的唯一可靠方式，但一旦 pin 住 tool_choice，模型就不能再调用其他工具。所以上下文收集和结构化输出必须在不同的 API 调用中完成。

### Prompt Cache 优化

系统提示被拆为两个 block：

| Block | 内容 | 缓存策略 |
|-------|------|---------|
| Block 0 | 审查规范 (~1024 token 以上) | `ephemeral` (长缓存) |
| Block 1 | 检索到的记忆规则 | 不缓存 (每次不同) |

实测：Sonnet 4 需要 ~1024 token 的最小缓存前缀才会生成 cache 条目。Block 0 刻意保持在这个阈值之上，且每一段都是真实的审查指导而非 padding。

### 两级检索

纯向量检索存在跨语言泄漏问题（Python 任务检索到了 Go 规则）。修复为两阶段：

1. **Meta 规则**（关于"如何报告"的规则）无条件注入
2. **其他规则**先按 scope/lang/area 标签过滤，再用 cosine × confidence × 半衰期衰减排序

## 快速开始

```bash
# 安装依赖
pip install -e ".[dev]"

# 配置 API
cp .env.example .env
# 编辑 .env，填入 ANTHROPIC_AUTH_TOKEN 和 ANTHROPIC_BASE_URL

# 检查连通性
python -m app.cli preflight

# 启动 Web UI
uvicorn app.web:app --reload --port 8077
```

## CLI 命令

```bash
python -m app.cli preflight              # 检查模型连通性
python -m app.cli review billing         # 审查 billing.py (记忆 ON)
python -m app.cli review pay --no-memory # 审查 pay.py (记忆 OFF 基线)
python -m app.cli feedback               # 对上轮审查结果教学
python -m app.cli rules                  # 浏览记忆库
python -m app.cli metrics                # 查看费用和延迟统计
python -m app.cli demo                   # 完整 learn-then-apply 演示
```

## Web UI

访问 `http://127.0.0.1:8077`：

1. 选择任务，点击 **Review** — Agent 会读取 diff 并生成 findings
2. 在 findings 表中点击 **delete**（不报告此类问题）或 **→ high**（提升严重级别）
3. 点击 **Teach the agent** — 系统将你的反馈蒸馏为可复用的规则
4. 右侧面板实时显示注入的规则、记忆库内容和费用统计

## 评测 (A/B)

```bash
# 运行记忆 ON vs OFF 的 A/B 评测
python -m eval.run_ab --repeats 3

# 输出 JSON
python -m eval.run_ab --repeats 3 --json ab_results.json
```

A/B 报告包含三个评分标准：

| 标准 | 度量 |
|------|------|
| 记忆成本 | embedding 费用 + 蒸馏费用占总费用的比例 |
| 对话速度 | 用户可见延迟 (检索 + 生成) |
| 记忆有效性 | 规则合规率 (ON - OFF delta) |

## 项目结构

```
app/
  __init__.py       # 包标记
  __main__.py       # python -m app 入口
  agent.py          # 两阶段审查 Agent
  catalog.py        # 审查任务定义 (pay, billing, auth, worker)
  cli.py            # Rich CLI 渲染层
  config.py         # 模型 ID、定价、路径、凭据
  distill.py        # 反馈 → 规则的蒸馏器
  embed.py          # 向量化 (httpx → /v1/embeddings)
  llm.py            # Anthropic SDK 封装 (重试、cache、费用)
  memory.py         # SQLite 记忆存储 + 两级检索
  metrics.py        # 每回合费用/延迟/合规记录
  pipeline.py       # CLI 和 Web 共享的编排层
  schemas.py        # 封闭词汇表和 JSON schema
  tools.py          # Agent 可调用的预设工具
  web.py            # FastAPI HTTP 接口
  static/
    index.html      # 单页 Web 前端
eval/
  __init__.py
  cases.py          # 种子规则集和 A/B 任务序列
  checkers.py       # 规则合规检查器 (确定性 + LLM judge)
  run_ab.py         # A/B 评测 Harness
tests/
  conftest.py       # pytest fixtures (fake embed)
  test_schemas.py   # 词汇表 + 规范化
  test_catalog.py   # 任务目录
  test_config.py    # 配置和定价
  test_tools.py     # Agent 工具
  test_distill.py   # 蒸馏和规则压缩
  test_memory.py    # 记忆存储和检索
  test_checkers.py  # 合规检查器
fixtures/
  repo/             # 模拟代码仓库
  diffs/            # 审查 diff 文件
data/               # 运行时数据 (gitignored)
```

## 封闭词汇表

所有 findings 的 `category` 必须来自以下集合，确保规则可被机器验证：

`correctness`, `error-handling`, `security`, `validation`, `reliability`, `observability`, `performance`, `style`

严重级别：`low`, `medium`, `high`

## 打包为 Windows .exe

需要 PyInstaller：

```bash
pip install pyinstaller
python build_exe.py
```

输出在 `dist/ReviewAgent/ReviewAgent.exe`。整个文件夹可以复制到任何 Windows 机器运行，无需安装 Python。

打包体积约 100-150 MB（包含 uvicorn、httpx、numpy 等依赖）。

## 依赖版本

| 包 | 版本 |
|----|------|
| Python | 3.11+ |
| anthropic | 1.0.0 |
| fastapi | 0.141.1 |
| pydantic | 2.13.4 |
| pytest | 9.1.1 |

# 本地 GUI Agent 分步确认 API 设计

## 1. 背景

Week 4 代码已经具备截图、OCR、多模态 Planner、安全策略、桌面动作执行器，以及有最大步数限制的观察—规划—执行循环。当前入口是同步 CLI：任务运行期间，安全策略通过终端输入逐动作确认。项目尚未提供 HTTP 服务，也没有可供客户端查询、确认、拒绝或取消某一步动作的状态模型。

本设计在 `codex/week4-end-to-end-agent` 工作树上增加一个仅限本机使用的 Web API。API 同时保留本地 Qwen3-VL 和 OpenAI-compatible Planner，让调用方能够创建 GUI 任务、检查模型建议的动作、逐项确认真实鼠标键盘操作，并观察每一步结果。

## 2. 目标

- 提供版本化的 FastAPI HTTP 接口和自动生成的 OpenAPI 3.1 定义。
- 支持 `qwen` 与 `openai-compatible` 两种现有模型 Provider。
- 把同步 Agent 循环整理为可暂停、可确认、可拒绝和可取消的分步执行核心。
- 默认使用 dry-run；真实执行必须在创建任务时显式开启，且每个待执行动作必须再次确认。
- 服务只监听 `127.0.0.1`，使用 Bearer Token，并保证同一时间只有一个活动 GUI 任务。
- 提供自动化测试以及能够在用户自己的 Windows 桌面上逐步执行的中文操作测试文档。
- 保持现有 `gui-agent run` CLI 的参数和安全行为兼容。

## 3. 非目标

- 不提供局域网或公网部署方式。
- 不提供多用户、数据库持久化、任务队列或跨进程恢复。
- 不把截图像素、模型密钥或确认令牌写入磁盘或普通日志。
- 不允许 HTTP 请求传入任意模型地址、API Key、系统提示词或桌面控制后端。
- 不绕过 PyAutoGUI fail-safe，不自动批准模型动作。
- 不把真实邮件、支付、账号、系统设置或不可逆操作纳入验收测试。

## 4. 选择的方案

采用有状态、逐动作确认 API。每个创建或确认请求只推进 Agent 到下一个稳定状态：终态或 `awaiting_confirmation`。HTTP 请求不会一次性授权模型未来可能生成的动作。

没有选择以下方案：

- 一次请求执行完整任务：无法逐项审查后续动作，真实桌面风险过高。
- 用 HTTP 包装 CLI 子进程：状态、取消、错误类型和确认过程难以可靠建模，并会复制 CLI 输出解析逻辑。

## 5. 总体架构

```text
本机 API 客户端
  │ Bearer Token
  ▼
FastAPI 路由与请求模型
  ▼
RunManager（单活动任务、历史上限、确认令牌）
  ▼
AgentRunSession（分步状态机）
  ├─ ObservationBuilder → ScreenCapture + EasyOCR
  ├─ MultimodalPlanner → Qwen3-VL / OpenAI-compatible
  ├─ SafetyPolicy      → 参数、坐标和安全边界校验
  └─ ActionExecutor    → DesktopController → PyAutoGUI
```

### 5.1 分层职责

- `gui_agent.agent.session`：与 HTTP 无关的分步 Agent 状态机。负责观察、规划、生成下一动作、执行已批准动作、最大步数、重复动作检测和终态结果。
- `gui_agent.agent.loop`：保留 `GUIAgent.run()` 同步入口，内部驱动分步状态机，并继续使用现有终端逐动作确认。
- `gui_agent.api.session`：管理 API 任务、一次性确认令牌、单活动任务互斥和有限的内存历史。
- `gui_agent.api.models`：定义请求、响应、观察摘要和错误结构。API 模型不直接暴露 NumPy 图像。
- `gui_agent.api.config`：读取并校验服务端环境变量，构造允许使用的 Provider。敏感配置不接受请求覆盖。
- `gui_agent.api.app`：创建 FastAPI 应用、认证依赖、路由和异常到 HTTP 的映射。

### 5.2 同步推进模型

`POST /v1/runs` 会同步完成初次截图、OCR、计划生成和第一个动作生成；`confirm` 会同步执行一个动作，再完成一次截图、OCR 和下一动作生成。调用结束时，任务必须位于一个稳定状态。因此客户端不需要轮询短暂的 `planning` 或 `executing` 状态，但响应和查询模型仍允许显示这两个内部状态以便诊断并发请求。

本地模型推理可能耗时较长，但项目只允许一个活动任务，不引入后台队列。取消请求只能阻止尚未开始的下一步，不能中断已经进入 PyAutoGUI 的单个动作或正在进行的模型调用。

## 6. Agent 分步状态机

状态集合：

```text
planning
awaiting_confirmation
executing
succeeded
failed
rejected
cancelled
```

合法转换：

```text
POST /runs
  planning
    ├─ 有待执行动作 → awaiting_confirmation
    ├─ finish(success=true) → succeeded
    ├─ finish(success=false) → failed
    └─ 观察/规划/校验失败 → failed

POST /runs/{id}/confirm
  awaiting_confirmation → executing
    ├─ 执行、再观察并得到下一动作 → awaiting_confirmation
    ├─ 得到 finish(success=true) → succeeded
    ├─ 得到 finish(success=false) → failed
    └─ 执行/观察/规划/校验失败 → failed

POST /runs/{id}/reject
  awaiting_confirmation → rejected

POST /runs/{id}/cancel
  planning | awaiting_confirmation → cancelled
```

每个非 `finish` 动作都进入 `awaiting_confirmation`，包括 `wait`，以保持客户端流程一致。`finish` 不触碰桌面，由服务校验后自动形成终态。

分步核心必须保留现有循环的行为：

- 空目标和非法 `max_steps` 在开始前失败。
- 初次观察后只创建一次计划。
- 每次已批准动作后重新观察，再请求下一动作。
- 连续两次完全相同的动作在第二次执行前停止。
- 达到最大步数后返回 `stopped` 语义；API 将其表示为 `failed`，错误代码为 `MAX_STEPS_REACHED`，并保留已完成结果。
- Planner 的 `current_step_id` 必须引用当前计划内的步骤，否则动作不得进入待确认状态。

## 7. HTTP API

所有 `/v1` 路由要求：

```http
Authorization: Bearer <GUI_AGENT_API_TOKEN>
Content-Type: application/json
```

`GET /health` 不要求认证，只返回服务是否启动，不暴露 Provider 地址、模型路径或密钥。

### 7.1 `GET /health`

返回：

```json
{
  "status": "ok",
  "service": "gui-agent-api",
  "version": "0.3.0"
}
```

### 7.2 `GET /v1/providers`

返回两个 Provider 的可用状态和非敏感模型名：

```json
{
  "providers": [
    {"name": "qwen", "configured": true, "model": "Qwen/Qwen3-VL-4B-Instruct"},
    {"name": "openai-compatible", "configured": false, "model": null}
  ]
}
```

不得返回 `api_key`、完整鉴权头或确认令牌。

### 7.3 `POST /v1/runs`

请求：

```json
{
  "goal": "在本地测试窗口中搜索 week4 safe search",
  "provider": "qwen",
  "monitor": 1,
  "max_steps": 10,
  "execute": true,
  "allow_remote_image": false
}
```

约束：

- `goal` 去除首尾空白后长度为 1 至 1000。
- `provider` 只能是 `qwen` 或 `openai-compatible`。
- `monitor` 是大于等于 1 的整数。
- `max_steps` 是 1 至 20 的整数。
- `execute` 默认为 `false`。
- `allow_remote_image` 默认为 `false`。远程 Provider 只有在服务端总开关和本次请求均为 `true` 时才可接收截图。

若已有活动任务，返回 `409 RUN_ALREADY_ACTIVE`。成功时返回完整 `RunResponse`；通常状态为 `awaiting_confirmation`，也可能因模型直接返回 `finish` 而成为终态。

### 7.4 `GET /v1/runs/{run_id}`

返回当前任务快照。响应不包含截图像素；观察摘要包含：

- 屏幕原点、宽度、高度、显示器编号和截图时间。
- OCR 检测的文字、置信度、边界框和中心点。
- 当前观察的步骤编号。

任务历史仅保存在当前服务进程内，最多保留最近 20 个终态任务；活动任务永不被淘汰。服务重启后所有任务和令牌失效。

### 7.5 `POST /v1/runs/{run_id}/confirm`

请求：

```json
{
  "confirmation_token": "opaque-single-use-token"
}
```

令牌由 `POST /runs` 或上一次 `confirm` 响应生成，绑定 `run_id`、步骤序号和动作内容。服务使用恒定时间比较；令牌只保存于内存，确认、拒绝、取消、任务终止或生成新动作后立即失效。

确认前，服务再次对同一个动作和产生它的观察结果执行安全校验。校验成功后令牌先失效，再调用执行器，避免重试 HTTP 请求导致动作重复执行。响应返回下一项待确认动作或终态。

### 7.6 `POST /v1/runs/{run_id}/reject`

请求可带 1 至 200 字的非敏感原因：

```json
{
  "reason": "坐标不在测试窗口内"
}
```

仅 `awaiting_confirmation` 状态可拒绝。拒绝后令牌失效，任务变为 `rejected`，不会执行当前动作。

### 7.7 `POST /v1/runs/{run_id}/cancel`

取消尚未进入单个执行动作的活动任务。终态任务、已进入不可中断动作的任务或重复取消返回 `409 RUN_STATE_CONFLICT`。

## 8. 响应模型

`RunResponse` 的主要字段：

```json
{
  "run_id": "uuid",
  "status": "awaiting_confirmation",
  "goal": "...",
  "provider": "qwen",
  "execute": true,
  "max_steps": 10,
  "plan": {
    "goal": "...",
    "steps": [{"id": "step-1", "description": "..."}]
  },
  "observation": {
    "step_index": 0,
    "monitor_index": 1,
    "captured_at": "2026-09-03T12:00:00Z",
    "origin": {"x": 0, "y": 0},
    "width": 1920,
    "height": 1080,
    "detections": []
  },
  "pending_action": {
    "step_index": 0,
    "decision": {
      "current_step_id": "step-1",
      "rationale_summary": "...",
      "action": {"kind": "click", "x": 500, "y": 300, "button": "left", "clicks": 1},
      "expected_outcome": "搜索框获得焦点"
    },
    "confirmation_token": "opaque-single-use-token"
  },
  "decisions": [],
  "results": [],
  "failure_stage": null,
  "message": "waiting for confirmation"
}
```

`decisions` 只包含已经提出的结构化决策；`results` 只包含已执行、dry-run、拒绝或失败的步骤结果。真实键盘输入会在本机认证响应中出现，方便人工核对，但不会写入服务日志。

## 9. 错误模型

统一错误结构：

```json
{
  "error": {
    "code": "ACTION_CONFIRMATION_STALE",
    "message": "确认令牌已失效或不属于当前动作",
    "run_id": "uuid"
  }
}
```

错误映射：

| HTTP | 代码示例 | 含义 |
|---|---|---|
| 400 | `RUN_REQUEST_INVALID` | 跨字段约束或业务参数无效 |
| 401 | `AUTHENTICATION_REQUIRED` | Bearer Token 缺失或错误 |
| 404 | `RUN_NOT_FOUND` | 任务不存在或已经从内存历史淘汰 |
| 409 | `RUN_ALREADY_ACTIVE` | 已有一个活动任务 |
| 409 | `ACTION_CONFIRMATION_STALE` | 令牌错误、过期或已使用 |
| 409 | `RUN_STATE_CONFLICT` | 当前状态不允许确认、拒绝或取消 |
| 422 | `REQUEST_VALIDATION_FAILED` | Pydantic 字段校验失败 |
| 502 | `PLANNER_FAILED` | Provider 调用失败或返回结构无效 |
| 503 | `PROVIDER_UNAVAILABLE` | Provider 未配置、依赖未安装或不可初始化 |
| 500 | `OBSERVATION_FAILED` | 截图或 OCR 失败 |
| 500 | `ACTION_EXECUTION_FAILED` | 已批准动作执行失败 |

响应不得包含堆栈、API Key、完整模型原始输出或内部文件路径。完整异常通过 `__cause__` 保留给本地调试，但普通日志只记录异常类型、错误代码和任务 ID。

## 10. 安全设计

### 10.1 网络边界

- 官方启动命令固定绑定 `127.0.0.1`，不接受 `0.0.0.0` 或自定义主机参数。
- 不安装或启用 CORS 中间件。
- `/v1` 使用必填的 `GUI_AGENT_API_TOKEN`；启动时拒绝空白或少于 16 个字符的令牌。
- 使用 `secrets.compare_digest` 校验 Bearer Token。

### 10.2 Provider 边界

- 请求只能选择服务端已经配置的 Provider 名称。
- 本地 Qwen 模型名来自 `GUI_AGENT_QWEN_MODEL`，缺省值沿用现有 Qwen 常量。
- 远程配置来自 `GUI_AGENT_REMOTE_MODEL`、`GUI_AGENT_API_BASE` 和 `GUI_AGENT_API_KEY`。
- 远程截图需要 `GUI_AGENT_ALLOW_REMOTE_IMAGE=true` 与请求 `allow_remote_image=true` 双重满足。
- 为兼容现有 CLI，已有 `GUI_AGENT_MODEL` 仍可作为对应模型变量未设置时的后备值。

### 10.3 桌面执行边界

- `execute=false` 时 `DesktopController` 始终为 dry-run。
- `execute=true` 不代表自动授权，只让已确认动作能够进入真实控制器。
- 动作在生成时和确认执行前各校验一次。
- 每次确认令牌只能消费一次；消费先于桌面调用。
- PyAutoGUI fail-safe 保持开启。操作测试文档要求执行前把鼠标移到主显示器左上角验证紧急停止方式。
- API 不接受文件路径、shell 命令、任意 Python 或额外工具定义。

### 10.4 并发边界

- `RunManager` 使用进程内锁保护创建、确认、拒绝和取消。
- 只允许一个非终态任务；终态后立即释放活动任务槽位。
- 同一动作的并发确认只有一个请求能消费令牌，其余请求返回 `409`。
- 多进程 Uvicorn 会破坏进程内互斥，因此官方入口固定单 worker，并在文档中禁止 `--workers`。

## 11. 配置与启动

新增 `api` 可选依赖组，包含 FastAPI、Uvicorn 和 API 测试所需的 HTTP 客户端。普通核心安装不被强制引入 Web 框架。

`.env.example` 记录变量名但不放真实密钥：

```text
GUI_AGENT_API_TOKEN=replace-with-at-least-16-characters
GUI_AGENT_QWEN_MODEL=Qwen/Qwen3-VL-4B-Instruct
GUI_AGENT_REMOTE_MODEL=
GUI_AGENT_API_BASE=
GUI_AGENT_API_KEY=
GUI_AGENT_ALLOW_REMOTE_IMAGE=false
GUI_AGENT_API_PORT=8765
```

官方入口：

```powershell
uv run --extra agent --extra api --extra ocr gui-agent api
```

入口固定主机为 `127.0.0.1`、固定单 worker，端口从 `GUI_AGENT_API_PORT` 读取，缺省为 `8765`。应用工厂保持无启动副作用，自动化测试通过依赖注入假的 Session/Provider/桌面控制器。

## 12. 代码改动

新增：

```text
src/gui_agent/agent/session.py
src/gui_agent/api/__init__.py
src/gui_agent/api/app.py
src/gui_agent/api/config.py
src/gui_agent/api/models.py
src/gui_agent/api/session.py
tests/test_agent_session.py
tests/test_api_app.py
tests/test_api_session.py
tests/test_api_openapi.py
scripts/export_openapi.py
docs/api/gui-agent-api.md
docs/api/openapi.json
docs/test-reports/week4-api-operation-test.md
```

修改：

```text
src/gui_agent/agent/loop.py
src/gui_agent/agent/policy.py
src/gui_agent/cli.py
pyproject.toml
uv.lock
.env.example
README.md
```

其中 `SafetyPolicy` 增加不触发终端输入的纯校验入口；原 `authorize()` 继续组合校验与 CLI 人工确认。`GUIAgent.run()` 改为驱动 `AgentRunSession`，保留现有返回模型和终端行为。

## 13. API 定义文档

- FastAPI 运行时提供 `/openapi.json` 和 `/docs`。
- `scripts/export_openapi.py` 从应用工厂确定性生成 `docs/api/openapi.json`。
- `tests/test_api_openapi.py` 校验已提交 Schema 与应用生成结果一致，防止文档漂移。
- `docs/api/gui-agent-api.md` 使用中文解释认证、配置、端点、状态机、完整 PowerShell 调用示例、错误代码和隐私边界。

## 14. 测试策略

### 14.1 分步 Agent 单元测试

使用真实状态机与 fake observer、planner、policy validator 和 executor，覆盖：

- 初次观察、一次计划创建及首个待确认动作。
- 未确认时执行器零调用。
- 确认后只执行当前动作，并重新观察和生成下一动作。
- 拒绝和取消不会调用执行器。
- 连续重复动作、最大步数和非法计划步骤 ID。
- 观察、规划、安全校验和执行阶段的稳定失败结果。
- `finish` 自动形成成功或任务失败终态。
- 同步 `GUIAgent.run()` 的现有行为保持不变。

### 14.2 API 与 Session 管理测试

使用 FastAPI TestClient 和 fake 依赖，覆盖：

- 健康检查、Bearer Token 正确/缺失/错误。
- 两种 Provider 的可用状态和服务端配置隔离。
- 创建任务的字段与跨字段校验。
- 单活动任务冲突。
- 一次性令牌、错误令牌、重复确认和并发确认。
- 确认、拒绝、取消及所有终态响应。
- 远程图片双重授权。
- Provider、截图/OCR、执行异常的 HTTP 映射。
- 响应及日志中不出现 API Key、截图像素或内部异常文本。
- OpenAPI 导出与实际应用 Schema 一致。

所有普通自动化测试必须使用 fake 后端，不得截图真实桌面、加载真实模型、联网或触发鼠标键盘。

### 14.3 本机人工操作测试

`docs/test-reports/week4-api-operation-test.md` 是可执行的中文测试手册，而不是宣称已经完成的结果报告。它包含：

1. 环境、显示器编号、模型与凭据预检。
2. 启动本地 `examples/gui_testbed.py`。
3. 启动 API 并验证仅监听 `127.0.0.1`。
4. 使用 dry-run 创建任务，核对真实屏幕尺寸、OCR 文本、计划和待执行坐标。
5. 使用本地 Qwen 在 Testbed 完成打开 Browser、输入搜索内容、打开本地示例文件、发送内存消息和关闭 Testbed 五类任务。
6. 对每个动作执行 `GET`、人工核对 `pending_action`、复制令牌、调用 `confirm`、观察窗口变化并记录结果。
7. 使用 OpenAI-compatible Provider 重复一个低风险任务，明确记录远程截图双重授权。
8. 验证错误令牌、重复确认、拒绝、取消和第二个并发任务。
9. 验证 PyAutoGUI fail-safe，并记录恢复 API/Testbed 的步骤。
10. 填写预期结果、实际结果、证据文件名、通过/失败和问题备注。

手册中的真实执行命令仅针对本项目 Testbed。每个会触碰桌面的章节开头重复显示风险提示和紧急停止方式。

## 15. 验证命令

实现完成后运行：

```powershell
uv lock --check
uv run --extra agent --extra api ruff check .
uv run --extra agent --extra api mypy src tests examples scripts
uv run --extra agent --extra api pytest -m "not integration" --cov=gui_agent --cov-report=term-missing
uv run --extra agent --extra api python scripts/export_openapi.py --check
```

真实 Qwen、远程 API 和真实桌面操作不属于普通测试门禁，必须按人工操作测试文档显式运行。

## 16. 验收标准

- API 只能通过官方入口绑定 `127.0.0.1`，并以单 worker 运行。
- 未认证请求无法访问 `/v1`。
- 本地 Qwen 与 OpenAI-compatible Provider 均可由服务端配置并由请求选择。
- 远程截图只有双重授权后才能发送。
- 同一时间只能存在一个活动任务。
- 未提交当前一次性令牌时，任何模型动作都不会到达桌面控制器。
- 一个令牌最多导致一次动作执行，旧令牌和并发重复确认均被拒绝。
- 拒绝、取消、失败和最大步数都形成明确终态并释放活动任务槽位。
- 原 `gui-agent run` CLI 的同步 dry-run 与真实逐动作确认行为保持兼容。
- OpenAPI 定义、中文接口文档和本机逐步操作测试文档均已提交。
- 普通非集成测试、Ruff、mypy、锁文件检查和 OpenAPI 漂移检查全部通过。

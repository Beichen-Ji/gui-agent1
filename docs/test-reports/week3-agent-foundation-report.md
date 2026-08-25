# Week 3 数据与多模态 Planner 测试报告

## 1. 范围与日期

- 测试日期：2026-08-25（America/Toronto）。
- 分支：`codex/week3-data-agent-foundation`。
- 基线：`origin/master` 的 `95a026e`（Week 2 PR #2 合并提交）。
- 范围：Agent/Pydantic schema、三个公开 GUI 数据适配器、结构化 prompt、fake Planner、OpenAI-compatible Planner、本地 Qwen3-VL Planner、合成界面 smoke test。
- 不包含：第 4 周观察—动作循环、真实桌面截图、真实输入控制、外部 API 账户调用、训练或微调。

## 2. 本机环境

| 项目 | 实测值 |
|---|---|
| Windows / Python | Windows x64 / Python 3.11.16 |
| GPU | NVIDIA GeForce RTX 5070 Ti，16303 MiB |
| NVIDIA driver | 595.95 |
| PyTorch / CUDA build | 2.11.0+cu128 / 12.8 |
| Accelerate | 1.14.0 |
| Transformers | 5.15.1 |
| Safetensors | 0.8.0 |
| Hugging Face Hub | 1.28.0 |
| Hugging Face Datasets | 4.8.5 |
| LangChain / langchain-openai | 1.3.17 / 1.6.0 |
| Pydantic | 2.13.4 |
| pytest | 9.1.1 |

`torch.cuda.is_available()` 返回 `True`。本地 Qwen 使用 BF16 与 `device_map="auto"`；普通测试不初始化 CUDA 模型。

## 3. 实现结果

### 3.1 Agent schema

已实现并严格验证：

- `ClickAction`、`TypeTextAction`、`HotkeyAction`、`ScrollAction`、`DragAction`、`WaitAction`、`FinishAction`；
- `TaskStep`、`TaskPlan`、`AgentDecision`、`StepResult`；
- 携带截图、OCR 检测、历史决策与执行结果的 `Observation`、`AgentState`；
- 三个数据源共用的 `NormalizedGUIRecord` 与 `DatasetManifest`。

Pydantic 模型禁止额外字段并采用不可变配置；动作使用 `kind` 判别联合，避免运行时猜测字典形状。

### 3.2 Planner 边界

- `FakePlanner`：单元测试和后续 Agent 循环的确定性替身。
- `LangChainPlanner`：通过 `ChatOpenAI` 兼容接口发送图像消息，使用 JSON Schema 结构化输出；未显式允许远程图像时拒绝调用。
- `QwenTransformersPlanner`：延迟加载 Qwen3-VL，直接传递内存中的 Pillow 图像，限制图像长边和输出 token。
- prompt 包含目标、当前步骤、OCR、历史结果和动作白名单，并遮盖常见本地路径与疑似 token/key 文本。

Provider、解析或加载异常统一包装为 `PlannerError`，同时保留原异常为 `__cause__`，便于诊断。

## 4. 真实公开数据 smoke test

三个来源都固定 revision，并只写入前 100 条标准化记录：

| 数据源 | revision | 许可 | 写入 | 跳过 | `records.jsonl` SHA-256 |
|---|---|---|---:|---:|---|
| ScreenAgent | `2312c852908dda5fbc43d903fad5929a9dec649d` | 数据集 Apache-2.0；代码 MIT | 100 | 139 | `602352e252c7a9cdb597ee52542843ba3881075639599ce02d6397024ff55f72` |
| Mind2Web | `17ece8eb89862368edc0cc806acee6fca5163474` | CC BY 4.0 | 100 | 14 | `e4799a8ec325b6f41a8e08df817771fb37af91bcc38e4f506b1b64b72006c05e` |
| WebArena | `dce04686a56253aefba7b18a4fa0937cf1dc987b` | Apache-2.0 | 100 | 1 | `f839a7e23b143d6ed8814aa2a8ca48f6af194eba079fe23930e9d9b5bb73b91e` |

`跳过` 同时包含无法无歧义映射的源动作，以及在确定性排序后被 `--limit` 截断的记录。CLI 最多读取 `limit + 1` 个流式记录，不会为了 100 条样本耗尽整个 Mind2Web 数据流。

许可证核对时发现 ScreenAgent manifest 最初只写了代码 MIT。上游同一 revision 的 README 明确区分“数据集 Apache-2.0、代码 MIT”；增加回归测试并修正后重新生成 manifest，JSONL SHA 保持不变。

### 4.1 本机磁盘占用

| 路径 | 本机占用 |
|---|---:|
| `data/processed/`（三个 JSONL + manifest） | 999,808 bytes（0.95 MiB） |
| `external/ScreenAgent/` 浅克隆 | 293,075,814 bytes（279.50 MiB） |
| `external/webarena/` 浅克隆 | 7,593,247 bytes（7.24 MiB） |
| Mind2Web Hub 元数据缓存 | 4,348 bytes；样本使用 streaming |

以上路径全部被 Git 忽略；`git status` 不包含任何上游数据或标准化记录。

## 5. 本地 Qwen 真实 CUDA 测试

### 5.1 模型与缓存

- 模型：`Qwen/Qwen3-VL-4B-Instruct`。
- 实测 revision：`ebb281ec70b05090aa6165b016eac8ec08e71b17`。
- 两片权重：8,875,719,344 bytes（8.266 GiB）。
- 缓存位置：仓库内 `.cache/huggingface/`，已被 Git 忽略。
- 当前模型缓存总占用：11,886,213,219 bytes（11.070 GiB）。其中 2,998,927,360 bytes（2.793 GiB）是首次下载被中断后保留的 `*.incomplete` 文件；未在进程运行期间删除它。

首次缓存准备约 30 分钟，包含代理连接停滞、下载器重启和断点恢复，不代表正常网络的模型下载性能。最终两片权重均按 Hub tree 元数据验证大小，第二片额外通过完整 SHA-256 `046296a2a387efb43b0c997d5833c789604d168834f6e0d3064bf7bb13d002a6` 校验。随后设置 `HF_HUB_OFFLINE=1`，确认模型可完全离线加载。

### 5.2 合成界面 smoke test

执行：

```powershell
uv run python examples/model_smoke.py `
  --provider qwen `
  --model Qwen/Qwen3-VL-4B-Instruct `
  --synthetic
```

结果：退出码 0；模型为“Open the synthetic browser”生成一条合法计划，并返回位于合成 Browser 按钮内部的左键单击 `(150, 100)`。命令总耗时 18.1 秒。没有读取、保存或发送真实桌面图像，也没有调用控制器。

### 5.3 集成测试与性能

执行：

```powershell
$env:GUI_AGENT_RUN_LOCAL_QWEN = "1"
uv run pytest -m integration tests/integration/test_local_qwen.py -v -s
```

| 指标 | 实测值 |
|---|---:|
| 模型加载 | 7.813 s |
| 计划推理 | 2.735 s |
| 动作推理 | 3.928 s |
| PyTorch 峰值分配显存 | 9.021 GiB |
| pytest | 1 passed in 16.13 s |

这些数字只代表本机、当前驱动、640×360 合成图和短输出；不是通用吞吐或延迟保证。

## 6. 完整质量门禁

执行：

```powershell
uv lock --check
uv run ruff check .
uv run mypy src tests examples scripts
uv run pytest -m "not integration" --cov=gui_agent --cov-report=term-missing
git diff --check
```

结果：

- `uv lock --check`：通过，解析 150 个锁定包；
- Ruff：通过；
- mypy strict：40 个源文件通过；
- pytest：收集 196 项，1 个真实模型集成测试取消选择，195 项普通测试全部通过；
- 覆盖率：90%（1014 statements，101 missed）；
- `git diff --check`：通过。

普通门禁不联网、不下载/加载模型、不截取屏幕、不执行鼠标键盘动作。真实 Qwen 测试由独立环境变量显式启用，并已按第 5 节单独通过。

### 6.1 干净 CI 回归

草稿 PR 的首次 GitHub Actions 运行暴露了本机全量环境没有显示的问题：旧 workflow 只安装基础/dev 依赖，因此 mypy 找不到第 3 周所需的 Pydantic 和 `langchain-openai`；同时 Qwen 单元测试曾在文件顶层导入 torch，会迫使普通 CI 安装大模型依赖。

修复后，workflow 安装轻量 `agent` extra 并把 `scripts` 纳入 mypy；fake Qwen 测试注入 dtype，真实 integration 测试只在 `GUI_AGENT_RUN_LOCAL_QWEN=1` 后导入 torch。使用以下隔离命令模拟全新 CI：

```powershell
uv run --isolated --locked --group dev --extra agent `
  python -c "import importlib.util; assert importlib.util.find_spec('torch') is None"
uv run --isolated --locked --group dev --extra agent ruff check .
uv run --isolated --locked --group dev --extra agent mypy src tests examples scripts
uv run --isolated --locked --group dev --extra agent `
  pytest -m "not integration" --cov=gui_agent --cov-report=term-missing
```

隔离环境确认 torch 未安装；Ruff、mypy 和 195 项普通测试仍全部通过。

## 7. 隐私与安全结论

- Week 3 所有模型实测只使用程序内生成的合成图片。
- 本地 Qwen 图像只存在于进程内存和 GPU；模型缓存不含桌面内容。
- `LangChainPlanner` 默认 `allow_remote_image=False`，未授权时在编码/调用 provider 前失败。
- API key 使用 `SecretStr` 传入 LangChain；`.env`、日志、截图、模型和数据缓存均在 `.gitignore` 中。
- prompt 文本脱敏不能移除截图像素里的个人信息；未来真实远程观察仍需显式用户同意和发送前检查。
- 结构化模型输出不会在第 3 周自动传给 `DesktopController`。

## 8. 已知限制

- 尚未实现观察—动作循环、步骤重试、重新规划、终止条件、轨迹日志或控制器桥接。
- schema 验证动作形状，但尚未强制 `AgentDecision.current_step_id` 一定引用当前计划中的 step；真实 smoke 输出出现过计划 ID `click_browser`、决策 ID `step_0` 的语义不一致。第 4 周运行时必须在执行前校验并拒绝/重规划。
- OpenAI-compatible 路径通过依赖注入 fake 覆盖图像编码、结构化输出、授权和异常链；本次没有使用真实外部 API 凭据做网络集成测试。
- 三个数据来源只做 100 条 smoke；ScreenAgent 中 Plan/Evaluate 等高层动作被明确跳过，当前 schema 不试图猜测其含义。
- 本地默认模型跟随 Hub `main`；未来重新下载可能得到新 revision，需要重新记录版本与性能。
- Windows 原生不使用 vLLM；当前采用 Transformers。WSL/Linux 服务化属于后续部署工作。

## 9. 第 4 周入口条件

本报告对应的 Week 3 分支应先创建草稿 PR，并由用户人工审阅、合并。只有确认 PR 已合并到 `master` 后，才从更新后的 `master` 创建第 4 周分支，开始 Agent 循环与执行安全策略。

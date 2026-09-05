# 第 3 周数据集与多模态模型运行指南

本文说明如何在 Windows PowerShell 中准备三个公开 GUI 数据源，并运行 fake、本地 Qwen 或 OpenAI-compatible Planner。第 3 周只验证“观察 → 计划/动作”的模型边界；示例始终使用程序生成的合成界面，不截取桌面，也不执行鼠标键盘动作。

## 1. 安装依赖

项目要求 Python `>=3.11,<3.12`。先在仓库根目录安装 uv 环境：

```powershell
uv python install 3.11
uv sync --locked --group dev --extra agent --extra datasets
```

本地 Qwen 还需要 Transformers、Accelerate 和 CUDA PyTorch。PyTorch 复用现有 `ocr` extra，因此本地模型安装命令为：

```powershell
uv sync --locked --group dev `
  --extra agent `
  --extra datasets `
  --extra local-model `
  --extra ocr
```

检查 GPU：

```powershell
uv run python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

本地 Qwen 路径需要可用的 NVIDIA CUDA GPU。fake 和 OpenAI-compatible 路径不要求本机加载模型。

## 2. 准备公开数据集

所有仓库、原始数据和标准化输出都位于 `.gitignore` 已排除的 `external/`、`.cache/huggingface/` 和 `data/`。不要把上游数据、网页内容或截图提交到本仓库。

### 2.1 获取固定 revision

```powershell
git clone --depth 1 https://github.com/niuzaisheng/ScreenAgent external/ScreenAgent
git clone --depth 1 https://github.com/web-arena-x/webarena external/webarena

git -C external/ScreenAgent rev-parse HEAD
git -C external/webarena rev-parse HEAD
```

本次验证使用：

| 数据源 | revision | 本地核对的许可 |
|---|---|---|
| ScreenAgent | `2312c852908dda5fbc43d903fad5929a9dec649d` | 数据集 Apache-2.0；代码 MIT |
| Mind2Web | `17ece8eb89862368edc0cc806acee6fca5163474` | CC BY 4.0 |
| WebArena | `dce04686a56253aefba7b18a4fa0937cf1dc987b` | Apache-2.0 |

许可证结论分别来自对应 revision 的 ScreenAgent `README.md`/`LICENSE`、Mind2Web Hugging Face dataset card 和 WebArena `LICENSE`。许可证与数据条款可能随上游变化；重新选择 revision 或再分发数据前必须再次核对。

### 2.2 生成每个来源的 100 条标准记录

```powershell
uv run python scripts/prepare_gui_datasets.py screenagent `
  --input external/ScreenAgent/data/ScreenAgent `
  --output data/processed/screenagent `
  --revision 2312c852908dda5fbc43d903fad5929a9dec649d `
  --limit 100

uv run python scripts/prepare_gui_datasets.py mind2web `
  --dataset osunlp/Mind2Web `
  --split train `
  --stream `
  --output data/processed/mind2web `
  --revision 17ece8eb89862368edc0cc806acee6fca5163474 `
  --limit 100

uv run python scripts/prepare_gui_datasets.py webarena `
  --input external/webarena/config_files `
  --output data/processed/webarena `
  --revision dce04686a56253aefba7b18a4fa0937cf1dc987b `
  --limit 100
```

每个输出目录包含：

- `records.jsonl`：按来源、split、episode 和 step 确定性排序的统一记录；
- `manifest.json`：来源 URL、revision、许可、写入/跳过数量和 JSONL SHA-256。

适配器只映射能够无歧义转换为项目动作 schema 的记录。无法支持或格式错误的动作会计入 `records_skipped` 并报告源路径，不会被静默吞掉。Mind2Web 使用 streaming，因此 `--limit 100` 不会下载整个训练集。

## 3. 运行不加载模型的 fake smoke test

先用 fake 验证完整的 CLI 和结构化输出：

```powershell
uv run python examples/model_smoke.py --provider fake --synthetic
```

预期打印一个合法 `TaskPlan` 和一个 `AgentDecision`。`--synthetic` 是强制参数；该程序没有真实截图入口。

## 4. 运行本地 Qwen3-VL

### 4.1 设置仓库内缓存

```powershell
$env:HF_HOME = Join-Path $PWD ".cache\huggingface"
$env:GUI_AGENT_MODEL = "Qwen/Qwen3-VL-4B-Instruct"
```

`HF_HOME` 必须在导入 Hugging Face 库之前设置。该路径已被 Git 忽略。首次运行会下载约 8.9 GB 十进制权重；建议至少预留 12 GB 磁盘空间，因为中断的下载可能留下额外 `*.incomplete` 文件。

### 4.2 首次联网运行

```powershell
uv run python examples/model_smoke.py `
  --provider qwen `
  --model $env:GUI_AGENT_MODEL `
  --synthetic
```

实现按 Qwen 官方模型卡使用 `AutoProcessor` 和 `AutoModelForMultimodalLM`，以 `torch.bfloat16`、`device_map="auto"` 延迟加载模型；输入图像长边最多 1280 像素，单次最多生成 512 token。

缓存完整后可禁止所有 Hub HTTP 请求，验证离线加载：

```powershell
$env:HF_HUB_OFFLINE = "1"
uv run python examples/model_smoke.py --provider qwen --synthetic
```

若缓存不完整，离线模式会明确失败。删除当前会话中的离线设置后联网补齐：

```powershell
Remove-Item Env:HF_HUB_OFFLINE -ErrorAction SilentlyContinue
```

Hugging Face 官方说明见 [`HF_HOME`、`HF_HUB_OFFLINE` 等环境变量](https://huggingface.co/docs/huggingface_hub/package_reference/environment_variables)，Qwen 加载接口见 [Qwen3-VL-4B-Instruct 模型卡](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct)。

### 4.3 真实模型集成测试

普通 pytest 默认跳过模型测试。只有显式设置开关时才加载真实模型：

```powershell
$env:GUI_AGENT_RUN_LOCAL_QWEN = "1"
uv run pytest -m integration tests/integration/test_local_qwen.py -v -s
```

测试只使用 640×360 合成界面，输出加载时间、计划/动作推理时间和 PyTorch 峰值分配显存，不读取真实屏幕。

### 4.4 加载 Week 5 LoRA adapter

`--adapter` 接受 `training train/check` 的输出目录，或其中的 `adapter/` 子目录。例如：

```powershell
uv run --no-sync gui-agent run `
  --task-id open-browser `
  --provider qwen `
  --adapter artifacts/week5/qwen3vl-gui-lora `
  --max-steps 8
```

Planner 会在加载权重前检查 `run-manifest.json`、adapter 文件哈希、基础模型、1000
坐标网格和 prompt profile。未传 `--adapter` 时仍使用 Week 4 基础模型和
`week4-baseline` prompt；adapter 路径只支持本地 Qwen provider。adapter 不包含密钥，
也不会绕过 dry-run、安全策略或逐动作确认。

## 5. 运行 OpenAI-compatible Provider

`.env.example` 只是字段模板，项目不会自动读取 `.env`。推荐在当前 PowerShell 会话或系统密钥管理器中设置：

```powershell
$env:GUI_AGENT_API_BASE = "http://127.0.0.1:8000/v1"
$env:GUI_AGENT_API_KEY = "replace-with-your-key"
$env:GUI_AGENT_MODEL = "your-vision-model"
```

远程/服务端模型会收到 PNG 图像。Planner 默认拒绝发送；必须显式添加 `--allow-remote-image`：

```powershell
uv run python examples/model_smoke.py `
  --provider openai-compatible `
  --model $env:GUI_AGENT_MODEL `
  --api-base $env:GUI_AGENT_API_BASE `
  --allow-remote-image `
  --synthetic
```

不要把 API key 写入命令行、日志、Markdown 或 Git。CLI 会从 `GUI_AGENT_API_KEY` 读取密钥，因此示例不传 `--api-key`。Provider 必须支持视觉消息和 OpenAI 风格的 JSON Schema 结构化输出；不支持时会返回保留原异常为 `__cause__` 的 `PlannerError`。

## 6. 图片隐私边界

| 路径 | 图像去向 | 默认保护 |
|---|---|---|
| fake | 不调用模型 | 仅返回固定结构 |
| 本地 Qwen | 进程内内存与本机 GPU | 不发往网络；缓存仅含模型文件 |
| OpenAI-compatible | 编码为 PNG data URL 后发给配置的 API | `allow_remote_image=False` 时拒绝调用 |

文本 prompt 会遮盖常见本地绝对路径和疑似 key/token 字符串，但这不能清除截图像素里的个人信息。真实桌面截图在发送前仍必须由用户自行检查。第 3 周 smoke test 只生成合成图，完全不触碰桌面。

## 7. 为什么 Windows 默认不用 vLLM

vLLM 官方 GPU 安装文档当前要求 Linux，并明确说明不原生支持 Windows；Windows 需要 WSL 或社区维护分支。因此本阶段直接使用原生 Windows Transformers，减少额外虚拟化与 CUDA 兼容层。后续若在 WSL/Linux 部署 vLLM，可把它作为 OpenAI-compatible 服务，再由 `LangChainPlanner` 调用。官方说明见 [vLLM GPU installation](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/)。

## 8. 常见问题

### 模型下载超时或停滞

先确认代理、防火墙和剩余磁盘空间。慢速连接可在导入 Hugging Face 前增加超时：

```powershell
$env:HF_HUB_DOWNLOAD_TIMEOUT = "300"
```

若 Xet 后端在当前代理环境停滞，可临时使用普通 HTTP：

```powershell
$env:HF_HUB_DISABLE_XET = "1"
```

不要在 Python/uv 下载进程仍运行时清理缓存，也不要递归删除仓库或用户目录。先停止本项目启动的准确进程，再只检查模型缓存下的 `*.incomplete` 文件。

### CUDA 显存不足

关闭占用 GPU 的其他程序并重试。本机 16 GB 显存的合成测试峰值分配约 9.0 GiB；更大的截图、上下文或输出上限会增加占用。不要同时运行 EasyOCR CUDA 和本地 Qwen 压力测试。

### API 返回结构化输出错误

确认服务支持图像消息、所选模型具备视觉能力，并支持 OpenAI 风格 JSON Schema。先运行 fake 路径排除本地 schema/CLI 问题，再查看 `PlannerError.__cause__` 中的 provider 原始错误。

## 9. 当前范围限制

- 第 3 周只有 schema、prompt、数据适配器和 Planner；观察—动作循环、重试、历史管理与安全执行属于第 4 周。
- smoke test 不读取真实桌面，Planner 的结构化动作也不会自动交给 `DesktopController`。
- 本地实测 revision 只是报告中的可复核快照；以后从模型 `main` 重新下载可能得到新 revision，应重新运行集成测试并更新报告。

# GUI Agent：桌面感知、数据与多模态规划基础

这是“大模型 GUI Agent”实习项目第 1.6 节至第 5 周的实现：先用 MSS、EasyOCR 和 PyAutoGUI 建立桌面感知/安全控制，再加入公开 GUI 数据标准化、结构化 Agent、本地 Qwen 多模态规划、安全的观察—执行循环，以及可复现的 4-bit QLoRA 训练与离线评测。项目不包含第 1 周调研报告。

## 已实现

- 多显示器整屏/区域截图，统一输出 `uint8` BGR 图像和绝对屏幕坐标。
- 可替换的 OCR 接口，以及默认中英文 EasyOCR CUDA 后端。
- 精确/包含、区分/忽略大小写的文字查找。
- 非破坏式 OCR 框、中心点、文字和置信度标注。
- 默认 `dry_run=True` 的鼠标键盘控制器。
- 截图、OCR、控制和端到端四个命令行演示。
- ScreenAgent、Mind2Web、WebArena 三个公开数据源的确定性预处理与 manifest。
- 严格、不可变的任务计划、动作、决策、观察和执行结果 schema。
- 共享 `MultimodalPlanner` 接口的 fake、LangChain API 与本地 Qwen3-VL 实现。
- 默认 dry-run、逐动作安全确认、最大步数和重复动作停止的端到端 Agent 循环。
- 按 episode 隔离的训练/验证切分、Qwen 多模态 collator 和 4-bit QLoRA 训练入口。
- 可选 PEFT adapter 加载，以及基础提示词、grounded 提示词和 adapter 的固定合成评测。
- 只使用合成界面的模型 smoke test；普通测试不会下载或加载模型。
- Windows + Python 3.11 CI；普通 CI 不下载模型、不截图、不操作鼠标键盘。

架构图、模块边界和安全设计见[桌面感知与控制设计](docs/superpowers/specs/2026-08-17-desktop-perception-control-design.md)。第 5/6 周范围见[项目计划](docs/PROJECT_PLAN_WEEKS_5_6.md)，模型和训练环境分别见[模型 Provider 配置指南](docs/setup/model-provider-setup.md)与 [LoRA 训练指南](docs/setup/lora-training.md)。

## 安全规则

控制器和演示默认只记录动作，不会真实移动或点击。单次控制演示的真实点击必须同时满足：

1. 命令显式包含 `--execute`；
2. 端到端流程恰好找到一个候选；
3. 用户逐字输入 `CLICK THIS CANDIDATE`。

实时模式启用 PyAutoGUI fail-safe：把鼠标快速移到主显示器左上角可触发紧急中止。首次尝试前请先在默认 dry-run 模式核对坐标，尤其是在缩放比例不同的多显示器环境中。

完整 Agent 循环使用更严格的逐动作确认：命令必须包含 `--execute`，并在每次 proposed action 后逐字输入 `EXECUTE ACTION`。该确认不能复用于下一动作。

## 快速安装（PowerShell）

项目要求 Python `>=3.11,<3.12`。安装 uv 后，在仓库根目录运行：

```powershell
uv python install 3.11
uv sync --locked --group dev `
  --extra ocr `
  --extra agent `
  --extra datasets `
  --extra local-model `
  --extra training
$env:EASYOCR_MODULE_PATH = Join-Path $PWD "models\easyocr"
$env:HF_HOME = Join-Path $PWD ".cache\huggingface"
uv run python --version
```

只运行普通测试/CI 时可不安装 OCR 可选依赖：

```powershell
uv sync --locked --group dev
```

完整 Windows、CUDA、VS Code 和故障排查说明见 [docs/setup/windows-setup.md](docs/setup/windows-setup.md)。

## 运行演示

以下命令均在仓库根目录执行。

| 目标 | 命令 | 默认效果 |
|---|---|---|
| 内存截图 | `uv run python examples/capture_demo.py --monitor 1` | 打印尺寸和原点，不保存 |
| 保存截图 | `uv run python examples/capture_demo.py --monitor 1 --output artifacts/screen.png` | 仅写入显式路径 |
| 图片 OCR | `uv run python examples/ocr_demo.py artifacts/screen.png --gpu cuda` | 打印文字、置信度、框和中心点 |
| 模拟点击 | `uv run python examples/control_demo.py --x 500 --y 300` | 只记录 dry-run 动作 |
| 感知到控制 | `uv run python examples/perception_control_demo.py "保存" --monitor 1 --annotation artifacts/annotated.png` | 截图、OCR、查找、标注，并模拟唯一候选点击 |
| 数据适配器帮助 | `uv run python scripts/prepare_gui_datasets.py --help` | 列出三个数据源及参数，不下载数据 |
| fake Planner | `uv run python examples/model_smoke.py --provider fake --synthetic` | 对合成界面打印计划和动作，不加载模型 |
| 本地 Qwen Planner | `uv run python examples/model_smoke.py --provider qwen --synthetic` | 首次下载模型；之后在本机 GPU 生成计划和动作 |

查看所有选项：

```powershell
uv run python examples/capture_demo.py --help
uv run python examples/ocr_demo.py --help
uv run python examples/control_demo.py --help
uv run python examples/perception_control_demo.py --help
uv run python examples/model_smoke.py --help
uv run python scripts/prepare_gui_datasets.py --help
```

不建议直接启用真实点击。确需本地验证时，在确认目标坐标和桌面状态后添加 `--execute`，再按提示输入完整确认短语。

## Week 5：训练与固定评测

先使用已经标准化、带本地图像路径的公开数据构建 split。下面的路径是示例，请替换为你自己的 ScreenAgent 输出和图片根目录：

```powershell
uv run gui-agent training build `
  --input "screenagent=data/processed/screenagent/records.jsonl" `
  --image-root "screenagent=external/ScreenAgent/data/ScreenAgent" `
  --validation-ratio 0.10 `
  --seed 20260904 `
  --output data/training/week5

uv run gui-agent training check `
  --config configs/week5_qwen3vl_qlora.toml `
  --data data/training/week5 `
  --output artifacts/week5/lora-smoke

uv run gui-agent training train `
  --config configs/week5_qwen3vl_qlora.toml `
  --data data/training/week5 `
  --output artifacts/week5/qwen3vl-gui-lora

uv run gui-agent training evaluate `
  --cases configs/week5_eval_cases.json `
  --model Qwen/Qwen3-VL-4B-Instruct `
  --adapter artifacts/week5/qwen3vl-gui-lora `
  --prompt-profile week5-grounded `
  --output artifacts/week5/adapter.json
```

训练和评测需要本地 CUDA 模型，但不需要 Qwen API key。`data/`、`artifacts/`、checkpoint 和 `*.safetensors` 均被 Git 忽略。当前实验没有得到准确率提升，因此 Week 6 默认不加载该 adapter；完整指标见 [Week 5 QLoRA 对比报告](docs/test-reports/week5-lora-comparison-report.md)。

## 检查与测试

```powershell
uv lock --check
uv run ruff check .
uv run mypy src tests examples scripts
uv run pytest -m "not integration" --cov=gui_agent --cov-report=term-missing
```

2026-09-04 本机结果为 332 项普通测试全部通过、总覆盖率 86%，另有 2 项需显式启用的模型/GPU 集成测试。本轮真实 QLoRA 单步检查、正式训练和四组固定评测结果见 [Week 5 QLoRA 对比报告](docs/test-reports/week5-lora-comparison-report.md)；历史基线见 [Week 3 测试报告](docs/test-reports/week3-agent-foundation-report.md)与 [Week 2 测试报告](docs/test-reports/week2-test-report.md)。

## 目录

```text
src/gui_agent/
├─ types.py                  # 坐标、区域、截图和 OCR 数据类型
├─ agent/
│  ├─ types.py              # 任务计划、动作、决策、观察与结果 schema
│  ├─ prompts.py            # 脱敏后的计划/动作 prompt
│  ├─ planner.py            # fake 与 OpenAI-compatible Planner
│  └─ qwen.py               # 本地 Qwen Transformers Planner
├─ datasets/                # 三个公开数据源的适配器、schema 与 writer
├─ training/                # 确定性 split、QLoRA、adapter manifest 与固定评测
├─ perception/
│  ├─ capture.py            # MSS 截图
│  ├─ ocr.py                # OCR 协议与 EasyOCR 后端
│  └─ localization.py       # 查找与标注
└─ control/
   └─ controller.py         # dry-run 控制器与 PyAutoGUI 适配器

examples/                   # 感知/控制演示与合成模型 smoke test
scripts/                    # 数据集预处理 CLI
tests/                      # 默认隔离的单元测试；integration 需显式启用
docs/                       # 计划、设计、安装说明和测试报告
```

## 当前限制

- EasyOCR 首次运行需要联网下载检测与中英文识别权重；模型和缓存不会提交 Git。
- OCR 框是轴对齐矩形；复杂旋转文字、极小字体和拥挤界面仍可能误识别。
- EasyOCR 可能改变英文大小写，例如合成图中的 `Save` 曾识别为 `SaVe`；查找时可使用 `--ignore-case`。
- Windows 显示缩放、远程桌面、应用自绘控件和跨 DPI 显示器可能造成视觉坐标与输入坐标偏差，真实控制前必须 dry-run 验证。
- PyAutoGUI 的 `write` 对非 ASCII/输入法文本支持有限；当前中文能力主要用于 OCR，不保证中文键盘输入。
- 本地 Qwen 首次运行需要联网下载约 8.9 GB 权重；模型缓存不提交 Git，16 GB 显存的本机合成测试峰值分配约 9.0 GiB。
- OpenAI-compatible 路径只有显式允许后才发送图片；截图像素可能含敏感信息，prompt 文本脱敏不能替代发送前人工检查。
- 当前 Week 5 adapter 在固定小型评测上没有提高点击命中率，因此不作为默认运行配置；更复杂的失败恢复、结果验证和感知 benchmark 属于第 6 周。

截图、标注、PDF、模型权重、缓存、日志及 `.env` 均由 `.gitignore` 排除。不要把包含个人信息的桌面图像提交到仓库。

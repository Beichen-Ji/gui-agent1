# GUI Agent：桌面感知与安全控制基础

这是“大模型 GUI Agent”实习项目的第 1.6 节与第 2 周实现：用 Python 3.11、MSS、EasyOCR 和 PyAutoGUI 搭建桌面截图、文字识别、元素定位及安全控制基础。目前不接入大模型，也不包含第 1 周调研报告。

## 已实现

- 多显示器整屏/区域截图，统一输出 `uint8` BGR 图像和绝对屏幕坐标。
- 可替换的 OCR 接口，以及默认中英文 EasyOCR CUDA 后端。
- 精确/包含、区分/忽略大小写的文字查找。
- 非破坏式 OCR 框、中心点、文字和置信度标注。
- 默认 `dry_run=True` 的鼠标键盘控制器。
- 截图、OCR、控制和端到端四个命令行演示。
- Windows + Python 3.11 CI；普通 CI 不下载模型、不截图、不操作鼠标键盘。

架构图、模块边界和安全设计见[桌面感知与控制设计](docs/superpowers/specs/2026-08-17-desktop-perception-control-design.md)。详细执行计划见[实现计划](docs/superpowers/plans/2026-08-17-desktop-perception-control.md)。

## 安全规则

控制器和演示默认只记录动作，不会真实移动或点击。真实点击必须同时满足：

1. 命令显式包含 `--execute`；
2. 端到端流程恰好找到一个候选；
3. 用户逐字输入 `CLICK THIS CANDIDATE`。

实时模式启用 PyAutoGUI fail-safe：把鼠标快速移到主显示器左上角可触发紧急中止。首次尝试前请先在默认 dry-run 模式核对坐标，尤其是在缩放比例不同的多显示器环境中。

## 快速安装（PowerShell）

项目要求 Python `>=3.11,<3.12`。安装 uv 后，在仓库根目录运行：

```powershell
uv python install 3.11
uv sync --locked --group dev --extra ocr
$env:EASYOCR_MODULE_PATH = Join-Path $PWD "models\easyocr"
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

查看所有选项：

```powershell
uv run python examples/capture_demo.py --help
uv run python examples/ocr_demo.py --help
uv run python examples/control_demo.py --help
uv run python examples/perception_control_demo.py --help
```

不建议直接启用真实点击。确需本地验证时，在确认目标坐标和桌面状态后添加 `--execute`，再按提示输入完整确认短语。

## 检查与测试

```powershell
uv lock --check
uv run ruff check .
uv run mypy src tests examples
uv run pytest -m "not integration" --cov=gui_agent --cov-report=term-missing
```

2026-08-18 本机结果为 153 项测试全部通过、总代码覆盖率 93%。真实桌面只做了一次隐私安全的内存探针；详细环境与性能数据见 [Week 2 测试报告](docs/test-reports/week2-test-report.md)。

## 目录

```text
src/gui_agent/
├─ types.py                  # 坐标、区域、截图和 OCR 数据类型
├─ perception/
│  ├─ capture.py            # MSS 截图
│  ├─ ocr.py                # OCR 协议与 EasyOCR 后端
│  └─ localization.py       # 查找与标注
└─ control/
   └─ controller.py         # dry-run 控制器与 PyAutoGUI 适配器

examples/                   # 四个安全 CLI 演示
tests/                      # 不接触真实桌面/模型的单元测试
docs/                       # 计划、设计、安装说明和测试报告
```

## 当前限制

- EasyOCR 首次运行需要联网下载检测与中英文识别权重；模型和缓存不会提交 Git。
- OCR 框是轴对齐矩形；复杂旋转文字、极小字体和拥挤界面仍可能误识别。
- EasyOCR 可能改变英文大小写，例如合成图中的 `Save` 曾识别为 `SaVe`；查找时可使用 `--ignore-case`。
- Windows 显示缩放、远程桌面、应用自绘控件和跨 DPI 显示器可能造成视觉坐标与输入坐标偏差，真实控制前必须 dry-run 验证。
- PyAutoGUI 的 `write` 对非 ASCII/输入法文本支持有限；当前中文能力主要用于 OCR，不保证中文键盘输入。
- 当前控制决策只按文本候选工作；多模态模型、规划器和复杂 Agent 循环属于后续周次。

截图、标注、PDF、模型权重、缓存、日志及 `.env` 均由 `.gitignore` 排除。不要把包含个人信息的桌面图像提交到仓库。

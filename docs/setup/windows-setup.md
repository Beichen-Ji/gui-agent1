# Windows 环境配置与运行说明

本文对应项目计划 1.6 和第 2 周桌面感知/控制模块。命令以 64 位 Windows PowerShell 为例。

## 1. 前置条件

- Windows x64。
- Git。
- Python 3.11（推荐交给 uv 安装，不使用系统 Python 3.13）。
- OCR GPU 模式需要支持 CUDA 12.8 wheel 的 NVIDIA 驱动；CPU 模式不要求 NVIDIA GPU。

安装 uv 的官方 PowerShell 方式：

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

重新打开 PowerShell 后确认：

```powershell
uv --version
```

## 2. 创建项目环境

在仓库根目录运行：

```powershell
uv python install 3.11
uv sync --locked --group dev --extra ocr
uv run python --version
```

`uv sync` 会在项目内创建 `.venv`。`.python-version` 和 `pyproject.toml` 把解释器限制在 Python 3.11；不要用系统 Python 3.13 直接运行脚本。

在 VS Code 中选择：

```text
<仓库>\.venv\Scripts\python.exe
```

普通 CI 不需要真实 OCR，可只安装基础和开发依赖：

```powershell
uv sync --locked --group dev
```

## 3. EasyOCR 模型目录

为避免把权重写到用户目录，并确保模型位于已忽略路径，当前工作区使用：

```powershell
$env:EASYOCR_MODULE_PATH = Join-Path $PWD "models\easyocr"
```

首次创建 EasyOCR Reader 时会下载检测模型与中英文识别模型。`models/` 已被 `.gitignore` 排除。默认 Reader 关闭 EasyOCR 的 Unicode 进度条，以避免旧式 Windows CP1252 控制台在下载时发生编码错误。

## 4. 验证 CUDA

```powershell
uv run python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

2026-08-18 本机实测：

```text
Python 3.11.16
torch 2.11.0+cu128
PyTorch CUDA build 12.8
torch.cuda.is_available() True
NVIDIA GeForce RTX 5070 Ti
显存 15.92 GiB（nvidia-smi: 16303 MiB）
NVIDIA driver 595.95
```

核心包实测版本：

| 包 | 版本 |
|---|---:|
| uv | 0.12.5 |
| EasyOCR | 1.7.2 |
| MSS | 10.2.0 |
| NumPy | 2.4.6 |
| OpenCV headless | 4.14.0.94 |
| Pillow | 12.3.0 |
| PyAutoGUI | 0.9.54 |
| pynput | 1.8.2 |
| pytest | 9.1.1 |
| Ruff | 0.16.3 |
| mypy | 1.20.2 |

## 5. 截图与坐标

不保存的内存截图：

```powershell
uv run python examples/capture_demo.py --monitor 1
```

显式保存到已忽略的运行目录：

```powershell
uv run python examples/capture_demo.py --monitor 1 --output artifacts\screen.png
```

绝对区域截图支持负坐标：

```powershell
uv run python examples/capture_demo.py --region -1920 0 1920 1080 --output artifacts\left-monitor.png
```

MSS 的 `monitors[0]` 表示整个虚拟桌面，物理显示器编号从 1 开始。项目统一使用虚拟桌面的绝对像素坐标和半开边界；位于主屏左侧或上方的显示器可产生负坐标。

Windows 不同显示器的缩放比例可能影响视觉坐标与输入坐标的一致性。首次实时操作前应：

1. 使用截图和标注确认 OCR 框位置；
2. 使用默认 dry-run 检查动作中心点；
3. 从 100% 缩放或同缩放显示器开始验证；
4. 不在远程桌面切换、窗口动画或界面正在变化时执行点击。

## 6. OCR 与完整流程

对调用方提供的图片做 OCR：

```powershell
uv run python examples/ocr_demo.py artifacts\screen.png --gpu cuda
```

CPU 回退：

```powershell
uv run python examples/ocr_demo.py artifacts\screen.png --gpu cpu
```

截图、OCR、文本匹配、标注和模拟点击：

```powershell
uv run python examples/perception_control_demo.py "保存" `
  --monitor 1 `
  --ignore-case `
  --annotation artifacts\annotated.png
```

默认流程只记录一个 dry-run 点击。零候选或多个候选都不会选择目标。真实点击必须添加 `--execute`，恰好命中一个候选，并输入完整确认短语 `CLICK THIS CANDIDATE`。

## 7. PyAutoGUI 安全

- `DesktopController` 默认 `dry_run=True`。
- dry-run 不实例化 PyAutoGUI，也不会移动、点击、输入、滚动或拖拽。
- 实时适配器设置 `pyautogui.FAILSAFE=True`。
- 紧急情况下把鼠标快速移到主显示器左上角，PyAutoGUI 会抛出 fail-safe 异常并停止后续动作。
- 控制器会在调用后端前验证虚拟桌面边界、按钮、点击次数、按键和持续时间。

## 8. 常见问题

### `torch.cuda.is_available()` 为 `False`

确认安装了 `--extra ocr`、当前解释器来自 `.venv`、NVIDIA 驱动可见，并检查 `torch.__version__` 是否带 `+cu128`。不要用另一个全局 Python 运行项目。

### EasyOCR 下载失败

确认网络可以访问模型地址，检查代理/防火墙，并确认 `EASYOCR_MODULE_PATH` 指向可写目录。删除不完整的 `models/easyocr/model/temp.zip` 后可重试；不要删除整个仓库或用户目录。

### OCR 显存不足

使用 `--gpu cpu`。也可缩小截图区域；高分辨率整屏 OCR 的内存和耗时均更高。

### 坐标偏移

先确认显示器排列、缩放和截图原点。标注函数必须接收对应截图的 `origin`；端到端示例已自动传递。真实控制前使用 dry-run，并从稳定的小区域开始。

### 中文输入不工作

PyAutoGUI `write` 主要适合基础键盘字符。项目已支持中文 OCR，但未实现剪贴板/输入法驱动的可靠中文输入。

### Tkinter 报 `Can't find a usable init.tcl`

先检查当前解释器的 Tcl/Tk：

```powershell
uv run python -m tkinter
```

如果仍然报 `init.tcl` 错误，说明当前 Python 的 Tcl/Tk 安装不可用。`examples/gui_testbed.py`
只依赖标准库，可以临时改用任意 Tkinter 正常的本机 Python：

```powershell
& "C:\path\to\python.exe" examples\gui_testbed.py
```

长期使用时，建议安装带 Tcl/Tk 的 Python 3.11，并用它重新创建项目 `.venv`；其余项目命令仍应
遵守 `pyproject.toml` 的 Python 3.11 约束。

## 9. 开发门禁

```powershell
uv lock --check
uv run ruff check .
uv run mypy src tests examples
uv run pytest -m "not integration" --cov=gui_agent --cov-report=term-missing
```

普通门禁通过 fake 后端验证，不读取真实桌面、不下载模型，也不触发真实输入。

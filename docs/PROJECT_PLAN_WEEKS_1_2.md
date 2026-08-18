# 桌面 GUI 智能体项目：第 1-2 周实施计划

## 1. 项目目标

本计划仅覆盖项目大纲的第 1 周和第 2 周，目标是完成：

- GUI 智能体技术调研和开发环境搭建。
- 跨平台屏幕截图模块。
- OCR 屏幕文字识别模块。
- 鼠标键盘控制模块。
- 基于 OCR 的 UI 元素定位与边界框绘制。
- 完整的单元测试、测试报告和 GitHub 版本管理流程。

前两周不实现完整 Agent、多模态大模型调用、数据集处理或 LoRA 微调。

## 2. 已确认的项目配置

- GitHub 仓库：<https://github.com/Beichen-Ji/gui-agent1>
- Git 用户名：`Beichen-Ji`
- Git 邮箱：`bji7@uwo.ca`
- 本地默认分支：`main`
- Python 版本：`3.11`
- 开发系统：Windows
- GPU：NVIDIA GeForce RTX 5070 Ti（16 GB 显存）

## 3. 建议技术路线

- 屏幕截图：`mss`
- 图像数据：`Pillow` + `NumPy`
- 图像处理和标注：`OpenCV`
- OCR：优先 `PaddleOCR`，通过统一接口保留替换 `EasyOCR` 的能力
- 鼠标键盘控制：`PyAutoGUI` + `pynput`
- 测试：`pytest` + `pytest-cov`
- 代码质量：`ruff` + `mypy`
- 项目配置：`pyproject.toml`
- 持续集成：GitHub Actions，Windows + Python 3.11

## 4. 最终仓库结构

```text
gui-agent1/
├─ .github/
│  └─ workflows/
│     └─ ci.yml
├─ docs/
│  ├─ PROJECT_PLAN_WEEKS_1_2.md
│  ├─ research/
│  │  └─ gui-agent-technical-survey.md
│  ├─ setup/
│  │  └─ windows-setup.md
│  └─ test-reports/
│     └─ week2-test-report.md
├─ examples/
│  ├─ capture_demo.py
│  ├─ ocr_demo.py
│  └─ control_demo.py
├─ src/
│  └─ gui_agent/
│     ├─ __init__.py
│     ├─ types.py
│     ├─ capture.py
│     ├─ ocr.py
│     ├─ control.py
│     └─ localization.py
├─ tests/
│  ├─ fixtures/
│  ├─ test_capture.py
│  ├─ test_ocr.py
│  ├─ test_control.py
│  └─ test_localization.py
├─ .gitignore
├─ LICENSE
├─ pyproject.toml
└─ README.md
```

### 各文件职责

| 路径 | 职责 |
|---|---|
| `docs/research/gui-agent-technical-survey.md` | 第 1 周 GUI 智能体技术调研报告 |
| `docs/setup/windows-setup.md` | Python 3.11、CUDA 和项目依赖安装与验证文档 |
| `docs/test-reports/week2-test-report.md` | 第 2 周单元测试、性能和已知限制报告 |
| `src/gui_agent/types.py` | 截图、OCR 结果和边界框的公共数据类型 |
| `src/gui_agent/capture.py` | 屏幕和指定区域截图 |
| `src/gui_agent/ocr.py` | OCR 后端接口和默认 OCR 实现 |
| `src/gui_agent/control.py` | 安全的鼠标键盘操作和 `dry_run` |
| `src/gui_agent/localization.py` | 文本查找、坐标计算和边界框绘制 |
| `examples/` | 可人工执行的脱敏功能演示 |
| `tests/` | 单元测试、mock 测试和脱敏 OCR 图片 |

## 5. GitHub 版本管理 Todo

- [ ] 确认本地 `main` 分支和 GitHub 远程仓库状态。
- [ ] 创建 Python `.gitignore`，忽略以下内容：
  - `.venv/`
  - `__pycache__/`
  - `.pytest_cache/`
  - `.ruff_cache/`
  - `.mypy_cache/`
  - `.env`
  - 日志
  - 本地截图
  - OCR 缓存
  - 模型权重
- [ ] 创建 `README.md`，说明项目目标、进度、安装入口和安全边界。
- [ ] 创建 GitHub Milestone：`Week 1 - Research & Environment`。
- [ ] 创建 GitHub Milestone：`Week 2 - Perception & Control`。
- [ ] 为本文档中的每个一级开发任务创建 GitHub Issue。
- [ ] 创建第 1 周分支：`week1/research-and-environment`。
- [ ] 第 1 周完成后通过 Pull Request 合并到 `main`。
- [ ] 创建标签：`v0.1.0-week1`。
- [ ] 创建第 2 周分支：`week2/perception-and-control`。
- [ ] 第 2 周完成后通过 Pull Request 合并到 `main`。
- [ ] 创建标签：`v0.2.0-week2`。
- [ ] 为 `main` 开启分支保护，要求 CI 成功后才能合并。
- [ ] 提交信息使用 Conventional Commits：
  - `chore: ...`
  - `docs: ...`
  - `feat: ...`
  - `test: ...`
  - `fix: ...`

个人项目使用 `main + 短期功能分支 + Pull Request` 即可，暂不创建 `develop` 分支。

---

# 第 1 周：行业技术调研与开发环境搭建

## 第 1 周交付物

- 《GUI 智能体技术调研报告》
- Windows + Python 3.11 开发环境配置文档
- 可复现的项目依赖配置
- 可运行的 GitHub Actions 基础 CI

## 1.1 建立项目基础结构

- [ ] 创建 `pyproject.toml`。
- [ ] 在 `pyproject.toml` 中指定 `requires-python = ">=3.11,<3.12"`。
- [ ] 创建 `src/gui_agent/` 包结构。
- [ ] 创建 `tests/`、`examples/` 和 `docs/` 目录。
- [ ] 创建 `README.md`。
- [ ] 选择并添加开源许可证。
- [ ] 完成第一次项目初始化提交。

建议提交：

```text
chore: initialize Python project structure
```

## 1.2 调研 GUI 智能体基础概念

- [ ] 定义 GUI Agent。
- [ ] 说明多模态大模型在 GUI Agent 中的作用。
- [ ] 说明 GUI Agent 典型执行闭环：

```text
用户指令
→ 屏幕截图
→ 视觉/文字理解
→ 任务规划
→ 动作生成
→ 鼠标键盘执行
→ 获取新截图
→ 验证执行结果
```

- [ ] 区分并解释：
  - OCR
  - UI 元素检测
  - Visual Grounding
  - 任务规划
  - Tool Use
  - Action Execution
  - 执行反馈
- [ ] 说明前两周只实现感知和控制基础设施。

建议提交：

```text
docs: define GUI agent scope and terminology
```

## 1.3 深入调研代表性项目

调研对象：

- [ ] UI-TARS
- [ ] Claude Computer Use
- [ ] ScreenAgent

每个项目都需记录：

- [ ] 项目目标和应用场景。
- [ ] 屏幕输入形式。
- [ ] 支持的动作类型。
- [ ] UI 元素或坐标定位方式。
- [ ] 任务规划方式。
- [ ] 是否使用记忆或历史上下文。
- [ ] 模型训练方式和数据来源。
- [ ] 是否开源、许可证和可复现性。
- [ ] 主要优点。
- [ ] 主要限制。
- [ ] 对本项目可借鉴的设计。

调研报告需包含横向对比表：

| 维度 | UI-TARS | Claude Computer Use | ScreenAgent |
|---|---|---|---|
| 是否开源 | | | |
| 屏幕输入 | | | |
| 动作空间 | | | |
| UI 元素定位 | | | |
| 任务规划 | | | |
| 反馈机制 | | | |
| 部署要求 | | | |
| 主要优势 | | | |
| 主要限制 | | | |

- [ ] 所有技术结论附原始论文、官方文档或 GitHub 链接。
- [ ] 记录资料访问日期。
- [ ] 用自己的语言总结，不复制大段原文。

建议提交：

```text
docs: compare leading GUI agent systems
```

## 1.4 设计本项目技术架构

- [ ] 绘制项目架构图。
- [ ] 定义 Screen Capture 模块的输入和输出。
- [ ] 定义 OCR 模块的输入和输出。
- [ ] 定义 UI Localization 模块的输入和输出。
- [ ] 定义 Mouse/Keyboard Controller 模块的输入和输出。
- [ ] 在架构图中标出后续 Planner 和 Multimodal Model 的位置。
- [ ] 明确截图坐标、OCR 边界框和鼠标坐标使用统一坐标系。
- [ ] 说明为什么第 2 周的模块不应依赖 Agent 框架或大模型。
- [ ] 说明如何替换 OCR 后端而不影响上层逻辑。

## 1.5 分析关键技术挑战

报告至少覆盖：

- [ ] Windows DPI 缩放导致的坐标偏移。
- [ ] 多显示器和不同屏幕分辨率。
- [ ] OCR 对小字体、中英文和深色主题的识别。
- [ ] 图标、画布等无文字元素无法仅靠 OCR 定位。
- [ ] 截图、模型推理和动作执行延迟。
- [ ] 模型可能产生错误点击或不可逆操作。
- [ ] 页面变化后历史坐标失效。
- [ ] 任务成功状态的判断。
- [ ] 错误检测、重试和回滚。
- [ ] 截图可能含有个人或敏感信息。
- [ ] 自动化程序的权限和安全边界。

## 1.6 搭建 Python 3.11 开发环境

- [ ] 创建独立 Python 3.11 虚拟环境。
- [ ] 确认项目不使用系统 Python 3.13。
- [ ] 安装基础开发工具：
  - `pytest`
  - `pytest-cov`
  - `ruff`
  - `mypy`
- [ ] 安装第 2 周基础依赖：
  - `mss`
  - `Pillow`
  - `numpy`
  - `opencv-python`
  - `pyautogui`
  - `pynput`
  - `paddleocr` 或 `easyocr`
- [ ] 按 PyTorch 官方安装器选择与本机匹配的 CUDA 版本。
- [ ] 验证 Python 版本。
- [ ] 验证所有基础依赖能够导入。
- [ ] 验证 `torch.cuda.is_available()` 返回 `True`。
- [ ] 打印 GPU 名称和可用显存。
- [ ] 使用 `mss` 获取一张测试截图。
- [ ] 将安装命令、验证命令和实际输出写入 `docs/setup/windows-setup.md`。
- [ ] 记录 CUDA 不可用、OCR 模型下载、DPI 缩放和 PyAutoGUI fail-safe 等常见问题。

建议提交：

```text
chore: configure Python 3.11 development environment
docs: add Windows environment setup guide
```

## 1.7 建立基础 CI

- [ ] 创建 `.github/workflows/ci.yml`。
- [ ] CI 使用 Windows 和 Python 3.11。
- [ ] CI 执行 `ruff check .`。
- [ ] CI 执行不包含真实桌面操作的 `pytest`。
- [ ] CI 不移动鼠标、不输入文字、不点击 UI。
- [ ] OCR 模型下载和真实桌面测试标记为 integration test，不在普通 CI 中执行。

建议提交：

```text
ci: add Windows lint and unit test workflow
```

## 1.8 第 1 周验收标准

- [ ] 调研报告比较 UI-TARS、Claude Computer Use 和 ScreenAgent。
- [ ] 报告包含技术架构图。
- [ ] 报告包含关键挑战和安全风险。
- [ ] 所有资料有可访问的来源。
- [ ] 可以根据文档从零创建 Python 3.11 环境。
- [ ] Python 能够识别 RTX 5070 Ti。
- [ ] README 包含安装和文档入口。
- [ ] GitHub Actions 成功。
- [ ] Week 1 Pull Request 已审查并合并。
- [ ] 已创建 `v0.1.0-week1` 标签。

---

# 第 2 周：桌面感知与控制模块开发

## 第 2 周交付物

- 屏幕截图模块。
- OCR 文字与边界框识别模块。
- 鼠标键盘控制模块。
- UI 元素文本定位和边界框绘制模块。
- 完整单元测试。
- 第 2 周测试报告。

## 2.1 定义公共数据结构

- [ ] 定义截图结果类型，包含：
  - 图像数据
  - 宽度和高度
  - 显示器编号
  - 截图时间
  - 截图区域原点
- [ ] 定义 OCR 检测结果类型，包含：
  - 文本
  - 置信度
  - 边界框
  - 中心坐标
- [ ] 统一使用屏幕像素坐标。
- [ ] 明确 RGB 和 BGR 转换位置。
- [ ] 为公共数据类型编写单元测试。

建议提交：

```text
feat: define screen and OCR result types
```

## 2.2 实现屏幕截图模块

- [ ] 使用 `mss` 实现屏幕截图。
- [ ] 支持选择显示器。
- [ ] 支持完整屏幕截图。
- [ ] 支持指定区域截图。
- [ ] 不硬编码屏幕尺寸或分辨率。
- [ ] 将截图转换为 NumPy/OpenCV 可处理格式。
- [ ] 支持主动保存调试截图，默认不保存。
- [ ] 对无效显示器编号给出明确错误。
- [ ] 对无效截图区域给出明确错误。
- [ ] 创建 `examples/capture_demo.py`。
- [ ] 测试截图图像非空。
- [ ] 测试图像尺寸正确。
- [ ] 测试区域截图尺寸正确。
- [ ] 测试不同分辨率下无硬编码错误。
- [ ] 测试无效参数能够正确报错。

建议提交：

```text
feat: add cross-platform screen capture
test: cover screen capture behavior
```

## 2.3 实现 OCR 模块

- [ ] 定义统一 `OCRBackend` 接口。
- [ ] 实现一个可工作的 OCR 后端。
- [ ] 支持中文和英文。
- [ ] 输入 NumPy 图像，输出统一 OCR 检测结果列表。
- [ ] 支持最低置信度过滤。
- [ ] 无识别结果时返回空列表。
- [ ] 对 OCR 后端未安装给出明确错误。
- [ ] 对模型加载失败给出明确错误。
- [ ] 对非法图片格式给出明确错误。
- [ ] 准备不含个人信息的 OCR 测试图片：
  - 英文文本
  - 中文文本
  - 小字体
  - 深色背景
  - 多行文本
  - 高 DPI 截图
- [ ] 创建 `examples/ocr_demo.py`。
- [ ] 记录 OCR 模型首次加载耗时。
- [ ] 记录单张图片识别耗时。

建议提交：

```text
feat: add pluggable OCR backend
test: add OCR fixtures and result validation
```

## 2.4 实现鼠标键盘控制模块

- [ ] 实现鼠标移动。
- [ ] 实现鼠标单击。
- [ ] 实现鼠标双击。
- [ ] 实现鼠标右键点击。
- [ ] 实现键盘文本输入。
- [ ] 实现键盘快捷键。
- [ ] 实现垂直滚动。
- [ ] 实现拖拽。
- [ ] 支持动作间隔，避免操作过快。
- [ ] 开启 PyAutoGUI fail-safe。
- [ ] 实现 `dry_run` 模式，只记录动作而不执行。
- [ ] 校验坐标是否越界。
- [ ] 校验按键名称是否合法。
- [ ] 校验持续时间是否合法。
- [ ] 在单元测试中 mock 底层桌面 API。
- [ ] 验证测试不会实际移动鼠标或输入文本。
- [ ] 创建 `examples/control_demo.py`，并默认启用 `dry_run`。

建议提交：

```text
feat: add safe mouse and keyboard controller
test: mock desktop control operations
```

## 2.5 实现 UI 元素定位与边界框绘制

- [ ] 支持根据 OCR 结果精确匹配文本。
- [ ] 支持忽略英文大小写匹配。
- [ ] 支持包含关键词匹配。
- [ ] 匹配到多个结果时返回候选列表，不自动选择并点击。
- [ ] 计算 OCR 边界框中心点。
- [ ] 在截图上绘制边界框。
- [ ] 在截图上绘制文本和置信度。
- [ ] 在截图上绘制中心点。
- [ ] 使用不同颜色标识高、低置信度。
- [ ] 绘制结果不修改原始图像对象。
- [ ] 测试文本匹配、多候选、坐标计算和图像绘制。

建议提交：

```text
feat: add OCR-based UI localization and annotation
```

## 2.6 创建安全的端到端演示

演示只串联第 2 周模块，不接入大模型。

- [ ] 截取当前屏幕。
- [ ] 调用 OCR 识别文本。
- [ ] 搜索用户指定的文本。
- [ ] 绘制所有匹配候选的边界框。
- [ ] 打印预计点击坐标。
- [ ] 默认使用 `dry_run`。
- [ ] 只有在用户明确确认后才允许真实点击。
- [ ] 只在本地测试窗口或不含敏感信息的应用中演示。
- [ ] 确认个人截图不会被提交到 GitHub。

## 2.7 编写第 2 周测试报告

`docs/test-reports/week2-test-report.md` 需包含：

- [ ] 测试操作系统和硬件。
- [ ] Python、CUDA、OCR 和主要依赖版本。
- [ ] 单元测试数量和通过情况。
- [ ] 完整测试命令。
- [ ] 代码覆盖率。
- [ ] 屏幕截图功能测试结果。
- [ ] OCR 测试样例、识别结果和已知限制。
- [ ] 鼠标键盘 mock 测试结果。
- [ ] DPI、分辨率和坐标一致性测试。
- [ ] 平均截图时间。
- [ ] OCR 模型首次加载和平均识别时间。
- [ ] 未解决问题。
- [ ] 对第 3 周的建议。

建议提交：

```text
docs: add week two test report
```

## 2.8 第 2 周验收标准

- [ ] `ruff check .` 通过。
- [ ] 普通 `pytest` 全部通过。
- [ ] GitHub Actions 通过。
- [ ] 截图模块不依赖固定分辨率。
- [ ] OCR 能够输出文本、置信度和边界框。
- [ ] 程序能够根据文本找到候选坐标。
- [ ] 程序能够正确绘制边界框。
- [ ] 控制模块支持点击、输入、滚动和拖拽。
- [ ] 控制模块具备 fail-safe 和 `dry_run`。
- [ ] 单元测试不操作真实鼠标键盘。
- [ ] 仓库不包含敏感截图、模型权重或 `.env`。
- [ ] 测试报告包含环境、结果、耗时和已知限制。
- [ ] Week 2 Pull Request 已审查并合并。
- [ ] 已创建 `v0.2.0-week2` 标签。

---

## 6. 建议提交顺序

1. `chore: initialize Python project structure`
2. `docs: define GUI agent scope and terminology`
3. `docs: compare leading GUI agent systems`
4. `chore: configure Python 3.11 development environment`
5. `docs: add Windows environment setup guide`
6. `ci: add Windows lint and unit test workflow`
7. `feat: define screen and OCR result types`
8. `feat: add cross-platform screen capture`
9. `test: cover screen capture behavior`
10. `feat: add pluggable OCR backend`
11. `test: add OCR fixtures and result validation`
12. `feat: add safe mouse and keyboard controller`
13. `test: mock desktop control operations`
14. `feat: add OCR-based UI localization and annotation`
15. `docs: add week two test report`

## 7. 安全与合规要求

- [ ] 仅在本人拥有或获得明确授权的电脑和应用上执行自动化。
- [ ] 不用于未经授权的系统操作。
- [ ] 不将用户桌面截图或敏感信息提交到 GitHub。
- [ ] 所有演示优先使用 `dry_run`。
- [ ] 保留 PyAutoGUI fail-safe 和明确的紧急停止方式。
- [ ] 真实点击或文本输入前需要用户明确确认。
- [ ] 检查所有开源依赖和参考项目的许可证。

## 8. 前两周不包含的工作

- LangChain 或 LlamaIndex Agent 框架。
- Qwen-VL、GLM-4V 或 Llama Vision 部署。
- 大模型 API 调用。
- ScreenAgent、WebArena 或 Mind2Web 数据集处理。
- 用户任务自动拆解和规划。
- 完整端到端 GUI Agent。
- LoRA 微调。
- 复杂 PyQt 用户界面。

这些内容应从第 3 周开始另行设计和实现。

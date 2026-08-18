# Week 2 桌面感知与控制测试报告

## 1. 范围与日期

- 测试日期：2026-08-18（America/Toronto）。
- 分支：`agent/week2-perception-control`。
- 范围：项目计划 1.6 环境配置，以及第 2 周截图、OCR、定位、标注、控制和安全演示。
- 不包含：调研报告、大模型调用、Agent 规划器、训练/微调、真实鼠标压力测试。

## 2. 本机环境

| 项目 | 实测值 |
|---|---|
| Windows 注册表 | Home，DisplayVersion 25H2，build 26200.8875 |
| 系统架构 | x86_64 |
| Python | 3.11.16 |
| uv | 0.12.5 |
| GPU | NVIDIA GeForce RTX 5070 Ti |
| 驱动 | 595.95 |
| 显存 | 15.92 GiB（16303 MiB） |
| PyTorch | 2.11.0+cu128 |
| PyTorch CUDA build | 12.8 |
| `torch.cuda.is_available()` | `True` |
| EasyOCR | 1.7.2 |
| MSS | 10.2.0 |
| NumPy | 2.4.6 |
| OpenCV headless | 4.14.0.94 |
| Pillow | 12.3.0 |
| PyAutoGUI / pynput | 0.9.54 / 1.8.2 |

注册表的 `ProductName` 仍返回 `Windows 10 Home`，而 DisplayVersion/build 为 25H2/26200.8875；报告保留原始字段，不据此推断市场名称。CIM 内存查询因当前权限被拒绝，未将系统内存写入结果。

## 3. 普通测试与覆盖率

执行：

```powershell
uv run pytest -m "not integration" --cov=gui_agent --cov-report=term-missing
```

结果：154 项全部通过；普通测试没有截图、模型下载或真实输入。

| 测试文件 | 数量 | 主要覆盖 |
|---|---:|---|
| `test_types.py` | 29 | 坐标、区域、截图/OCR 类型校验 |
| `test_capture.py` | 12 | MSS 参数、负原点、BGRA→BGR、保存和异常 |
| `test_ocr.py` | 27 | 结果标准化、空文字候选过滤、阈值、GPU 选择、错误与 Windows 下载编码回归 |
| `test_localization.py` | 19 | 文本匹配、颜色、字体、原图不变、原点换算 |
| `test_control.py` | 47 | 所有动作、dry-run、实时 fake、fail-safe、边界和校验 |
| `test_examples.py` | 20 | 确认短语、无输出默认、CLI 帮助和端到端安全门 |

覆盖率：

| 模块 | 覆盖率 |
|---|---:|
| `gui_agent.types` | 100% |
| `gui_agent.control.controller` | 97% |
| `gui_agent.perception.capture` | 89% |
| `gui_agent.perception.localization` | 97% |
| `gui_agent.perception.ocr` | 86% |
| 总计 | 93%（451 statements，30 missed） |

未覆盖行主要是本机依赖缺失、MSS/PyAutoGUI 默认适配器的真实导入路径和少数防御性异常；这些路径由隐私安全集成探针或人工环境检查补充，不在普通 CI 触发。

## 4. 隐私安全集成探针

探针约束：

- 真实屏幕仅截取到内存，用于类型、尺寸和耗时断言。
- 不保存真实截图，不打印像素，不对真实截图运行 OCR。
- OCR 输入是 Pillow 在内存中生成的 1200×640 合成图。
- EasyOCR 权重写入已忽略的 `models/easyocr`。
- 未执行任何真实鼠标、键盘、滚轮或拖拽操作。

### 4.1 MSS 截图

物理显示器 1 的实测帧尺寸为 2560×1440。连续 20 次结果：

| 指标 | 耗时 |
|---|---:|
| 平均 | 37.185 ms |
| 中位数 | 36.904 ms |
| 最小 | 34.738 ms |
| 最大 | 44.055 ms |

每帧均满足非空、`numpy.uint8`、三通道 BGR。结果只代表当前机器、显示器和当时桌面状态，不作为跨机器性能保证。

### 4.2 EasyOCR CUDA

合成图包含英文按钮、简体中文、两行文字、18 px 小字体和深色背景白字。使用 `("ch_sim", "en")`、GPU、最低置信度 0.1。

| 指标 | 结果 |
|---|---:|
| 缓存模型后的首次 Reader 初始化 + 推理 | 2.562 s |
| 后续推理次数 | 3 |
| 后续平均 | 0.132 s |
| 后续中位数 | 0.132 s |
| 后续最小 / 最大 | 0.130 s / 0.134 s |
| 检测数量 | 9 |

识别片段为：`SaVe`、`Settings`、`取消`、`确定`、`Multi-line`、`desktop test`、`第二行中文文本`、`Small text 18px 小字体`、`Dark mode 深色模式`。英文 `Save` 出现大小写漂移，因此实际查找建议使用忽略大小写模式。

最初的冷模型下载已经成功，但该次探针在最终向 CP1252 控制台输出中文 JSON 时失败，没有保留可信的冷下载总耗时；本报告不填造该值。该过程同时发现 EasyOCR 默认 Unicode 进度条可导致 Windows 控制台编码失败，现已通过默认 `verbose=False` 和回归测试修复。上表“首次”明确指模型已缓存后，本进程首次创建 Reader 并推理。

## 5. 坐标与控制结论

- 公共点和框统一为虚拟桌面物理像素绝对坐标；负坐标有效。
- 截图结果携带 `origin`，OCR 将局部框转换为绝对框，标注时再按图像原点换回局部坐标。
- 区域与桌面边界采用半开区间，右/下边界不允许点击。
- 47 项控制器测试全部使用 fake；确认 dry-run 不构造 PyAutoGUI，实时 fake 的参数映射正确。
- PyAutoGUI 适配器启用 `FAILSAFE=True`，但本次没有执行真实控制，避免影响用户桌面。

## 6. CI

GitHub Actions 使用 Windows runner 和 Python 3.11，执行锁文件同步、Ruff、mypy 和普通 pytest/coverage。CI 不安装 `ocr` extra，因此不会下载 EasyOCR/PyTorch 模型；OCR、MSS 和输入动作都由 fake 覆盖。

本机此前另建全新的 CI 虚拟环境验证了这一配置：只安装基础/开发组共 29 个包，确认 `easyocr` 和 `torch` 均不存在；当时 Ruff、mypy 以及 153 项测试/93% 覆盖率全部通过。

## 7. 已知限制

- 没有对不同缩放比例的真实多显示器执行点击；只验证了负坐标和边界换算。
- 没有保存或人工检查真实桌面截图，以保护隐私。
- 冷模型下载耗时未保留；只报告缓存模型后的首次初始化与推理。
- OCR 在英文大小写、极小字体、旋转/拥挤文本上可能漂移。
- PyAutoGUI 不保证可靠输入中文，当前没有剪贴板或输入法适配器。
- 没有在远程桌面、锁屏、UAC 安全桌面或无桌面 CI 会话中做集成验证。

## 8. Week 3 建议

1. 增加截图缩放/DPI 标定与窗口级坐标测试。
2. 把 OCR、图标检测和多模态模型结果统一成候选元素协议。
3. 引入任务状态、观察—行动循环和可回放轨迹，同时保持控制器安全门。
4. 对失败 OCR、重复候选和界面变化增加重试/重新观察策略。
5. 在专用测试桌面中加入显式启用的真实输入集成测试，避免影响日常桌面。

# Week 6 鲁棒 GUI Agent v2 测试报告

- 日期：2026-09-05
- 分支：`codex/week6-robust-agent-v2`
- 基线：Week 5 合并提交 `8314fdce817342548fe4b3c7eedb341b5b15f72a`
- 本机：Windows 10 build 26200、Python 3.11.16、NVIDIA GeForce RTX 5070 Ti
- 关键版本：PyTorch 2.11.0+cu128、EasyOCR 1.7.2、NumPy 2.4.6、OpenCV 4.14.0、Pillow 12.3.0

## 1. 结论

Week 6 的 8 个固定故障注入场景全部满足预期：5 个可恢复场景在限制内成功，policy 拒绝和用户拒绝确认均为零 retry，恢复额度耗尽时以 `retry_exhausted` 受控失败。每个场景最后都有 `run_finished` 事件。

本轮自动测试只使用合成帧、fake planner/executor、虚拟 clock 和本地状态机，不读取真实桌面、不联网、不加载 Qwen，也不操作鼠标键盘。真实桌面 live 测试没有在无人确认时自动执行，仍需操作者在 testbed 前逐动作确认；因此本报告不把 live 测试标记为已通过。

## 2. 运行配置

- 自动鲁棒性模型：确定性 `FakePlanner`，无模型下载和网络访问。
- 真实运行默认模型：`Qwen/Qwen3-VL-4B-Instruct`；本轮自动场景未加载。
- Adapter：不加载。Week 5 默认配置和唯一预定义变体均未超过 baseline，详见 [Week 5 QLoRA 对比报告](week5-lora-comparison-report.md)。
- OCR：自动场景使用合成 observation，不调用 OCR；真实 CLI 默认使用 `balanced` profile。
- 恢复边界：每步最多 2 次 retry、退避 0.5/1.0 秒、最多 1 次 replan、禁止原样重放动作。
- 事件：目标仅记录 SHA-256；OCR 仅记录数量和摘要哈希；键入动作仅记录字符数。

## 3. 固定故障注入结果

表中的“恢复耗时”是注入 virtual clock 的退避时间总和，不是容易受机器负载影响的墙钟时间。“确认次数”由测试输入函数模拟，不代表真实人工操作。

| 场景 | 结果 | 动作数 | retry | replan | 恢复耗时 | 最终 reason code | 确认次数 |
|---|---:|---:|---:|---:|---:|---|---:|
| 首次 OCR 瞬态错误 | 成功 | 2 | 1 | 0 | 0.5 s | `null` | 0 |
| 点击后界面无变化，改用不同动作 | 成功 | 3 | 1 | 0 | 0.5 s | `null` | 0 |
| 结果延迟出现，改用 wait | 成功 | 3 | 1 | 0 | 0.5 s | `null` | 0 |
| 错误 tab，触发一次 replan | 成功 | 3 | 0 | 1 | 0.0 s | `null` | 0 |
| executor 首次瞬态异常 | 成功 | 3 | 1 | 0 | 0.5 s | `null` | 0 |
| policy 拒绝越界动作 | 预期受控失败 | 0 | 0 | 0 | 0.0 s | `policy_denied` | 0 |
| 用户拒绝 live 确认 | 预期受控失败 | 0 | 0 | 0 | 0.0 s | `confirmation_rejected` | 1（模拟） |
| retry/replan 全部耗尽 | 预期受控失败 | 4 | 2 | 1 | 1.0 s | `retry_exhausted` | 0 |

自动命令与结果：

```powershell
uv run --no-sync pytest tests/integration/test_week6_robustness.py `
  -m integration -v `
  --basetemp artifacts/pytest-week6-integration
```

结果：`8 passed in 0.11s`。

## 4. 感知 benchmark

固定合成 UI 共 2 个 case，warmup 2 次，每个 profile 正式运行 5 次。原始 JSON 写入忽略目录 `artifacts/perception-benchmark/week6.json`，未提交 Git。

| Profile | Precision | Recall | F1 | 冷帧 median | 冷帧 p95 | 缓存 median | median 降幅 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `fast` | 0.2857 | 0.4000 | 0.3333 | 46.34 ms | 50.31 ms | 0.242 ms | 99.48% |
| `balanced` | 0.5000 | 0.6000 | 0.5455 | 48.93 ms | 51.47 ms | 0.232 ms | 99.53% |
| `accurate` | 0.2857 | 0.4000 | 0.3333 | 57.47 ms | 63.52 ms | 0.227 ms | 99.61% |

`balanced` 的 F1 高于 `fast`，冷帧 median 仅增加约 5.59%，相同帧缓存 median 降幅超过 50%，所以选为默认 profile。样本量很小，这些数值只用于本项目回归与 profile 选择，不代表通用 OCR 基准。

## 5. Dry-run 与真实桌面测试状态

已用 `delayed-search` 配置完成 fake dry-run 事件链验证。由于 dry-run 不改变合成界面，成功文本不会凭空出现，完成验证拒绝了无证据的成功声明；过程没有真实输入动作。该行为符合“dry-run 用于核对建议，不等于任务真实完成”的安全语义。

真实桌面测试待操作者在场完成：

1. 第一个终端运行 `uv run --no-sync python examples/gui_testbed.py --fault-profile delayed`。
2. 第二个终端运行 `uv run --no-sync gui-agent run --task-id delayed-search --provider qwen --ocr-profile balanced --max-steps 12 --max-retries-per-step 2 --max-replans 1 --run-dir artifacts/agent-runs/week6-delayed-search`。
3. 先检查 dry-run 的动作、坐标和事件。
4. 只有确认无误后才显式添加 `--execute`，并对每个动作逐字输入 `EXECUTE ACTION`。
5. 记录最终 summary、retry/replan 数和人工确认次数；不要提交截图、完整 OCR 或输入全文。

## 6. 全量自动验收

```text
uv lock --check                                      PASS
uv run --no-sync ruff check .                       PASS
uv run --no-sync mypy src tests examples scripts    PASS (89 source files)
uv run --no-sync pytest -m "not integration" ...    PASS (404 passed, 10 deselected, coverage 87%)
Week 6 failure-injection integration                 PASS (8 passed)
```

普通测试和 Week 6 integration 均不访问真实桌面、网络或模型。Git 差异、忽略路径和跟踪文件检查在提交前再次执行。

## 7. 已知限制

- 自动故障场景验证的是 Agent 控制流，不等同于真实 Qwen、EasyOCR 或 Windows 输入链路测试。
- OCR benchmark 只有 2 个合成 case，不能推断真实应用总体准确率。
- 真实桌面上的 DPI、主题、窗口遮挡和坐标偏差仍需人工验证。
- 恢复策略有意受限；超过 retry/replan 额度后明确失败，不会无限尝试。
- Week 5 adapter 没有改善固定小型评测，因此 Week 6 默认不加载；本报告不声称微调提升。

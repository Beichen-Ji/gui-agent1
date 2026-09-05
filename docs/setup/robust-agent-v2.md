# Week 6 鲁棒 GUI Agent v2 使用说明

Week 6 在原有“观察—规划—授权—执行”循环后加入计划进度、结果验证、限次重试、一次受控重规划、OCR profile、精确帧缓存和实时事件流。默认仍是 dry-run；只有显式添加 `--execute`，且逐动作输入 `EXECUTE ACTION`，程序才会操作桌面。

## 环境准备

在 `week6-robust-agent-v2` worktree 的 PowerShell 中运行：

```powershell
Remove-Item Env:VIRTUAL_ENV -ErrorAction SilentlyContinue
uv sync --locked --group dev --extra ocr --extra agent --extra local-model
$env:EASYOCR_MODULE_PATH = Join-Path $PWD "models\easyocr"
```

本地 Qwen 不需要 API key。第一次下载公开模型或 EasyOCR 权重时需要联网。

## 安全 dry-run

```powershell
uv run --no-sync gui-agent run `
  --task-id search-content `
  --provider qwen `
  --ocr-profile balanced `
  --max-steps 12 `
  --max-retries-per-step 2 `
  --max-replans 1 `
  --run-dir artifacts/agent-runs/week6-search
```

如果要使用 Week 5 adapter，再显式添加：

```powershell
--adapter artifacts/week5/qwen3vl-gui-lora --prompt-profile week5-grounded
```

Week 5 实测 adapter 未超过基线，因此 Week 6 默认不加载 adapter。

## 输出说明

- 最终兼容 JSON 写到 stdout。
- 实时状态写到 stderr，不影响脚本解析最终 JSON。
- 指定 `--run-dir` 后，会生成 `events.jsonl` 和 `run-summary.json`。
- `--trace-dir` 暂时保留为 `--run-dir` 的弃用别名；二者不能同时使用。
- 事件日志不保存截图、完整 goal、完整 OCR 或键入文本。goal 只写 SHA-256，OCR 只写数量和摘要哈希，键入动作只写字符数。

## 恢复边界

- 每步骤默认最多重试 2 次，退避为 0.5 秒、1.0 秒。
- 重试会重新观察、重新规划、重新授权；不会直接重放上次动作。
- 相同动作连续出现时立即停止。
- policy 拒绝、用户拒绝确认、非法动作和 planner 非法输出不重试。
- 重试耗尽后最多重规划 1 次；之后受控失败。
- `finish(success=True)` 只有在计划结束且已有验证证据时才会成功。

## OCR profile 与 benchmark

`fast` 接近 Week 4 基线，`balanced` 是默认档，`accurate` 使用更重的 CLAHE、缩放和 beam search。本机合成 benchmark 显示 `balanced` 在冷帧延迟增加小于 10% 的情况下提高 F1。

重新运行 benchmark：

```powershell
uv run --no-sync python scripts/benchmark_ocr.py `
  --manifest tests/fixtures/ocr_benchmark/manifest.json `
  --profiles fast balanced accurate `
  --warmup 2 --runs 5 `
  --output artifacts/perception-benchmark/week6.json
```

`artifacts/`、模型、截图和详细日志都不会提交到 Git。

# Week 7 真实桌面小样本子集

状态：待操作者在场执行（2026-09-10 已完成任务选择、环境准备和首轮 dry-run 诊断）

## 目的与边界

本记录用 3 项本地 Tk testbed 任务观察 sim-to-real 差异。它不是自动评测的一部分，不提交截图、完整 OCR 文本或输入全文，也不能从 3 个样本外推总体能力。

Week 7 模拟套件与真实桌面 CLI 使用不同的任务 ID，所以下表记录语义等价映射。比较基准固定为 1920×1080、OCR、`week4-baseline`、无 adapter 的自动结果。

| 难度 | 模拟任务 | 真实桌面 CLI 任务 | testbed fault profile | 模拟结果 |
|---|---|---|---|---|
| easy | `browser-open-app` | `open-browser` | `none` | stopped，2 步，1 次 retry，0 次 replan |
| medium | `browser-search-transient` | `search-content` | `transient` | stopped / `repeated_action`，4 步，4 次 retry，0 次 replan |
| hard | `browser-delayed-search` | `delayed-search` | `delayed` | stopped / `repeated_action`，2 步，2 次 retry，0 次 replan |

## 执行前准备

在 Week 7 worktree 中打开两个 PowerShell 终端，并在两个终端都执行：

```powershell
Set-Location "C:\Users\jbc51\Documents\ChatGPT\gui agent\.worktrees\week7-system-evaluation"
$env:VIRTUAL_ENV = $null
$env:EASYOCR_MODULE_PATH = (Resolve-Path "..\..\models\easyocr").Path
$env:HF_HOME = (Resolve-Path "..\week4-end-to-end-agent\.cache\huggingface").Path
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
```

每项任务都按以下顺序执行：

1. 终端 1 启动对应 fault profile 的 testbed。
2. 终端 2 先执行不带 `--execute` 的 dry-run，核对模型动作、坐标和预期结果。
3. 关闭并重新启动 testbed，确保 live run 从干净状态开始。
4. 终端 2 在同一命令末尾追加 `--execute`。
5. 逐动作阅读终端提示；只有动作正确时才逐字输入 `EXECUTE ACTION`。动作不安全或明显偏离时输入其他内容拒绝。
6. 记录下表字段后关闭 testbed，再开始下一项。

dry-run 的最终 JSON 和 `run-summary.json` 会给出脱敏的 `action_previews`：点击、滚动和拖动保留坐标，输入动作只保留字符数，不记录输入正文。必须先核对这些坐标落在可见 testbed 的预期控件上。

### Easy：打开 Browser

```powershell
# 终端 1
uv run --no-sync python examples/gui_testbed.py --fault-profile none

# 终端 2：先 dry-run；核对后重新启动 testbed，再追加 --execute
uv run --no-sync gui-agent run --task-id open-browser --provider qwen `
  --ocr-profile balanced --max-steps 12 `
  --run-dir artifacts/agent-runs/week7-real-open-browser-dry-v2
```

### Medium：瞬态搜索恢复

```powershell
# 终端 1
uv run --no-sync python examples/gui_testbed.py --fault-profile transient

# 终端 2：先 dry-run；核对后重新启动 testbed，再追加 --execute
uv run --no-sync gui-agent run --task-id search-content --provider qwen `
  --ocr-profile balanced --max-steps 12 `
  --run-dir artifacts/agent-runs/week7-real-search-content-dry-v2
```

### Hard：延迟搜索

```powershell
# 终端 1
uv run --no-sync python examples/gui_testbed.py --fault-profile delayed

# 终端 2：先 dry-run；核对后重新启动 testbed，再追加 --execute
uv run --no-sync gui-agent run --task-id delayed-search --provider qwen `
  --ocr-profile balanced --max-steps 12 `
  --run-dir artifacts/agent-runs/week7-real-delayed-search-dry-v2
```

## 首轮 dry-run 诊断

2026-09-10 的首轮运行使用旧摘要格式，只记录动作类型，无法核对坐标，因此不作为 live 放行依据：

| 任务 | 首轮结果 | 诊断 |
|---|---|---|
| `open-browser` | failed / `planner_output_invalid` | 2 步计划执行 3 个 dry-run click 后，模型提出不存在的 `step3` |
| `search-content` | stopped / `repeated_action` | 连续提出相同 `type_text`；dry-run 不改变桌面，重复动作保护正常停止 |
| `delayed-search` | stopped / `repeated_action` | 3 个 dry-run click 后连续提出相同 `type_text`，重复动作保护正常停止 |

CLI 已增加脱敏 `action_previews`，需要用上面的 `*-dry-v2` 目录重跑后再决定是否进入 live。

## 人工记录

| 难度 / 任务 | 成功 | 步数 | 墙钟耗时 | 人工确认次数 | OCR 错认 | 坐标偏差 | 与模拟结果的差异 |
|---|---|---:|---:|---:|---|---|---|
| easy / `open-browser` | 待执行 | — | — | — | — | — | — |
| medium / `search-content` | 待执行 | — | — | — | — | — | — |
| hard / `delayed-search` | 待执行 | — | — | — | — | — | — |

## sim-to-real 结论

待 3 项任务完成后填写。结论必须写明样本量为 3，只能定性描述，不能外推。

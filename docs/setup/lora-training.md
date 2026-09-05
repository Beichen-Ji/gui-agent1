# Week 5 LoRA/QLoRA 训练环境

本文只说明环境和执行边界。训练数据构建、单步检查、正式训练和评估命令会随对应 Task 实现。

## 本机要求

- Windows x86-64、Python 3.11。
- NVIDIA GPU；本项目基线为 RTX 5070 Ti 16 GB 和 PyTorch CUDA 12.8。
- 本地 Qwen 不需要 API key。首次下载公开基础模型需要联网；只有 Hugging Face 限流或资源要求认证时才在当前终端设置 `HF_TOKEN`。
- 数据集、缓存、checkpoint、adapter 和训练日志都留在 Git 忽略目录，不提交到仓库。

## 安装

在 Week 5 worktree 根目录运行：

```powershell
Remove-Item Env:VIRTUAL_ENV -ErrorAction SilentlyContinue
$env:UV_CACHE_DIR = Join-Path $PWD ".uv-cache"
$env:HF_HOME = Join-Path $PWD ".cache\huggingface"
uv python install 3.11
uv sync --locked --group dev --extra training --extra ocr --extra datasets
uv run python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

清除 `VIRTUAL_ENV` 可以避免其他 worktree 的 `.venv` 被 uv 忽略时反复出现路径不匹配警告。把 `UV_CACHE_DIR` 放在当前 worktree 内，可以绕开用户临时目录或全局 uv 缓存的 Windows 权限问题。

## 固定配置

唯一默认配置是 `configs/week5_qwen3vl_qlora.toml`。它使用 4-bit NF4、BF16 计算、batch size 1、梯度累积和冻结视觉塔的语言层 LoRA。代码会验证所有字段并拒绝未知配置。

训练输出只允许写到仓库内的 `artifacts/` 或 `checkpoints/`。正式 adapter 建议使用：

```text
artifacts/week5/qwen3vl-gui-lora/
```

不要把输出改到 `src/`、`tests/`、`configs/` 或 `docs/`。

## 显存门禁与回退

正式训练前必须先执行单个 forward/backward/save/reload 的 `training check`。若 `max_image_pixels=401408` OOM，只允许明确降到 `200704` 再检查一次，并在 manifest 中记录回退。

如果第二次仍失败，则在 Google Colab 或 Linux NVIDIA GPU 环境中使用同一仓库、锁文件、TOML 和训练数据执行相同的 `uv run gui-agent training ...` 命令。不得改成全参数训练，也不得删除图像输入来掩盖失败。

## 产物边界

训练成功后，交付的“模型权重”是 PEFT adapter：

```text
adapter_model.safetensors
adapter_config.json
training-manifest.json
```

基础模型继续从 `Qwen/Qwen3-VL-4B-Instruct` 加载。adapter 默认保留本地；如需发布为 GitHub Release 附件，必须先核对 SHA-256、许可和文件大小，并由用户单独确认。

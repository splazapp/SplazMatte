# SplazMatte

[English](#english) | [中文](#中文)

---

## English

SplazMatte is a video matting tool built with `NiceGUI + SAM2/SAM3 + MatAnyone/VideoMaMa`.

Key capabilities:

- Interactive keyframe annotation (positive/negative clicks)
- Full-video mask propagation (`SAM2` / `SAM3`)
- Fine-grained video matting (`MatAnyone` / `VideoMaMa`)
- Queue-based batch processing
- Optional cloud upload (Cloudflare R2 / Aliyun OSS) and Feishu notifications

### 1) Overview

- **Web UI workflow**: upload video -> annotate keyframes -> propagate masks -> run matting -> view/download outputs
- **Propagation models**
  - `SAM2`: stable default choice
  - `SAM3`: supports text prompt detection (for example, `"person in red"`)
- **Matting engines**
  - `MatAnyone`: faster, suitable for common scenes
  - `VideoMaMa`: finer edges, suitable for hair/complex boundaries
- **Outputs**: `alpha.mp4` and `foreground.mp4`

### 2) Prerequisites

- macOS / Linux (Windows can use `start.bat`)
- Conda (Miniconda/Anaconda)
- Python 3.11 (created by setup script)
- `ffmpeg` / `ffprobe`
- `just` (recommended command runner)

> Device selection is automatic: `CUDA` -> `MPS` (Apple Silicon) -> `CPU`.

### 3) Installation

#### 3.1 Clone

```bash
git clone <your-repo-url>
cd SplazMatte
```

#### 3.2 Setup environment

```bash
just setup
```

Or force platform:

```bash
just setup --mps
# or
just setup --cuda
```

#### 3.3 Download model weights

Download all:

```bash
just download-models
```

Download specific models:

```bash
just download-models --sam2
just download-models --sam3
just download-models --matanyone
just download-models --videomama
```

Verify downloads:

```bash
just download-models --verify
```

> `SAM3` is a gated HuggingFace model. Run `huggingface-cli login` and request access first.

### 4) Environment variables

```bash
cp .env.example .env
```

Common variables:

- `SPLAZMATTE_PORT`: web port (default `7870`)
- `SPLAZMATTE_STORAGE_SECRET`: NiceGUI storage secret (change in production)
- `SPLAZMATTE_STORAGE_BACKEND`: `r2` / `oss` / empty (disable upload)
- `SPLAZMATTE_FEISHU_WEBHOOK_URL`: Feishu bot webhook
- `SPLAZMATTE_LAN_IP`: optional manual LAN IP override

Storage-specific variables:

- R2: `SPLAZMATTE_R2_*`
- OSS: `SPLAZMATTE_OSS_*`

### 5) Run and usage

Start Web UI:

```bash
just app
```

Custom port:

```bash
just app-port 7870
```

Open `http://localhost:<port>`.

Recommended flow:

1. Upload a video
2. Annotate keyframes (positive/negative clicks)
3. Save keyframes (cover motion/occlusion changes)
4. Run propagation (`SAM2`/`SAM3`)
5. Choose matting engine and run matting
6. Review `Alpha` and `Foreground` outputs
7. Add tasks to queue for batch processing

### 6) CLI processing

```bash
just list                 # list sessions and status
just run <session_id>     # process one session
just run-queue            # process all pending queue tasks
```

### 7) Command quick reference

```bash
just app
just setup
just install-deps
just download-models
just device
just check
just clean
just clean-logs
just version
```

### 8) Project structure

```text
SplazMatte/
├── app.py                # NiceGUI entry
├── app_logic.py          # UI interaction logic
├── run.py                # CLI entry
├── matting_runner.py     # pipeline execution core
├── session_store.py      # session persistence
├── queue_logic.py        # queue logic
├── engines/              # SAM2/SAM3/MatAnyone/VideoMaMa wrappers
├── pipeline/             # frame extraction and video encoding
├── scripts/              # setup and model download scripts
├── models/               # local model weights
├── workspace/            # sessions, intermediates, outputs
└── sdks/                 # upstream source references
```

> `sdks/` is for source reference; avoid direct business dependency on SDK paths.

### 9) Session outputs

Each session is under `workspace/sessions/<session_id>/`:

- `source.*`: source video
- `frames/`: extracted frames
- `masks/`: keyframe and propagated masks
- `alpha.mp4`: alpha video
- `foreground.mp4`: foreground video
- `meta.json` / `state.json`: session metadata/state

Queue state is stored in `workspace/queue.json`.

### 10) Troubleshooting

- **`ffmpeg` / `ffprobe` not found**
  - Install ffmpeg and ensure it is available in `PATH`.
- **SAM3 download fails**
  - Run `huggingface-cli login` and confirm access to `facebook/sam3`.
- **Out of memory / slow performance**
  - Prefer `MatAnyone`, reduce video duration/resolution.
  - For `VideoMaMa`, reduce `batch_size` and `overlap`.
- **Cloud upload missing**
  - Check `SPLAZMATTE_STORAGE_BACKEND` and related credentials.
- **No Feishu notification**
  - Check `SPLAZMATTE_FEISHU_WEBHOOK_URL`.

### 11) Development notes

- Commands are managed by `justfile`; use `just <recipe>` whenever possible.
- Main runtime path:
  - `app.py` (UI) -> `app_logic.py` / `queue_logic.py` (interaction) -> `matting_runner.py` (execution)
- Recommended iteration style: validate in CLI first, then integrate into UI.

[Back to top](#splazmatte)

---

## 中文

SplazMatte 是一个基于 `NiceGUI + SAM2/SAM3 + MatAnyone/VideoMaMa` 的视频抠像工具。

核心能力：

- 关键帧交互标注（正/负点点击）
- 全视频遮罩传播（`SAM2` / `SAM3`）
- 精细视频抠像（`MatAnyone` / `VideoMaMa`）
- 队列批处理
- 可选云上传（Cloudflare R2 / 阿里云 OSS）和飞书通知

### 1）功能概览

- **Web UI 工作流**：上传视频 -> 标注关键帧 -> 运行传播 -> 开始抠像 -> 查看/下载结果
- **传播模型**
  - `SAM2`：默认稳定方案
  - `SAM3`：支持文本提示检测（例如 `"person in red"`）
- **抠像引擎**
  - `MatAnyone`：速度更快，适合常规场景
  - `VideoMaMa`：边缘更细腻，适合毛发/复杂边界
- **输出结果**：`alpha.mp4` 与 `foreground.mp4`

### 2）运行前准备

- macOS / Linux（Windows 可使用 `start.bat`）
- Conda（Miniconda/Anaconda）
- Python 3.11（由安装脚本自动创建）
- `ffmpeg` / `ffprobe`
- `just`（推荐的命令入口）

> 设备会自动选择：`CUDA` -> `MPS`（Apple Silicon）-> `CPU`。

### 3）安装

#### 3.1 克隆仓库

```bash
git clone <your-repo-url>
cd SplazMatte
```

#### 3.2 创建环境

```bash
just setup
```

或手动指定平台：

```bash
just setup --mps
# 或
just setup --cuda
```

#### 3.3 下载模型权重

下载全部：

```bash
just download-models
```

按需下载：

```bash
just download-models --sam2
just download-models --sam3
just download-models --matanyone
just download-models --videomama
```

校验下载：

```bash
just download-models --verify
```

> `SAM3` 是 HuggingFace gated 模型，首次下载前请先执行 `huggingface-cli login` 并申请访问权限。

### 4）环境变量配置

```bash
cp .env.example .env
```

常用变量：

- `SPLAZMATTE_PORT`：Web 端口（默认 `7870`）
- `SPLAZMATTE_STORAGE_SECRET`：NiceGUI 存储密钥（生产环境务必修改）
- `SPLAZMATTE_STORAGE_BACKEND`：`r2` / `oss` / 空（关闭上传）
- `SPLAZMATTE_FEISHU_WEBHOOK_URL`：飞书机器人 webhook
- `SPLAZMATTE_LAN_IP`：可选，手动覆盖局域网 IP

存储相关变量：

- R2：`SPLAZMATTE_R2_*`
- OSS：`SPLAZMATTE_OSS_*`

### 5）启动与使用

启动 Web UI：

```bash
just app
```

指定端口：

```bash
just app-port 7870
```

打开 `http://localhost:<port>`。

推荐流程：

1. 上传视频
2. 在关键帧上点击标注（正/负点）
3. 保存关键帧（覆盖运动与遮挡变化）
4. 运行传播（`SAM2`/`SAM3`）
5. 选择抠像引擎并执行抠像
6. 查看 `Alpha` 与 `Foreground` 结果
7. 需要批处理时，将任务加入队列统一执行

### 6）CLI 批处理

```bash
just list                 # 查看会话与状态
just run <session_id>     # 处理单个会话
just run-queue            # 处理队列中所有待处理任务
```

### 7）常用命令速查

```bash
just app
just setup
just install-deps
just download-models
just device
just check
just clean
just clean-logs
just version
```

### 8）项目结构

```text
SplazMatte/
├── app.py                # NiceGUI 入口
├── app_logic.py          # UI 交互逻辑
├── run.py                # CLI 入口
├── matting_runner.py     # 推理执行核心
├── session_store.py      # Session 持久化
├── queue_logic.py        # 队列逻辑
├── engines/              # SAM2/SAM3/MatAnyone/VideoMaMa 封装
├── pipeline/             # 抽帧与编码工具
├── scripts/              # 环境安装与模型下载脚本
├── models/               # 本地模型权重
├── workspace/            # 会话、中间产物、输出
└── sdks/                 # 上游源码参考
```

> `sdks/` 仅用于参考上游实现，不建议在业务代码中直接依赖其路径。

### 9）会话输出

每个任务会话位于 `workspace/sessions/<session_id>/`：

- `source.*`：原始视频
- `frames/`：抽帧结果
- `masks/`：关键帧与传播遮罩
- `alpha.mp4`：透明度视频
- `foreground.mp4`：前景视频
- `meta.json` / `state.json`：会话元数据与状态

队列状态保存在 `workspace/queue.json`。

### 10）常见问题

- **提示 `ffmpeg` / `ffprobe` not found**
  - 安装 ffmpeg，并确保已加入 `PATH`。
- **SAM3 下载失败**
  - 执行 `huggingface-cli login`，并确认有 `facebook/sam3` 访问权限。
- **显存不足 / 速度慢**
  - 优先使用 `MatAnyone`，降低视频时长或分辨率。
  - 使用 `VideoMaMa` 时可调小 `batch_size`、`overlap`。
- **云上传无结果**
  - 检查 `SPLAZMATTE_STORAGE_BACKEND` 和对应凭证。
- **飞书不通知**
  - 检查 `SPLAZMATTE_FEISHU_WEBHOOK_URL`。

### 11）开发说明

- 项目命令统一在 `justfile` 中，建议优先使用 `just <recipe>`。
- 主要运行链路：
  - `app.py`（UI）-> `app_logic.py` / `queue_logic.py`（交互）-> `matting_runner.py`（执行）
- 建议采用最小验证单元：先 CLI 验证，再接入 UI。

[返回顶部](#splazmatte)

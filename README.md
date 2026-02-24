# SplazMatte

[English](#english) | [中文](#中文)

---

## English

SplazMatte is a video matting and point tracking tool built with `NiceGUI + SAM2/SAM3 + MatAnyone/VideoMaMa + CoTracker3`.

Key capabilities:

- Interactive keyframe annotation (positive/negative clicks)
- Full-video mask propagation (`SAM2` / `SAM3`)
- Fine-grained video matting (`MatAnyone` / `VideoMaMa`)
- Point trajectory tracking (`CoTracker3`) with After Effects export
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
- **Point tracking** (separate `/tracking` page)
  - `CoTracker3`: supports Online (forward) and Offline (bidirectional) modes
  - Three input modes: manual click, SAM-assisted object selection, full-frame grid
  - Export results as After Effects keyframe data (`.txt`)
- **Outputs**: `alpha.mp4` and `foreground.mp4` (matting); tracking result video and AE keyframe data (tracking)

### 2) Prerequisites

- macOS / Linux (Windows can use `start.bat`)
- Conda (Miniconda/Anaconda)
- Python 3.11 (created by setup script)
- `ffmpeg` / `ffprobe`
- `just` (recommended command runner) — a command runner that stores project recipes in a `justfile`; this project uses it for `setup`, `app`, `download-models`, etc. See [just](https://github.com/casey/just) for installation instructions (Homebrew, Cargo, pre-built binaries, etc.).

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
just download-models --cotracker
```

Verify downloads:

```bash
just download-models --verify
```

> `SAM3` is a gated HuggingFace model. Run `huggingface-cli login` and request access first.
> CoTracker3 requires its source code under `sdks/co-tracker/`. The setup script clones it automatically.

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

Recommended matting flow (`/`):

1. Upload a video
2. Annotate keyframes (positive/negative clicks)
3. Save keyframes (cover motion/occlusion changes)
4. Run propagation (`SAM2`/`SAM3`)
5. Choose matting engine and run matting
6. Review `Alpha` and `Foreground` outputs
7. Add tasks to queue for batch processing

Point tracking flow (`/tracking`):

1. Upload a video (or restore a previous tracking session)
2. Select tracking points using one of three methods:
   - **Manual click**: click directly on the preview frame to add query points
   - **SAM-assisted**: use SAM2/SAM3 to select an object, then auto-generate grid points inside the mask
   - **Grid mode**: enable grid mode to track a uniform point grid across the entire frame
3. Save keyframes (supports multi-frame query points)
4. Configure tracking options (forward-only or bidirectional)
5. Run tracking (`CoTracker3`)
6. Preview the result video with tracked trajectories
7. Download result video and After Effects keyframe data (`.txt`)

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
├── app.py                # NiceGUI entry (matting + tracking pages)
├── app_logic.py          # matting UI interaction logic
├── cotracker_logic.py    # tracking UI interaction logic
├── tracking_runner.py    # tracking task execution
├── tracking_export.py    # After Effects keyframe data export
├── tracking_session_store.py  # tracking session persistence
├── run.py                # CLI entry
├── matting_runner.py     # matting pipeline execution core
├── session_store.py      # matting session persistence
├── queue_logic.py        # queue logic (matting + tracking)
├── engines/              # SAM2/SAM3/MatAnyone/VideoMaMa/CoTracker wrappers
├── pipeline/             # frame extraction and video encoding
├── scripts/              # setup and model download scripts
├── models/               # local model weights
├── workspace/            # sessions, intermediates, outputs
└── sdks/                 # upstream source (co-tracker cloned by setup.sh)
```

> `sdks/co-tracker` is cloned automatically during `just setup`. Other directories under `sdks/` are for source reference only.

### 9) Session outputs

**Matting sessions** are under `workspace/sessions/<session_id>/`:

- `source.*`: source video
- `frames/`: extracted frames
- `masks/`: keyframe and propagated masks
- `alpha.mp4`: alpha video
- `foreground.mp4`: foreground video
- `meta.json` / `state.json`: session metadata/state

**Tracking sessions** are under `workspace/tracking_sessions/<session_id>/`:

- `frames/`: extracted frames
- `keyframes/*.json`: saved query point keyframes
- `raw_tracks.npy` / `raw_visibility.npy`: raw tracking results
- `meta.json` / `state.json`: session metadata/state

**Tracking results** are under `workspace/tracking_results/`:

- `tracking_*.mp4`: result video with visualized trajectories
- `ae_export_*.txt`: After Effects keyframe data

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
- **CoTracker import error / `No module named 'cotracker'`**
  - Run `just setup` to clone `sdks/co-tracker`, or manually: `git clone https://github.com/facebookresearch/co-tracker.git sdks/co-tracker`
- **CoTracker model weights missing**
  - Run `python scripts/download_models.py --cotracker` to download `scaled_online.pth` and `scaled_offline.pth`.
- **CoTracker slow on Apple Silicon**
  - CoTracker uses CPU on MPS by default (MPS lacks `grid_sampler_3d`). This is expected behavior.

### 11) Development notes

- Commands are managed by `justfile`; use `just <recipe>` whenever possible.
- Main runtime paths:
  - Matting: `app.py` -> `app_logic.py` / `queue_logic.py` -> `matting_runner.py`
  - Tracking: `app.py` (`/tracking`) -> `cotracker_logic.py` -> `tracking_runner.py` -> `engines/cotracker_engine.py`
- Recommended iteration style: validate in CLI first, then integrate into UI.

[Back to top](#splazmatte)

---

## 中文

SplazMatte 是一个基于 `NiceGUI + SAM2/SAM3 + MatAnyone/VideoMaMa + CoTracker3` 的视频抠像与轨迹追踪工具。

核心能力：

- 关键帧交互标注（正/负点点击）
- 全视频遮罩传播（`SAM2` / `SAM3`）
- 精细视频抠像（`MatAnyone` / `VideoMaMa`）
- 点轨迹追踪（`CoTracker3`），支持导出 After Effects 关键帧数据
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
- **点轨迹追踪**（独立 `/tracking` 页面）
  - `CoTracker3`：支持 Online（前向）和 Offline（双向）两种模式
  - 三种输入方式：手动点击、SAM 辅助选择目标、全帧网格
  - 导出 After Effects 关键帧数据（`.txt`）
- **输出结果**：`alpha.mp4` 与 `foreground.mp4`（抠像）；追踪结果视频与 AE 关键帧数据（追踪）

### 2）运行前准备

- macOS / Linux（Windows 可使用 `start.bat`）
- Conda（Miniconda/Anaconda）
- Python 3.11（由安装脚本自动创建）
- `ffmpeg` / `ffprobe`
- `just`（推荐的命令入口）— 一个命令运行器，将项目命令保存在 `justfile` 中；本项目用它执行 `setup`、`app`、`download-models` 等。安装方法请参阅 [just](https://github.com/casey/just)（支持 Homebrew、Cargo、预编译二进制等）。

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
just download-models --cotracker
```

校验下载：

```bash
just download-models --verify
```

> `SAM3` 是 HuggingFace gated 模型，首次下载前请先执行 `huggingface-cli login` 并申请访问权限。
> CoTracker3 需要 `sdks/co-tracker/` 下的源码，安装脚本会自动克隆。

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

抠像流程（`/`）：

1. 上传视频
2. 在关键帧上点击标注（正/负点）
3. 保存关键帧（覆盖运动与遮挡变化）
4. 运行传播（`SAM2`/`SAM3`）
5. 选择抠像引擎并执行抠像
6. 查看 `Alpha` 与 `Foreground` 结果
7. 需要批处理时，将任务加入队列统一执行

轨迹追踪流程（`/tracking`）：

1. 上传视频（或恢复之前的追踪会话）
2. 通过以下三种方式之一选择追踪点：
   - **手动点击**：在预览帧上直接点击添加查询点
   - **SAM 辅助**：使用 SAM2/SAM3 选择目标对象，自动在 Mask 内生成网格点
   - **网格模式**：启用网格模式，在整帧上追踪均匀分布的点
3. 保存关键帧（支持多帧查询点）
4. 配置追踪选项（前向追踪或双向追踪）
5. 运行追踪（`CoTracker3`）
6. 预览带有可视化轨迹的结果视频
7. 下载结果视频与 After Effects 关键帧数据（`.txt`）

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
├── app.py                # NiceGUI 入口（抠像 + 追踪页面）
├── app_logic.py          # 抠像 UI 交互逻辑
├── cotracker_logic.py    # 追踪 UI 交互逻辑
├── tracking_runner.py    # 追踪任务执行
├── tracking_export.py    # After Effects 关键帧数据导出
├── tracking_session_store.py  # 追踪会话持久化
├── run.py                # CLI 入口
├── matting_runner.py     # 抠像推理执行核心
├── session_store.py      # 抠像 Session 持久化
├── queue_logic.py        # 队列逻辑（抠像 + 追踪）
├── engines/              # SAM2/SAM3/MatAnyone/VideoMaMa/CoTracker 封装
├── pipeline/             # 抽帧与编码工具
├── scripts/              # 环境安装与模型下载脚本
├── models/               # 本地模型权重
├── workspace/            # 会话、中间产物、输出
└── sdks/                 # 上游源码（co-tracker 由 setup.sh 自动克隆）
```

> `sdks/co-tracker` 在执行 `just setup` 时自动克隆。`sdks/` 下其他目录仅供参考上游实现。

### 9）会话输出

**抠像会话**位于 `workspace/sessions/<session_id>/`：

- `source.*`：原始视频
- `frames/`：抽帧结果
- `masks/`：关键帧与传播遮罩
- `alpha.mp4`：透明度视频
- `foreground.mp4`：前景视频
- `meta.json` / `state.json`：会话元数据与状态

**追踪会话**位于 `workspace/tracking_sessions/<session_id>/`：

- `frames/`：抽帧结果
- `keyframes/*.json`：已保存的查询点关键帧
- `raw_tracks.npy` / `raw_visibility.npy`：原始追踪结果
- `meta.json` / `state.json`：会话元数据与状态

**追踪结果**位于 `workspace/tracking_results/`：

- `tracking_*.mp4`：带可视化轨迹的结果视频
- `ae_export_*.txt`：After Effects 关键帧数据

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
- **CoTracker 导入报错 / `No module named 'cotracker'`**
  - 执行 `just setup` 以克隆 `sdks/co-tracker`，或手动执行：`git clone https://github.com/facebookresearch/co-tracker.git sdks/co-tracker`
- **CoTracker 模型权重缺失**
  - 执行 `python scripts/download_models.py --cotracker` 下载 `scaled_online.pth` 和 `scaled_offline.pth`。
- **CoTracker 在 Apple Silicon 上运行较慢**
  - CoTracker 在 MPS 上默认使用 CPU（MPS 不支持 `grid_sampler_3d`），这是预期行为。

### 11）开发说明

- 项目命令统一在 `justfile` 中，建议优先使用 `just <recipe>`。
- 主要运行链路：
  - 抠像：`app.py` -> `app_logic.py` / `queue_logic.py` -> `matting_runner.py`
  - 追踪：`app.py`（`/tracking`）-> `cotracker_logic.py` -> `tracking_runner.py` -> `engines/cotracker_engine.py`
- 建议采用最小验证单元：先 CLI 验证，再接入 UI。

[返回顶部](#splazmatte)

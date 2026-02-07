# SplazMatte MVP 设计文档

> AI 视频抠图内部工具 — Splaz 团队内部使用
> 版本: v0.1 MVP
> 日期: 2026-02-07

---

## 1. 项目概述

### 1.1 是什么

SplazMatte 是 Splaz 团队内部使用的 AI 视频抠图工具。用户上传视频，通过文本描述或交互标注指定抠图目标，系统自动输出高质量 alpha matte 和前景视频。

### 1.2 为什么做

Splaz 作为社交平台，内容创作（短视频、AI Agent 形象等）需要高质量的视频抠图能力。市面上的商业工具（如 Runway、Unscreen）按量计费且无法定制，开源方案已经成熟到可以搭建内部工具。

### 1.3 MVP 目标

用最短时间搭建一个可用的 Gradio Web 应用，覆盖核心抠图流程：

- 上传视频 → 选择目标 → 自动抠图 → 下载结果
- 支持文本提示（SAM 3）和点击标注两种选择方式
- 支持 MatAnyone（人体快速）和 VideoMaMa（通用高质量）两种抠图引擎
- 单机部署，内部人员通过浏览器访问

### 1.4 不做什么（MVP 范围外）

- ❌ 前后端分离的 Web 应用（Phase 2）
- ❌ 与 Splaz 主系统集成
- ❌ 用户权限管理
- ❌ 批量处理 / 任务队列
- ❌ 逐帧手动修正
- ❌ 移动端适配

---

## 2. 核心模型与开源项目

### 2.1 技术选型

| 组件 | 项目 | 版本/来源 | 角色 | 许可证 |
|------|------|----------|------|--------|
| 目标分割 | **SAM 3** | facebookresearch/sam3 | 文本/视觉提示分割 + 视频追踪 | SAM License |
| 人体抠图 | **MatAnyone** | pq-yang/MatAnyone (CVPR 2025) | 首帧 mask → 全视频 alpha matte | NTU S-Lab 1.0 |
| 通用抠图 | **VideoMaMa** | cvlab-kaist/VideoMaMa | mask → 高质量 alpha matte (扩散模型) | CC BY-NC 4.0 |
| UI 框架 | **Gradio** | gradio-app/gradio | Web 界面 | Apache 2.0 |

### 2.2 模型能力定位

**SAM 3 — 目标选择层**

负责"找到目标"。相比 SAM 2 的核心升级：
- 文本提示：输入"穿红衣服的人"即可自动分割所有匹配实例
- 示例提示：上传参考图片，找到视觉相似的目标
- 视觉提示：兼容 SAM 2 的点击/框选方式
- 视频追踪：首帧分割后自动传播到全视频
- 解耦检测-追踪架构：848M 参数，支持 Presence Token 减少误检

**MatAnyone — 人体抠图引擎**

负责"把粗 mask 变成精细 alpha"（人体场景）。
- 基于记忆传播机制，只需首帧 mask
- 区域自适应融合：核心区域语义稳定 + 边界精细细节
- 轻量推理：~8GB VRAM (1080p)
- 支持多目标独立抠图
- 限制：主要针对人体优化

**VideoMaMa — 通用抠图引擎**

负责"把粗 mask 变成精细 alpha"（通用场景）。
- 基于 Stable Video Diffusion 的生成先验
- 通用物体支持（不限人体）
- 单次前向传播，mask 容错性高
- 限制：~16-24GB VRAM，速度较慢

### 2.3 模型协作关系

```
用户输入 (视频 + 提示)
        │
        ▼
   ┌─────────┐
   │  SAM 3  │  ← 目标选择（文本/点击/框选）
   └────┬────┘
        │ 输出: 首帧 binary mask + 视频 mask 序列
        │
        ├──── 人体场景 ────▶ MatAnyone ──▶ Alpha + Foreground
        │                    (快速、轻量)
        │
        └──── 通用场景 ────▶ VideoMaMa ──▶ Alpha + Foreground
                             (高质量、通用)
```

关键设计：SAM 3 负责"找什么"，MatAnyone/VideoMaMa 负责"抠得精"。分割和抠图是两个独立步骤，中间通过 mask 衔接。

---

## 3. 用户流程

### 3.1 主流程

```
Step 1: 上传视频
   │    支持 mp4/mov/avi，建议 ≤ 1080p, ≤ 60s (MVP 限制)
   ▼
Step 2: 选择目标
   │    方式 A: 文本描述 → "穿蓝色衣服的人"
   │    方式 B: 在首帧画面上点击目标
   │    方式 C: 在首帧画面上框选目标
   ▼
Step 3: 预览分割结果
   │    首帧显示 mask 叠加效果（半透明彩色）
   │    用户确认 or 重新选择
   ▼
Step 4: 选择抠图引擎 & 启动处理
   │    MatAnyone (推荐人物) / VideoMaMa (推荐通用)
   │    显示预计耗时，开始处理
   ▼
Step 5: 查看结果 & 下载
        预览: Alpha 视频 / 前景合成预览
        下载: Alpha 视频 / 前景视频 (透明背景)
```

### 3.2 UI 布局（Gradio）

```
┌──────────────────────────────────────────────────────────┐
│                    SplazMatte                             │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  ┌─────────────────────┐  ┌───────────────────────────┐  │
│  │                     │  │                           │  │
│  │   视频上传 / 首帧    │  │   分割结果预览             │  │
│  │   预览 + 交互标注    │  │   (mask 叠加在原图上)      │  │
│  │                     │  │                           │  │
│  └─────────────────────┘  └───────────────────────────┘  │
│                                                          │
│  ┌──────────────────────────────────────────────────┐    │
│  │ 文本提示: [穿红衣服的人________________] [分割]    │    │
│  │                                                   │    │
│  │ 抠图引擎: ○ MatAnyone (人物快速)                   │    │
│  │           ○ VideoMaMa (通用高质量)                  │    │
│  │                                                   │    │
│  │           [开始抠图]                               │    │
│  └──────────────────────────────────────────────────┘    │
│                                                          │
│  ┌─────────────────────┐  ┌───────────────────────────┐  │
│  │  Alpha 视频预览      │  │  前景视频预览              │  │
│  │  (黑白)             │  │  (绿色/棋盘格背景)         │  │
│  │  [下载 Alpha]       │  │  [下载前景]                │  │
│  └─────────────────────┘  └───────────────────────────┘  │
└──────────────────────────────────────────────────────────┘
```

### 3.3 交互细节

**目标选择交互：**
- 文本模式：输入框 + "分割"按钮，SAM 3 返回所有匹配实例的 mask
- 点击模式：在首帧图片上点击（positive point），支持多次点击细化
- 框选模式：在首帧图片上拖出 bounding box
- 分割结果以半透明彩色 overlay 显示在原图上
- 如果匹配多个实例，用不同颜色标注，用户可点击选择/取消

**处理进度：**
- 分割阶段：秒级，同步等待即可
- 抠图阶段：根据视频长度和分辨率，几十秒到几分钟
- 显示进度条 + 预计剩余时间
- 处理过程中 UI 不阻塞（Gradio 的 queue 机制）

---

## 4. 系统架构

### 4.1 整体架构

MVP 采用单机单进程架构，Gradio 作为 Web 服务器 + UI 框架：

```
浏览器 (内网访问)
    │
    │ HTTP / WebSocket
    ▼
┌──────────────────────────────────────────────┐
│  Gradio Server (Python)                      │
│                                              │
│  ┌────────────────────────────────────────┐  │
│  │  UI Layer (Gradio Components)          │  │
│  │  视频上传 / 首帧标注 / 结果预览         │  │
│  └──────────────┬─────────────────────────┘  │
│                 │                             │
│  ┌──────────────▼─────────────────────────┐  │
│  │  Pipeline Orchestrator                 │  │
│  │  流程编排: 上传→分割→抠图→后处理→输出   │  │
│  └──────────────┬─────────────────────────┘  │
│                 │                             │
│  ┌──────────────▼─────────────────────────┐  │
│  │  Engine Layer (模型推理)                │  │
│  │  ├── SAM3Engine      (分割/追踪)       │  │
│  │  ├── MatAnyoneEngine (人体抠图)        │  │
│  │  ├── VideoMaMaEngine (通用抠图)        │  │
│  │  └── PostProcessor   (合成/编码)       │  │
│  └──────────────┬─────────────────────────┘  │
│                 │                             │
│  ┌──────────────▼─────────────────────────┐  │
│  │  Storage (本地文件系统)                  │  │
│  │  uploads/ → frames/ → masks/ → outputs/ │  │
│  └────────────────────────────────────────┘  │
└──────────────────────────────────────────────┘
         │
    GPU (NVIDIA)
```

### 4.2 目录结构

```
splazmatte/
├── app.py                      # Gradio 应用入口
├── config.py                   # 配置 (模型路径、参数默认值)
│
├── pipeline/
│   ├── orchestrator.py         # 流程编排
│   ├── video_io.py             # 视频读写、帧提取、视频编码
│   └── postprocess.py          # Alpha 合成、背景替换、格式转换
│
├── engines/
│   ├── base.py                 # 引擎抽象基类
│   ├── sam3_engine.py          # SAM 3 分割引擎
│   ├── matanyone_engine.py     # MatAnyone 抠图引擎
│   └── videomama_engine.py     # VideoMaMa 抠图引擎
│
├── ui/
│   ├── components.py           # Gradio UI 组件组装
│   └── callbacks.py            # UI 事件回调函数
│
├── utils/
│   ├── image_utils.py          # 图像处理工具函数
│   └── mask_utils.py           # Mask 可视化、格式转换
│
├── workspace/                  # 运行时工作目录 (gitignore)
│   ├── uploads/                # 上传的原始视频
│   ├── frames/                 # 提取的视频帧
│   ├── masks/                  # SAM 3 输出的 mask
│   └── outputs/                # 最终输出文件
│
├── models/                     # 模型权重 (gitignore)
│   ├── sam3/
│   ├── matanyone/
│   └── videomama/
│
├── scripts/
│   ├── setup_env.sh            # 环境安装脚本
│   └── download_models.sh      # 模型下载脚本
│
├── requirements.txt
└── README.md
```

### 4.3 核心模块职责

**`app.py` — 入口**
- 初始化 Gradio 界面
- 加载配置
- 启动服务（指定 host/port）

**`pipeline/orchestrator.py` — 流程编排**
- 管理完整的处理流程
- 调度 Engine 执行
- 管理临时文件生命周期
- 错误处理和清理

**`engines/` — 模型推理层**

每个 Engine 遵循统一接口：

```
Engine 接口设计思路:

SAM3Engine:
  - load_model()           → 加载 SAM 3 权重
  - segment_image(         → 单帧分割
      image, prompt_type, prompt_data
    ) → masks, scores
  - track_video(           → 视频追踪
      video_frames, initial_mask
    ) → mask_sequence

MatAnyoneEngine:
  - load_model()           → 加载 MatAnyone 权重
  - process(               → 执行抠图
      video_path, mask_path, options
    ) → alpha_path, foreground_path

VideoMaMaEngine:
  - load_model()           → 加载 VideoMaMa 权重
  - process(               → 执行抠图
      frames_dir, masks_dir, options
    ) → alpha_path, foreground_path
```

**`pipeline/postprocess.py` — 后处理**
- Alpha 视频编码 (灰度 H.264)
- 前景视频编码 (带透明通道 / 绿幕 / 棋盘格背景)
- 合成预览 (替换为纯色背景展示效果)

---

## 5. 数据流

### 5.1 处理流水线

```
输入视频 (mp4)
    │
    ▼
[视频解码] → 帧序列 (PNG/JPG)
    │          保存到 workspace/frames/{session_id}/
    │
    ▼
[首帧提取] → 首帧图片
    │          传给前端展示 + SAM 3 输入
    │
    ▼
[SAM 3 分割] → 首帧 mask (PNG, 单通道)
    │             + mask 叠加预览图
    │             如果是视频追踪模式 → 全帧 mask 序列
    │
    ▼
[用户确认] → 选择引擎 → 开始抠图
    │
    ├── MatAnyone 路径:
    │   输入: 原始视频 + 首帧 mask
    │   输出: alpha 视频 + 前景视频
    │
    └── VideoMaMa 路径:
        输入: 帧序列 + mask 序列 (需 SAM 3 先追踪全帧)
        输出: alpha 帧序列 → 编码为视频
    │
    ▼
[后处理] → 最终输出文件
              ├── {session_id}_alpha.mp4      (灰度 alpha 视频)
              ├── {session_id}_foreground.mp4  (前景 + 绿幕/透明)
              └── {session_id}_preview.mp4     (合成预览)
```

### 5.2 两种引擎的数据流差异

**MatAnyone 路径（推荐人体）：**
```
原始视频 (.mp4) + 首帧 mask (.png)
        │
        ▼ MatAnyone 内部处理
  逐帧记忆传播，自动生成全帧 alpha
        │
        ▼
  alpha 视频 + foreground 视频
```
特点：MatAnyone 本身就有视频追踪能力，只需首帧 mask，无需 SAM 3 的视频追踪。

**VideoMaMa 路径（推荐通用）：**
```
帧序列 (frames/) + mask 序列 (masks/)
        │
        ▼ 需要先用 SAM 3 生成全帧 mask
        ▼ VideoMaMa 逐帧处理
  每帧: RGB + mask → alpha (扩散模型单步推理)
        │
        ▼
  alpha 帧序列 → ffmpeg 编码为视频
```
特点：VideoMaMa 是逐帧 mask-to-matte，需要每帧都有对应 mask，所以需要 SAM 3 先做视频追踪。

### 5.3 Session 管理

每次处理创建一个 session，用 UUID 标识：

```
workspace/
└── sessions/
    └── a1b2c3d4/
        ├── input.mp4           # 原始视频
        ├── frames/             # 提取的帧
        │   ├── 000000.jpg
        │   ├── 000001.jpg
        │   └── ...
        ├── masks/              # SAM 3 输出
        │   ├── 000000.png      # 首帧 mask
        │   └── ...             # 追踪 mask (VideoMaMa 路径)
        ├── alpha.mp4           # 输出: alpha
        ├── foreground.mp4      # 输出: 前景
        └── preview.mp4         # 输出: 合成预览
```

定期清理策略：保留最近 N 个 session 或超过 X 小时自动删除。

---

## 6. 关键设计决策

### 6.1 为什么 MVP 用 Gradio

| 考量 | 决策 |
|------|------|
| 开发速度 | Gradio 3 天出原型 vs 前后端分离 2-3 周 |
| 模型集成 | SAM 3 和 MatAnyone 都自带 Gradio demo，可直接参考 |
| 内部工具 | 不需要精致 UI，够用就行 |
| 验证优先 | 先确认模型效果和参数，再投入正式开发 |

### 6.2 SAM 3 vs SAM 2

选 SAM 3 因为文本提示能力：
- 用户说"穿红衣服的人"就能找到目标，无需手动标注
- 支持 Presence Token，相似目标区分能力更强
- 兼容 SAM 2 的所有交互方式（点击/框选）
- 视频追踪能力继承自 SAM 2

### 6.3 MatAnyone + VideoMaMa 双引擎

不选一个是因为两者互补：

| 场景 | MatAnyone | VideoMaMa |
|------|-----------|-----------|
| 人物抠图 | ✅ 专门优化，效果最好 | ✅ 可用但过重 |
| 通用物体 | ❌ 人体之外效果下降 | ✅ 通用支持 |
| 速度 | 快 (~8GB VRAM) | 慢 (~16-24GB VRAM) |
| 半透明处理 | 一般 | ✅ 扩散先验更擅长 |
| mask 依赖 | 只需首帧 | 需每帧 mask |

MVP 默认推荐 MatAnyone（覆盖最常见的人物场景），VideoMaMa 作为高级选项。

### 6.4 模型加载策略

MVP 阶段使用懒加载 + 常驻内存：

```
应用启动 → 不加载任何模型
首次使用某引擎 → 加载到 GPU → 保持在显存中
显存不足时 → 卸载最近最少使用的模型
```

SAM 3 常驻（几乎每次都用），MatAnyone/VideoMaMa 按需加载。
如果 GPU 显存足够（≥24GB），可以全部常驻。

---

## 7. 硬件与环境

### 7.1 GPU 需求

| 模型 | VRAM 需求 | 备注 |
|------|----------|------|
| SAM 3 | ~8GB | 848M 参数 |
| MatAnyone | ~8GB | 1080p 推理 |
| VideoMaMa | ~16-24GB | 基于 SVD |
| **同时加载 SAM 3 + MatAnyone** | **~16GB** | **MVP 推荐配置** |
| **三个全部常驻** | **~32-40GB** | 需要 A100 或多卡 |

**MVP 推荐硬件：**
- GPU: NVIDIA RTX 4090 (24GB) 或 A10G (24GB)
- 可以同时常驻 SAM 3 + MatAnyone
- VideoMaMa 需要卸载其中一个后加载

### 7.2 软件环境

```
Python 3.10+
PyTorch 2.x + CUDA 12.x
FFmpeg (视频编解码)
Gradio 4.x+
```

### 7.3 模型权重获取

```
SAM 3:
  - 需在 HuggingFace 申请权限: https://huggingface.co/facebook/sam3
  - hf auth login 认证后下载

MatAnyone:
  - 直接下载: https://huggingface.co/PeiqingYang/MatAnyone
  - 或 pip install git+https://github.com/pq-yang/MatAnyone

VideoMaMa:
  - 下载: https://huggingface.co/SammyLim/VideoMaMa
  - 依赖 Stable Video Diffusion 权重
```

---

## 8. 限制与约束（MVP）

### 8.1 输入限制

| 参数 | 限制 | 原因 |
|------|------|------|
| 视频时长 | ≤ 60 秒 | 单机处理避免超时 |
| 分辨率 | ≤ 1080p | GPU 显存限制 |
| 格式 | mp4, mov, avi | FFmpeg 支持的常见格式 |
| 帧率 | 保持原始帧率 | 不做帧率转换 |

### 8.2 并发限制

- Gradio queue 模式：同一时间只处理 1 个抠图任务
- 其他请求排队等待
- 分割操作（SAM 3 单帧）可以快速响应

### 8.3 已知限制

- 不支持逐帧手动修正 mask
- 不支持多目标同时抠图输出（需分别处理）
- VideoMaMa 路径处理时间较长，缺少细粒度进度
- Session 文件需手动管理磁盘空间

---

## 9. 输出格式

### 9.1 Alpha 视频

- 格式：H.264 MP4，灰度编码
- 含义：白色 = 完全前景，黑色 = 完全背景，灰度 = 半透明
- 用途：可导入 AE/PR/DaVinci 作为遮罩通道

### 9.2 前景视频

- 格式：H.264 MP4 (绿幕背景) 或 PNG 序列 (透明通道)
- 绿幕版本：前景保持原色，背景替换为 #00FF00
- PNG 序列：每帧 RGBA PNG，适合精确合成

### 9.3 合成预览

- 格式：H.264 MP4
- 内容：前景合成在灰色/棋盘格背景上，便于直观检查边缘质量

---

## 10. Phase 2 规划（MVP 之后）

MVP 上线内部使用后，根据反馈决定是否推进 Phase 2：

### 10.1 前后端分离重构

```
Vue 3 前端:
  - 专业视频帧浏览器 (时间轴拖动)
  - Canvas 交互标注 (点击/框选/画笔)
  - 实时 mask 叠加预览
  - 多目标管理面板
  - 任务列表 + 进度管理

FastAPI 后端:
  - POST /segment → 同步分割 (秒级)
  - POST /matting → 异步抠图任务
  - GET /tasks/{id} → 查询进度
  - WebSocket → 实时进度推送

GPU Worker:
  - Celery + Redis 任务队列
  - 支持多任务排队
  - GPU 资源管理
```

### 10.2 功能扩展

- 逐帧 mask 修正（画笔增减 mask 区域）
- 多目标同时抠图 + 独立输出
- 批量处理（队列管理）
- 结果存储到 Cloudflare R2
- 与 Splaz Admin 集成

### 10.3 引擎扩展

- RVM (Robust Video Matting)：实时预览模式，无需 mask
- BRIA RMBG 2.0：关键帧单帧 alpha 精修
- SAM2Long：长视频分割增强

---

## 11. 开发计划

### 11.1 开发顺序（使用 Claude Code）

```
Day 1-2: 环境搭建 + 模型验证
  ├── 搭建 conda 环境，安装依赖
  ├── 分别验证 SAM 3 / MatAnyone / VideoMaMa 的推理
  └── 确认 GPU 显存使用、推理速度

Day 3-4: Engine 层开发
  ├── SAM3Engine: 文本分割 + 点击分割 + 视频追踪
  ├── MatAnyoneEngine: 封装推理接口
  └── VideoMaMaEngine: 封装推理接口

Day 5-6: Pipeline + 后处理
  ├── Orchestrator: 串联 分割→抠图→后处理
  ├── video_io: FFmpeg 帧提取/视频编码
  └── postprocess: Alpha 编码、前景合成

Day 7-8: Gradio UI
  ├── 视频上传 + 首帧展示
  ├── 文本/点击分割交互
  ├── 引擎选择 + 处理进度
  └── 结果预览 + 下载

Day 9-10: 测试 + 调优
  ├── 不同场景测试 (人物/物体/复杂背景)
  ├── 参数调优 (MatAnyone 的 warmup/erode/dilate)
  └── 错误处理、边界情况
```

### 11.2 Claude Code 开发建议

- 每个 Engine 独立开发和测试，先跑通单模型再组合
- 先做 MatAnyone 路径（依赖最少），再做 VideoMaMa 路径
- Gradio UI 最后做，先用命令行验证 Pipeline
- 善用 MatAnyone 自带的 Gradio demo 作为参考
- 模型权重路径通过 config.py 集中管理，不硬编码

---

## 12. 许可证评估

| 项目 | 许可证 | 内部工具使用 | 注意事项 |
|------|--------|------------|---------|
| SAM 3 | SAM License | ✅ | 非 Apache 2.0，需确认具体条款 |
| MatAnyone | NTU S-Lab License 1.0 | ✅ | 不可商业再分发模型本身 |
| VideoMaMa | CC BY-NC 4.0 | ✅ | 仅限非商业用途 |
| Gradio | Apache 2.0 | ✅ | 无风险 |

**结论：** 作为内部工具，所有许可证均可使用。如果未来考虑作为 Splaz 用户功能对外提供，需重新评估 MatAnyone 和 VideoMaMa 的许可证合规性。

---

## 附录 A: 参考资料

- SAM 3 GitHub: https://github.com/facebookresearch/sam3
- SAM 3 论文: https://arxiv.org/abs/2511.16719
- MatAnyone GitHub: https://github.com/pq-yang/MatAnyone
- MatAnyone 论文 (CVPR 2025): https://arxiv.org/abs/2501.14677
- VideoMaMa GitHub: https://github.com/cvlab-kaist/VideoMaMa
- VideoMaMa 论文: https://arxiv.org/abs/2601.14255
- VideoMaMa HuggingFace Demo: https://huggingface.co/spaces/SammyLim/VideoMaMa

## 附录 B: 术语表

| 术语 | 含义 |
|------|------|
| Alpha Matte | 透明度遮罩，0-255 灰度值表示像素的前景占比 |
| Binary Mask | 二值遮罩，每个像素只有前景(1)或背景(0) |
| Matting | 从图像/视频中提取精确的 alpha 通道（含半透明边缘） |
| Segmentation | 将图像/视频中的目标区域分割出来（通常是二值 mask） |
| Prompt | 提示，告诉模型"找什么"的输入（文本/点/框/图片） |
| Foreground | 前景，alpha > 0 的区域及其像素值 |
| Composition | 合成，用 alpha 将前景叠加到新背景上 |
| Trimap | 三分图，将图像分为前景/背景/未知三个区域 |

---

**文档状态:** MVP 设计初稿
**最后更新:** 2026-02-07
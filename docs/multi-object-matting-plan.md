# Multi-Object Matting Pipeline 工作计划

> 解决多人物、跨帧标注场景下的视频抠图问题

---

## 1. 问题背景

### 当前实现（单 object matting）

```
用户标注 keyframe mask → MatAnyone 逐帧传播 → alpha matte
```

当前 pipeline 中 MatAnyone 将所有 keyframe mask 视为同一个前景 object。每次注入新 keyframe 时会调用 `clear_temp_mem()` 重置记忆，导致之前学习的 object 信息丢失。

**局限性：无法处理多人物场景。**

### 典型问题场景

```
帧 0:   人物 A 出现在画面中间
帧 10:  人物 A 离开，人物 B 出现在画面中间
帧 20:  人物 A、B 同时出现
帧 40:  人物 A 单独出现
帧 50:  人物 B 单独出现
```

用户希望仅标注帧 0（A）和帧 10（B），就能在整段视频中将 A 和 B 都抠出来。

当前实现的问题：
- 帧 0 注入 A 的 mask → 记忆中只有 A
- 帧 10 注入 B 的 mask → **记忆重置** → A 的信息丢失
- 帧 20 起 A 不再被识别为前景

---

## 2. 解决方案概述

**核心思路：SAM2 Video Predictor 做多目标跟踪 + mask 传播，MatAnyone 做精细 alpha matting。**

```
SAM2 Image Predictor    →  单帧交互式标注（点击生成 mask）
SAM2 Video Predictor    →  多目标联合跟踪 + mask 传播到全视频（新增）
MatAnyone               →  基于传播后的 combined mask 做精细 alpha matting
```

### 为什么需要 SAM2 Video Predictor

| 能力              | SAM2 Image Predictor | SAM2 Video Predictor | MatAnyone          |
|-------------------|---------------------|---------------------|--------------------|
| 多 object 区分     | 不支持              | 支持（obj_id 区分）  | 不支持（前景/背景） |
| 跨帧传播           | 不支持              | Memory Attention     | 有，但注入时重置    |
| 互斥约束           | 不适用              | 同一像素只属于一个 obj | 不适用            |
| Alpha matting      | 不支持（硬边缘）     | 不支持（硬边缘）      | 支持（头发丝级别）  |

### 为什么不能独立传播后合并

如果对每个 object 独立运行 SAM2 VP 再做 `mask_A | mask_B`，会出现身份漂移问题：

```
独立传播 A: 帧 0 跟踪 A → 帧 10 A 消失 → tracker 漂移到 B 身上
独立传播 B: 帧 10 跟踪 B → 正常

结果: 帧 20 A 重新出现时，A 的 tracker 仍跟着 B → A 丢失
```

**联合推理**通过互斥约束避免此问题：B 已被标记为 obj_id=2，obj_id=1 的 tracker 不会抢占 B 的像素。

---

## 3. Pipeline 架构

### 3.1 整体流程

```
┌─────────────────────────────────────────────────────────┐
│                   用户交互层（Gradio）                     │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Step 1: 帧级标注                                        │
│  ┌──────────────────────────────────────────┐           │
│  │ SAM2 Image Predictor                     │           │
│  │ - 用户在任意帧点击标注                      │           │
│  │ - 每个标注关联一个 object_id               │           │
│  │ - 输出: {(frame_idx, obj_id): mask}       │           │
│  └──────────────────────────────────────────┘           │
│                        ↓                                │
│  Step 2: 多目标传播（双向）                                │
│  ┌──────────────────────────────────────────┐           │
│  │ SAM2 Video Predictor                     │           │
│  │ - 联合推理所有 object                      │           │
│  │ - 双向传播: 正向 + 反向覆盖全视频           │           │
│  │ - 输出: 帧 0 ~ N 每帧每个 object 的 mask   │           │
│  └──────────────────────────────────────────┘           │
│                        ↓                                │
│  Step 3: 用户审核 + 补标（交互式迭代）                     │
│  ┌──────────────────────────────────────────┐           │
│  │ - 预览传播结果                             │           │
│  │ - 发现跟丢 → 在该帧补标 → 重新传播          │           │
│  │ - 满意后进入下一步                          │           │
│  └──────────────────────────────────────────┘           │
│                        ↓                                │
│  Step 4: Mask 合并 + Alpha Matting                      │
│  ┌──────────────────────────────────────────┐           │
│  │ - 合并: combined[t] = obj1[t] | obj2[t]  │           │
│  │ - 策略 C 采样注入帧（变化检测+最小间隔）     │           │
│  │ - MatAnyone 从帧 0 开始 → 全视频 alpha    │           │
│  └──────────────────────────────────────────┘           │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### 3.2 SAM2 Video Predictor 多目标用法

```python
# SAM2 VP 原生支持在同一 session 中注册多个 object
predictor = SAM2VideoPredictor(...)
state = predictor.init_state(video_path=video_frames_dir)

# 在不同帧注册不同 object
predictor.add_new_mask(
    state, frame_idx=0, obj_id=1, mask=person_a_mask
)
predictor.add_new_mask(
    state, frame_idx=10, obj_id=2, mask=person_b_mask
)

# 联合传播：模型同时跟踪所有已注册的 object
for frame_idx, obj_ids, masks in predictor.propagate_in_video(state):
    # masks: 每个 obj_id 对应一个 binary mask
    # 模型内部通过互斥约束避免身份漂移
    per_frame_masks[frame_idx] = masks
```

### 3.3 Mask 合并策略

SAM2 VP 输出 per-object mask 后，合并为 MatAnyone 可用的单通道前景 mask：

```python
def merge_masks(per_object_masks: dict[int, np.ndarray]) -> np.ndarray:
    """将多个 object 的 mask 合并为单个前景 mask。"""
    combined = np.zeros_like(list(per_object_masks.values())[0])
    for mask in per_object_masks.values():
        combined = np.maximum(combined, mask)
    return combined
```

### 3.4 MatAnyone Keyframe 采样策略

SAM2 VP 输出全视频每帧的 combined mask 后，需要决定在哪些帧注入 MatAnyone。

**采用策略 C：变化检测采样 + 最小间隔约束。**

#### 为什么不用固定间隔采样（策略 B）

```
固定每 30 帧注入:

帧 0:   注入（A）
帧 1-29: 自由传播
帧 15:  B 突然入画 ← MatAnyone 不知道 B 的存在
帧 16-29: B 被当成背景，连续 14 帧错误
帧 30:  注入（A+B）← 终于校准，但前面已经错了
```

固定间隔不关心内容变化，会漏掉关键时刻（人物进出画面），也会在无变化时浪费注入。

#### 策略 C：变化检测 + 最小间隔

核心逻辑：当相邻帧的 combined mask 变化超过阈值时注入，但两次注入之间至少间隔 `min_gap` 帧。

```python
def sample_injection_frames(
    combined_masks: list[np.ndarray],
    iou_threshold: float = 0.9,
    min_gap: int = 5,
) -> list[int]:
    """选择需要注入 MatAnyone 的帧索引。"""
    injection_frames = [0]  # 首帧必须注入
    last_inject = 0

    for t in range(1, len(combined_masks)):
        iou = compute_iou(combined_masks[t], combined_masks[t - 1])
        gap = t - last_inject

        if iou < iou_threshold and gap >= min_gap:
            injection_frames.append(t)
            last_inject = t

    return injection_frames
```

效果：

```
帧 0:    注入（A）
帧 1-14: IoU ≈ 0.98 → 不注入，MatAnyone 自由传播，时序平滑
帧 15:   B 入画，IoU = 0.6 → 立刻注入（A+B）← 及时校准
帧 16-44: IoU ≈ 0.97 → 不注入
帧 45:   A 离开，IoU = 0.5 → 立刻注入（B）
```

**该注入的时候注入，不该注入的时候让 MatAnyone 保持时序连续性。**

`min_gap` 的作用：防止剧烈运动场景（如跳舞）下每帧都注入，退化成"每帧重置记忆"导致 alpha 时序不平滑。

#### 默认参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `iou_threshold` | 0.9 | IoU 低于此值触发注入 |
| `min_gap` | 5 | 两次注入之间最少间隔帧数 |

---

## 4. 文件规划

### 新增文件

| 文件 | 职责 | 预估行数 |
|------|------|---------|
| `engines/sam2_video_engine.py` | SAM2 Video Predictor 封装，多目标联合传播 | ~100 |
| `pipeline/mask_propagation.py` | 多目标 mask 传播 + 合并 + keyframe 采样 | ~80 |

### 修改文件

| 文件 | 改动内容 |
|------|---------|
| `app.py` | 新增 UI 元素：object 管理、传播预览、补标流程 |
| `app_callbacks.py` | 新增回调：object 切换、传播执行、mask 审核 |
| `config.py` | 新增 SAM2 VP 相关配置 |
| `scripts/download_models.py` | SAM2 VP 与 Image Predictor 共用同一 checkpoint，无需额外下载 |

### 文件依赖

```
app.py / app_callbacks.py
├── engines/sam2_engine.py           （现有，单帧标注）
├── engines/sam2_video_engine.py     （新增，多目标传播）
├── engines/matanyone_engine.py      （现有，alpha matting）
├── pipeline/mask_propagation.py     （新增，mask 合并 + 采样）
└── pipeline/video_io.py             （现有）
```

---

## 5. UI 交互流程设计

### 5.1 标注阶段（已有 + 扩展）

```
┌─────────────────────────────────────────────────┐
│ Object 管理栏                                    │
│ [Object 1 (A)] [Object 2 (B)] [+ New Object]    │
├─────────────────────────────────────────────────┤
│                                                  │
│  帧画面 + mask overlay                            │
│  （当前选中的 object 高亮显示）                     │
│                                                  │
├─────────────────────────────────────────────────┤
│ Frame: [═══════●════════════════]                 │
│                                                  │
│ ● Positive  ○ Negative  [Undo] [Clear]           │
│ [Save Keyframe for Current Object]               │
└─────────────────────────────────────────────────┘
```

用户操作：
1. 选择 / 创建 Object（如 "Object 1"）
2. 导航到目标帧
3. 点击标注 → SAM2 Image Predictor 生成 mask
4. 保存为该 Object 在该帧的 keyframe
5. 切换到另一个 Object，在另一帧重复标注

### 5.2 传播阶段（新增）

```
┌─────────────────────────────────────────────────┐
│ [Run Propagation]                                │
│                                                  │
│ 传播进度: [████████░░░░░░] 60%                    │
│                                                  │
│ 传播预览:                                         │
│ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐        │
│ │ #0  │ │ #10 │ │ #20 │ │ #30 │ │ #40 │        │
│ │ A   │ │ B   │ │ A+B │ │ A+B │ │ A   │        │
│ └─────┘ └─────┘ └─────┘ └─────┘ └─────┘        │
│                                                  │
│ ⚠ 帧 35-40 可能存在跟踪漂移，建议检查             │
│ [Add Correction at Frame 38] [Re-propagate]      │
└─────────────────────────────────────────────────┘
```

### 5.3 Matting 阶段（已有，微调）

```
传播完成 → 自动合并 masks → 采样 keyframes → MatAnyone → alpha + foreground
```

---

## 6. 关键技术决策

### Q1: SAM2 VP 的 Image Predictor 和 Video Predictor 共用 checkpoint 吗？

是的。`sam2.1_hiera_large.pt` 同时支持 Image Predictor 和 Video Predictor，无需额外下载模型。

### Q2: Video Predictor 的传播方向？支持双向传播吗？

**支持双向传播。** SAM2 VP 的 `propagate_in_video()` 提供 `reverse` 参数：

```python
# sam2/sam2_video_predictor.py:546-581
def propagate_in_video(
    self,
    inference_state,
    start_frame_idx=None,
    max_frame_num_to_track=None,
    reverse=False,          # ← 控制传播方向
):
    if reverse:
        # 反向：从 start_frame_idx 往前传播到 frame 0
        end_frame_idx = max(start_frame_idx - max_frame_num_to_track, 0)
        processing_order = range(start_frame_idx, end_frame_idx - 1, -1)
    else:
        # 正向：从 start_frame_idx 往后传播到最后一帧
        end_frame_idx = min(start_frame_idx + max_frame_num_to_track, num_frames - 1)
        processing_order = range(start_frame_idx, end_frame_idx + 1)
```

默认行为（`reverse=False`）：从最早的标注帧向后传播。`start_frame_idx` 默认取所有标注帧中最小的那个。

**双向传播用法：调用两次 `propagate_in_video()`。**

```python
# 假设在帧 10 标注了 A，在帧 30 标注了 B

# Pass 1: 正向传播（帧 10 → 末尾）
for frame_idx, obj_ids, masks in predictor.propagate_in_video(state):
    video_segments[frame_idx] = masks

# Pass 2: 反向传播（帧 10 → 帧 0）
for frame_idx, obj_ids, masks in predictor.propagate_in_video(state, reverse=True):
    video_segments[frame_idx] = masks
```

注意：反向传播时如果 `start_frame_idx=0`，会跳过（因为已经在起点，无法再往前）。

**本场景的策略：始终双向传播，确保全视频覆盖。**

无论用户在哪一帧标注，SAM2 VP 通过双向传播可以生成帧 0 ~ 帧 N 的完整 per-frame mask。MatAnyone 始终从帧 0 开始正向 matting，使用 SAM2 VP 在帧 0 生成的 combined mask 作为首帧输入。

```
用户标注帧 10（A）、帧 30（B）
        ↓
SAM2 VP 反向传播: 帧 10 → 帧 0   → combined[0] ~ combined[9]
SAM2 VP 正向传播: 帧 10 → 末尾    → combined[10] ~ combined[N]
        ↓
全视频每帧都有 combined mask
        ↓
MatAnyone 从帧 0 开始（用 combined[0]），正向处理全视频
        ↓
输出: 帧 0 ~ N 的完整 alpha matte
```

这样用户不需要关心"标注在哪一帧"的问题，标注任意帧都能得到全视频的 matting 结果。

实现时默认总是执行双向传播：

```python
# 正向传播
for frame_idx, obj_ids, masks in predictor.propagate_in_video(state):
    video_segments[frame_idx] = masks

# 反向传播（覆盖标注帧之前的帧）
for frame_idx, obj_ids, masks in predictor.propagate_in_video(state, reverse=True):
    video_segments[frame_idx] = masks
```

### Q3: 身份漂移仍然发生怎么办？

联合推理 + 互斥约束能减少但无法完全消除漂移。应对策略：
- 提供传播结果预览，让用户快速定位问题帧
- 用户在问题帧补标 → 重新传播（交互式迭代）
- 实际使用中，大部分场景 2-3 轮迭代即可覆盖全视频

### Q4: MatAnyone 注入频率？

采用**策略 C（变化检测 + 最小间隔）**，详见 3.4 节。

核心逻辑：当相邻帧 combined mask 的 IoU 低于阈值（默认 0.9）时注入，两次注入之间至少间隔 `min_gap` 帧（默认 5）。这样在人物进出画面等关键时刻能及时校准，静态场景则让 MatAnyone 自由传播以保持时序平滑。

### Q5: 内存和性能考量？

SAM2 Video Predictor 需要维护 memory bank，视频越长内存越大。应对：
- 限制最大视频帧数（当前已有 `MAX_VIDEO_DURATION` 配置）
- SAM2 VP 内部有 memory bank 大小限制，超出后自动淘汰旧记忆

---

## 7. 实现计划

### Phase 1: SAM2 Video Engine 封装

- 封装 `SAM2VideoPredictor`，支持多 object 注册和联合传播
- 独立验证：标注 2 个 object → 传播 → 检查输出 mask 正确性

### Phase 2: Mask 传播 Pipeline

- 实现 mask 合并逻辑（per-object → combined）
- 实现 keyframe 采样策略（固定间隔 / 变化检测）
- 独立验证：给定 per-object masks → 输出采样后的 combined keyframes

### Phase 3: UI 扩展

- 新增 Object 管理 UI（创建 / 切换 / 删除 object）
- 新增传播触发和预览
- 新增补标 + 重新传播流程

### Phase 4: 端到端集成

- 标注 → 传播 → 审核 → matting 全流程打通
- 端到端测试：多人物视频完整工作流

---

## 8. 与当前实现的兼容性

新 pipeline 是当前实现的**超集**：

| 场景 | 当前实现 | 新 pipeline |
|------|---------|------------|
| 单人物、单 keyframe | 直接 MatAnyone | 跳过传播步骤，行为不变 |
| 单人物、多 keyframe | 直接 MatAnyone | 跳过传播步骤，行为不变 |
| 多人物 | 不支持 | SAM2 VP 传播 → MatAnyone |

当只有一个 object 且首个 keyframe 在帧 0 时，可以跳过 SAM2 VP 传播步骤，直接使用现有的 MatAnyone 多 keyframe 注入逻辑，保持向后兼容。

当标注帧不在帧 0 时（即使只有一个 object），也建议走 SAM2 VP 双向传播流程，这样 MatAnyone 可以从帧 0 开始处理，输出完整的全视频 alpha matte。

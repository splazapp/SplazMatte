"""SplazMatte — Gradio web app for MatAnyone matting with SAM2/SAM3 multi-frame annotation."""

import os
# MPS (Apple Silicon) lacks some ops used by SAM2/MatAnyone — fall back to CPU for those.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

from dotenv import load_dotenv
load_dotenv()

import logging

import gradio as gr

from config import (
    DEFAULT_DILATE,
    DEFAULT_ERODE,
    GRADIO_SERVER_PORT,
    PROCESSING_LOG_FILE,
    VIDEOMAMA_BATCH_SIZE,
    VIDEOMAMA_OVERLAP,
    VIDEOMAMA_SEED,
)
from app_callbacks import (
    empty_state,
    keyframe_gallery,
    on_clear_clicks,
    on_delete_keyframe,
    on_frame_click,
    on_model_change,
    on_run_propagation,
    on_save_keyframe,
    on_slider_change,
    on_start_matting,
    on_text_prompt,
    on_undo_click,
    on_upload,
)
from app_queue_ui import build_queue_section
from queue_callbacks import (
    on_add_to_queue,
    on_execute_queue,
    on_remove_from_queue,
    on_restore_from_queue,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)

# Write processing logs to file for the Gradio UI log panel
PROCESSING_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
PROCESSING_LOG_FILE.touch()
_file_handler = logging.FileHandler(str(PROCESSING_LOG_FILE), mode="w")
_file_handler.setFormatter(logging.Formatter(
    "%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%H:%M:%S",
))
logging.getLogger().addHandler(_file_handler)
logging.getLogger(__name__).info("SplazMatte 已启动，等待操作...")


def _read_processing_log() -> str:
    """Read the processing log file for display in the UI."""
    if PROCESSING_LOG_FILE.exists():
        return PROCESSING_LOG_FILE.read_text()
    return ""


def build_app() -> gr.Blocks:
    """Construct the Gradio Blocks application."""
    with gr.Blocks(title="SplazMatte") as app:
        gr.Markdown("# SplazMatte")
        gr.Markdown(
            "上传视频，在关键帧上标注遮罩，"
            "然后选择抠图引擎进行抠像。"
        )

        session_state = gr.State(empty_state())
        queue_state = gr.State([])

        with gr.Row():
            model_selector = gr.Radio(
                choices=["SAM2", "SAM3"],
                value="SAM2",
                label="标注模型",
                info="SAM3 支持文本提示词，但模型更大、加载更慢",
            )

        # ── Row 1: Upload + Frame annotation side by side ──
        with gr.Row(equal_height=True):
            with gr.Column(scale=1):
                video_input = gr.Video(
                    label="上传视频",
                    height=400,
                )
            with gr.Column(scale=2):
                frame_display = gr.Image(
                    label="帧预览",
                    type="numpy", interactive=False, height=400,
                )
                with gr.Row(visible=False) as text_prompt_row:
                    text_prompt_input = gr.Textbox(
                        label="文本提示词",
                        placeholder="输入描述，如：person, car, dog...",
                        scale=3,
                    )
                    text_prompt_btn = gr.Button(
                        "检测", variant="primary", scale=1,
                    )

        # ── Row 2: Slider + frame label ──
        with gr.Row():
            frame_slider = gr.Slider(
                minimum=0, maximum=1, step=1, value=0,
                label="帧序号",
                info="拖动滑块可在不同帧之间切换",
                visible=False, interactive=True,
            )
            frame_label = gr.Markdown("请先上传视频。")

        # ── Row 3: Click controls + keyframe actions ──
        with gr.Row():
            point_mode = gr.Radio(
                choices=["Positive", "Negative"],
                value="Positive",
                label="点击模式",
                info="Positive 标记前景区域，Negative 标记背景区域",
                scale=2,
            )
            undo_btn = gr.Button("撤销点击", scale=1)
            clear_btn = gr.Button("清除所有点", scale=1)
            save_kf_btn = gr.Button(
                "保存为关键帧", variant="primary", scale=1,
            )
            delete_kf_btn = gr.Button("删除关键帧", scale=1)

        # ── Row 4: Keyframe gallery (full width) ──
        keyframe_info = gr.Markdown("尚未保存任何关键帧。")
        kf_gallery = gr.Gallery(
            label="关键帧列表",
            columns=4,
            object_fit="contain",
            height=400,
        )

        # ── Row 5: SAM2 传播按钮 | 传播预览 ──
        with gr.Row(equal_height=True):
            propagate_btn = gr.Button(
                "运行传播", variant="secondary", scale=1,
            )
            propagation_preview = gr.Video(
                label="传播预览（右上角标注帧序号）",
                height=400, scale=2,
            )

        # ── Row 6: 参数设置 | 开始抠像 + 添加到队列 ──
        with gr.Row(equal_height=True):
            with gr.Accordion("参数设置", open=True):
                matting_engine_selector = gr.Radio(
                    choices=["MatAnyone", "VideoMaMa"],
                    value="MatAnyone",
                    label="抠图引擎",
                    info="MatAnyone 适合人物（快速），VideoMaMa 适合通用场景（需先运行传播）",
                )
                erode_slider = gr.Slider(
                    minimum=0, maximum=30, step=1, value=DEFAULT_ERODE,
                    label="腐蚀核大小",
                    info="增大可收缩遮罩边缘，去除毛刺",
                )
                dilate_slider = gr.Slider(
                    minimum=0, maximum=30, step=1, value=DEFAULT_DILATE,
                    label="膨胀核大小",
                    info="增大可扩展遮罩边缘，保留更多细节",
                )
                vm_batch_slider = gr.Slider(
                    minimum=4, maximum=128, step=4,
                    value=VIDEOMAMA_BATCH_SIZE,
                    label="批次大小",
                    info="每批推理帧数，越大越快但占用更多显存",
                    visible=False,
                )
                vm_overlap_slider = gr.Slider(
                    minimum=0, maximum=8, step=1,
                    value=VIDEOMAMA_OVERLAP,
                    label="重叠帧数",
                    info="批次间重叠帧数，用于平滑过渡",
                    visible=False,
                )
                vm_seed_input = gr.Number(
                    value=VIDEOMAMA_SEED,
                    label="随机种子",
                    info="固定种子可复现结果",
                    precision=0,
                    visible=False,
                )
            with gr.Column(scale=1, min_width=160):
                matting_btn = gr.Button("开始抠像", variant="primary")
                add_queue_btn = gr.Button("添加到队列", variant="secondary")

        # ── Row 6: Output videos + Processing log ──
        with gr.Row(equal_height=True):
            alpha_output = gr.Video(
                label="Alpha 通道视频",
                height=300,
            )
            fgr_output = gr.Video(
                label="前景视频",
                height=300,
            )
            with gr.Column(scale=1):
                log_display = gr.Textbox(
                    value=_read_processing_log,
                    every=0.5,
                    label="处理日志",
                    lines=8,
                    max_lines=20,
                    interactive=False,
                    autoscroll=True,
                )

        # ── Row 8: Task queue ──
        queue_ui = build_queue_section()

        # ── Event wiring ──

        video_input.upload(
            fn=on_upload,
            inputs=[video_input, session_state],
            outputs=[
                session_state, frame_display, frame_slider,
                frame_label, keyframe_info, kf_gallery,
            ],
        )

        frame_slider.change(
            fn=on_slider_change,
            inputs=[frame_slider, session_state],
            outputs=[frame_display, session_state, frame_label],
        )

        frame_display.select(
            fn=on_frame_click,
            inputs=[point_mode, model_selector, session_state],
            outputs=[frame_display, session_state],
        )

        undo_btn.click(
            fn=on_undo_click,
            inputs=[model_selector, session_state],
            outputs=[frame_display, session_state],
        )

        clear_btn.click(
            fn=on_clear_clicks,
            inputs=[session_state],
            outputs=[frame_display, session_state],
        )

        save_kf_btn.click(
            fn=on_save_keyframe,
            inputs=[session_state],
            outputs=[session_state, keyframe_info, kf_gallery],
        )

        delete_kf_btn.click(
            fn=on_delete_keyframe,
            inputs=[session_state],
            outputs=[session_state, frame_display, keyframe_info, kf_gallery],
        )

        model_selector.change(
            fn=on_model_change,
            inputs=[model_selector, session_state],
            outputs=[session_state, text_prompt_row, frame_display],
        )

        text_prompt_btn.click(
            fn=on_text_prompt,
            inputs=[text_prompt_input, session_state],
            outputs=[frame_display, session_state],
        )

        propagate_btn.click(
            fn=on_run_propagation,
            inputs=[model_selector, session_state],
            outputs=[propagation_preview, session_state],
        )

        matting_engine_selector.change(
            fn=lambda engine: (
                gr.update(visible=(engine == "MatAnyone")),
                gr.update(visible=(engine == "MatAnyone")),
                gr.update(visible=(engine == "VideoMaMa")),
                gr.update(visible=(engine == "VideoMaMa")),
                gr.update(visible=(engine == "VideoMaMa")),
            ),
            inputs=[matting_engine_selector],
            outputs=[
                erode_slider, dilate_slider,
                vm_batch_slider, vm_overlap_slider, vm_seed_input,
            ],
        )

        matting_btn.click(
            fn=on_start_matting,
            inputs=[
                matting_engine_selector, erode_slider, dilate_slider,
                vm_batch_slider, vm_overlap_slider, vm_seed_input,
                model_selector, session_state,
            ],
            outputs=[alpha_output, fgr_output, session_state],
        )

        # ── Queue event wiring ──

        add_queue_btn.click(
            fn=on_add_to_queue,
            inputs=[
                matting_engine_selector, erode_slider, dilate_slider,
                vm_batch_slider, vm_overlap_slider, vm_seed_input,
                session_state, queue_state,
            ],
            outputs=[
                session_state, queue_state,
                queue_ui["queue_status"], queue_ui["queue_table"],
                frame_display, frame_slider, frame_label,
                keyframe_info, kf_gallery, video_input,
                propagation_preview, alpha_output, fgr_output,
            ],
            api_name=False,
        )

        queue_ui["restore_btn"].click(
            fn=on_restore_from_queue,
            inputs=[queue_ui["remove_idx"], session_state, queue_state],
            outputs=[
                session_state, queue_state,
                queue_ui["queue_status"], queue_ui["queue_table"],
                frame_display, frame_slider, frame_label,
                keyframe_info, kf_gallery, video_input,
                model_selector, text_prompt_row,
                matting_engine_selector,
                erode_slider, dilate_slider,
                vm_batch_slider, vm_overlap_slider, vm_seed_input,
                propagation_preview, alpha_output, fgr_output,
            ],
            api_name=False,
        )

        queue_ui["remove_btn"].click(
            fn=on_remove_from_queue,
            inputs=[queue_ui["remove_idx"], queue_state],
            outputs=[
                queue_state,
                queue_ui["queue_status"], queue_ui["queue_table"],
            ],
            api_name=False,
        )

        queue_ui["execute_btn"].click(
            fn=on_execute_queue,
            inputs=[queue_ui["queue_progress"], queue_state],
            outputs=[
                queue_state,
                queue_ui["queue_status"], queue_ui["queue_table"],
                queue_ui["queue_progress"],
            ],
            api_name=False,
        )

    return app


if __name__ == "__main__":
    app = build_app()
    app.queue()
    app.launch(server_name="0.0.0.0", server_port=GRADIO_SERVER_PORT, theme=gr.themes.Soft())

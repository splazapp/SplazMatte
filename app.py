"""SplazMatte — Gradio web app for MatAnyone matting with SAM2 multi-frame annotation."""

import os
# MPS (Apple Silicon) lacks some ops used by SAM2/MatAnyone — fall back to CPU for those.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

from dotenv import load_dotenv
load_dotenv()

import logging

import gradio as gr

from config import DEFAULT_DILATE, DEFAULT_ERODE, GRADIO_SERVER_PORT, PROCESSING_LOG_FILE
from app_callbacks import (
    empty_state,
    keyframe_gallery,
    on_clear_clicks,
    on_delete_keyframe,
    on_frame_click,
    on_run_propagation,
    on_save_keyframe,
    on_slider_change,
    on_start_matting,
    on_undo_click,
    on_upload,
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
            "上传视频，在关键帧上通过 SAM2 标注遮罩，"
            "然后使用 MatAnyone 进行抠像。"
        )

        session_state = gr.State(empty_state())

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

        # ── Row 2: Slider + frame label ──
        with gr.Row():
            frame_slider = gr.Slider(
                minimum=0, maximum=1, step=1, value=0,
                label="帧序号",
                info="拖动滑块可在不同帧之间切换",
                visible=False, interactive=True,
            )
            frame_label = gr.Markdown(
                "请先上传视频。", scale=0, min_width=160,
            )

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

        # ── Row 4: Keyframe gallery + Settings / Matting ──
        with gr.Row(equal_height=True):
            with gr.Column(scale=2):
                keyframe_info = gr.Markdown("尚未保存任何关键帧。")
                kf_gallery = gr.Gallery(
                    label="关键帧列表",
                    columns=4,
                    object_fit="contain",
                    height=200,
                )
            with gr.Column(scale=1):
                with gr.Accordion("参数设置", open=True):
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
                matting_btn = gr.Button("开始抠像", variant="primary")

        # ── Row 5: SAM2 VP Propagation ──
        with gr.Row(equal_height=True):
            with gr.Column(scale=1):
                propagate_btn = gr.Button(
                    "运行 SAM2 传播", variant="secondary",
                )
                gr.Markdown(
                    "基于已保存的关键帧，通过 SAM2 Video Predictor "
                    "双向传播生成全视频逐帧遮罩预览。"
                )
            with gr.Column(scale=2):
                propagation_preview = gr.Video(
                    label="传播预览（右上角标注帧序号）",
                    height=400,
                )

        # ── Processing Log ──
        with gr.Accordion("处理日志", open=False):
            log_display = gr.Textbox(
                value=_read_processing_log,
                every=0.5,
                label="处理日志",
                lines=8,
                max_lines=20,
                interactive=False,
                autoscroll=True,
            )

        # ── Row 5: Output videos ──
        with gr.Row(equal_height=True):
            alpha_output = gr.Video(
                label="Alpha 通道视频",
                height=300,
            )
            fgr_output = gr.Video(
                label="前景视频",
                height=300,
            )

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
            inputs=[point_mode, session_state],
            outputs=[frame_display, session_state],
        )

        undo_btn.click(
            fn=on_undo_click,
            inputs=[session_state],
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

        propagate_btn.click(
            fn=on_run_propagation,
            inputs=[session_state],
            outputs=[propagation_preview, session_state],
        )

        matting_btn.click(
            fn=on_start_matting,
            inputs=[erode_slider, dilate_slider, session_state],
            outputs=[alpha_output, fgr_output, session_state],
        )

    return app


if __name__ == "__main__":
    app = build_app()
    app.queue()
    app.launch(server_port=GRADIO_SERVER_PORT, theme=gr.themes.Soft())

"""SplazMatte — Gradio web app for MatAnyone matting with SAM2/SAM3 multi-frame annotation."""

import os
# MPS (Apple Silicon) lacks some ops used by SAM2/MatAnyone — fall back to CPU for those.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
# Prevent system proxies (Clash, V2Ray, etc.) from intercepting Gradio's local self-check.
os.environ.setdefault("no_proxy", "localhost,127.0.0.1,0.0.0.0")

from dotenv import load_dotenv
load_dotenv()

import logging
import platform
from logging.handlers import RotatingFileHandler

import gradio as gr

from config import (
    DEFAULT_DILATE,
    DEFAULT_ERODE,
    GRADIO_SERVER_PORT,
    LOGS_DIR,
    PROCESSING_LOG_FILE,
    VIDEOMAMA_BATCH_SIZE,
    VIDEOMAMA_OVERLAP,
    VIDEOMAMA_SEED,
    get_device,
)
from app_callbacks import (
    keyframe_gallery,
    on_clear_clicks,
    on_delete_keyframe,
    on_frame_click,
    on_model_change,
    on_refresh_sessions,
    on_restore_session,
    on_run_propagation,
    on_save_keyframe,
    on_slider_change,
    on_start_matting,
    on_text_prompt,
    on_undo_click,
    on_upload,
)
from session_store import empty_state, list_sessions
from app_queue_ui import build_queue_section
from utils.feishu_notify import send_feishu_startup
from queue_callbacks import (
    on_add_to_queue,
    on_clear_queue,
    on_execute_queue,
    on_load_queue,
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

# --- logs/ 持久日志 ---
LOGS_DIR.mkdir(parents=True, exist_ok=True)
_persistent_fmt = logging.Formatter(
    "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

_persistent_handler = RotatingFileHandler(
    str(LOGS_DIR / "splazmatte.log"),
    maxBytes=5_000_000, backupCount=5, encoding="utf-8",
)
_persistent_handler.setFormatter(_persistent_fmt)
logging.getLogger().addHandler(_persistent_handler)

_error_handler = RotatingFileHandler(
    str(LOGS_DIR / "errors.log"),
    maxBytes=2_000_000, backupCount=3, encoding="utf-8",
)
_error_handler.setLevel(logging.ERROR)
_error_handler.setFormatter(_persistent_fmt)
logging.getLogger().addHandler(_error_handler)


def _read_processing_log() -> str:
    """Read the processing log file for display in the UI."""
    if PROCESSING_LOG_FILE.exists():
        return PROCESSING_LOG_FILE.read_text()
    return ""


def build_app() -> gr.Blocks:
    """Construct the Gradio Blocks application."""
    with gr.Blocks(title="SplazMatte") as app:
        device = get_device()
        device_label = {"cuda": "CUDA (GPU)", "mps": "MPS (Apple Silicon)", "cpu": "CPU"}
        hostname = platform.node() or "unknown"
        gr.Markdown("# SplazMatte")
        gr.Markdown(
            f"运行设备: **{device_label.get(device.type, device.type)}** | "
            f"主机: **{hostname}**"
        )
        gr.Markdown(
            "上传视频，在关键帧上标注遮罩，"
            "然后选择抠图引擎进行抠像。"
        )

        session_state = gr.State(empty_state())
        queue_state = gr.State([])

        # ── Step 1: 上传视频 / 恢复 Session ──
        with gr.Accordion("Step 1: 上传视频 / 恢复 Session", open=True):
            with gr.Row():
                with gr.Column(scale=2):
                    video_input = gr.Video(
                        label="上传视频",
                        height=300,
                    )
                with gr.Column(scale=1):
                    session_dropdown = gr.Dropdown(
                        choices=list_sessions(),
                        label="恢复 Session",
                        interactive=True,
                    )
                    with gr.Row():
                        refresh_sessions_btn = gr.Button(
                            "刷新", scale=0, min_width=50,
                        )
                        restore_session_btn = gr.Button(
                            "恢复", variant="secondary", scale=0, min_width=50,
                        )

        # ── Step 2: 标注关键帧 ──
        with gr.Accordion("Step 2: 标注关键帧", open=True):
            with gr.Row():
                model_selector = gr.Radio(
                    choices=["SAM2", "SAM3"],
                    value="SAM2",
                    label="标注模型",
                    info="SAM3 支持文本提示词，但模型更大、加载更慢",
                    scale=1,
                )
                point_mode = gr.Radio(
                    choices=["Positive", "Negative"],
                    value="Positive",
                    label="点击模式",
                    info="Positive 标记前景区域，Negative 标记背景区域",
                    scale=1,
                )
                undo_btn = gr.Button("撤销点击", scale=0)
                clear_btn = gr.Button("清除所有点", scale=0)
                save_kf_btn = gr.Button(
                    "保存为关键帧", variant="primary", scale=0,
                )
                delete_kf_btn = gr.Button("删除关键帧", scale=0)
            with gr.Row():
                with gr.Column(scale=2):
                    frame_display = gr.Image(
                        label="帧预览",
                        type="numpy", interactive=False, height=450,
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
                with gr.Column(scale=1):
                    keyframe_info = gr.Markdown("尚未保存任何关键帧。")
                    kf_gallery = gr.Gallery(
                        label="关键帧列表",
                        columns=4,
                        object_fit="contain",
                        height=450,
                    )
            with gr.Row():
                frame_slider = gr.Slider(
                    minimum=0, maximum=1, step=1, value=0,
                    label="帧序号",
                    info="拖动滑块可在不同帧之间切换",
                    visible=False, interactive=True,
                )
                frame_label = gr.Markdown("请先上传视频。")

        # ── Step 3: 传播 ──
        with gr.Accordion("Step 3: 传播", open=True):
            with gr.Row(equal_height=True):
                propagate_btn = gr.Button(
                    "运行传播", variant="secondary", scale=0,
                )
                propagation_preview = gr.Video(
                    label="传播预览（右上角标注帧序号）",
                    height=400, scale=2,
                )

        # ── Step 4: 抠像 ──
        with gr.Accordion("Step 4: 抠像", open=True):
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

        # ── Step 5: 任务队列 ──
        with gr.Accordion("Step 5: 任务队列", open=True):
            queue_ui = build_queue_section()

        # ── Event wiring ──

        video_input.upload(
            fn=on_upload,
            inputs=[video_input, session_state],
            outputs=[
                session_state, frame_display, frame_slider,
                frame_label, keyframe_info, kf_gallery,
                session_dropdown,
            ],
        )

        frame_slider.input(
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

        refresh_sessions_btn.click(
            fn=on_refresh_sessions,
            outputs=[session_dropdown],
        )

        restore_session_btn.click(
            fn=on_restore_session,
            inputs=[session_dropdown, session_state],
            outputs=[
                session_state, frame_display, frame_slider, frame_label,
                keyframe_info, kf_gallery, video_input,
                model_selector, text_prompt_row,
                propagation_preview,
                matting_engine_selector,
                erode_slider, dilate_slider,
                vm_batch_slider, vm_overlap_slider, vm_seed_input,
                alpha_output, fgr_output,
            ],
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

        queue_ui["clear_btn"].click(
            fn=on_clear_queue,
            inputs=[queue_state],
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

        # Refresh session dropdown every time a client connects
        app.load(fn=on_refresh_sessions, outputs=[session_dropdown])

        # Restore queue display on page load
        app.load(
            fn=on_load_queue,
            outputs=[queue_state, queue_ui["queue_status"], queue_ui["queue_table"]],
        )

    return app


if __name__ == "__main__":
    log = logging.getLogger(__name__)
    app = build_app()
    app.queue()
    log.info("Launching Gradio (port=%s)...", GRADIO_SERVER_PORT)
    _, local_url, _ = app.launch(
        server_name="0.0.0.0",
        server_port=GRADIO_SERVER_PORT,
        share=False,
        theme=gr.themes.Soft(),
        prevent_thread_lock=True,
    )
    log.info("Gradio launched — local=%s", local_url)
    send_feishu_startup(local_url)
    app.block_thread()

"""Gradio UI components for the task queue section."""

import gradio as gr


def build_queue_section() -> dict:
    """Build the queue UI section and return component references.

    Returns:
        Dict with keys: queue_status, queue_table, remove_idx,
        restore_btn, remove_btn, clear_btn, download_btn,
        download_file, execute_btn, queue_progress.
    """
    gr.Markdown("### 任务队列")
    queue_status = gr.Markdown("队列为空。")
    queue_table = gr.Dataframe(
        headers=["序号", "视频名", "帧数", "关键帧", "已传播", "引擎", "状态"],
        datatype=["number", "str", "number", "number", "str", "str", "str"],
        interactive=False,
        wrap=True,
    )
    with gr.Row():
        remove_idx = gr.Number(
            value=1, label="序号", precision=0, minimum=1, scale=1,
        )
        with gr.Column(scale=1):
            restore_btn = gr.Button("恢复编辑")
            remove_btn = gr.Button("移除")
            clear_btn = gr.Button("清空队列", variant="stop")
            execute_btn = gr.Button("执行队列", variant="primary")
            feishu_btn = gr.Button("发送飞书通知")
            download_btn = gr.Button("打包下载")
            download_file = gr.File(label="下载结果", interactive=False)
    queue_progress = gr.Markdown("")

    return {
        "queue_status": queue_status,
        "queue_table": queue_table,
        "remove_idx": remove_idx,
        "restore_btn": restore_btn,
        "remove_btn": remove_btn,
        "clear_btn": clear_btn,
        "download_btn": download_btn,
        "download_file": download_file,
        "feishu_btn": feishu_btn,
        "execute_btn": execute_btn,
        "queue_progress": queue_progress,
    }

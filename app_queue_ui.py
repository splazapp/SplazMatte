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
        with gr.Column(scale=1):
            gr.Markdown(
                "<small>"
                "**恢复编辑** 将指定序号任务恢复到编辑区<br>"
                "**移除** 从队列中删除指定序号任务<br>"
                "**清空队列** 移除所有任务<br>"
                "**重置状态** 将所有任务重置为 pending<br>"
                "**执行队列** 执行所有 pending 任务<br>"
                "**发送飞书通知** 为已完成任务重发通知<br>"
                "**打包下载** 将已完成结果打包为 zip"
                "</small>"
            )
        with gr.Column(scale=1):
            remove_idx = gr.Number(
                value=1, label="选择序号", precision=0, minimum=1,
            )
            restore_btn = gr.Button("恢复编辑")
            remove_btn = gr.Button("移除")
            clear_btn = gr.Button("清空队列", variant="stop")
            reset_status_btn = gr.Button("重置状态")
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
        "reset_status_btn": reset_status_btn,
        "execute_btn": execute_btn,
        "queue_progress": queue_progress,
    }

import gradio as gr

USER_BAR_CSS = """
#user-bar { display: flex; justify-content: flex-end; align-items: center; gap: 0.5rem; margin-bottom: 0.5rem; }
"""

def greet(request: gr.Request):
    return f"Hello, {request.username}!"

def get_user_display(request: gr.Request):
    """返回用于右上角展示的用户信息。"""
    return f"**{request.username}**" if request.username else ""

with gr.Blocks(css=USER_BAR_CSS) as demo:
    # 右上角：当前用户 + 登出
    with gr.Row(elem_id="user-bar"):
        user_label = gr.Markdown(show_label=False)
        logout_btn = gr.Button("Logout", link="/logout", size="sm")
    demo.load(get_user_display, None, user_label)

    out = gr.Textbox()
    btn = gr.Button("Who am I?")
    btn.click(greet, outputs=out)

with demo.route("设置", "/settings"):
    gr.Markdown("### 账户")
    gr.Button("登出", link="/logout", size="sm")

demo.launch(
    auth=[("alice", "password1"), ("bob", "password2")],
    auth_message="Please login to continue"
)
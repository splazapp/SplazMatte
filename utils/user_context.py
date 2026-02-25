"""用户上下文：通过 ContextVar 在日志中自动注入用户邮箱。"""

import logging
from contextvars import ContextVar

# 当前请求的用户邮箱，页面加载时设置
current_user_email: ContextVar[str] = ContextVar("current_user_email", default="-")


class UserEmailFilter(logging.Filter):
    """将当前用户邮箱注入到 LogRecord 中。"""

    def filter(self, record: logging.LogRecord) -> bool:
        record.user_email = current_user_email.get("-")
        return True


def set_current_user(email: str) -> None:
    """设置当前上下文的用户邮箱。在页面入口调用。"""
    current_user_email.set(email)

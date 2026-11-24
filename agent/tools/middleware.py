from typing import Callable

from langchain.agents import AgentState
from langchain.agents.middleware import (
    ModelRequest,
    before_model,
    dynamic_prompt,
    wrap_tool_call,
)
from langchain.tools.tool_node import ToolCallRequest
from langchain_core.messages import ToolMessage
from langgraph.runtime import Runtime
from langgraph.types import Command

from utils.logger_handler import logger
from utils.prompt_loader import load_report_prompts, load_system_prompts


@wrap_tool_call
def monitor_tool(
        #请求的数据封装
        request: ToolCallRequest,
        #执行的函数本身
        handler: Callable[[ToolCallRequest], ToolMessage | Command],
) -> ToolMessage | Command:     #工具执行路径
    logger.info(f"[tool monitor]执行工具：{request.tool_call['name']}")
    logger.info(f"[tool monitor]传入参数：{request.tool_call['args']}")
    try:
        result = handler(request)
        logger.info(f"[tool monitor]工具{request.tool_call['name']}调用成功")
        if request.tool_call['name'] == "fill_context_for_report":
            # 报告场景的“开关”：打开后 dynamic_prompt 会切换到报告提示词
            if request.runtime.context is not None:
                request.runtime.context["report"] = True
        return result
    except Exception as e:
        logger.error(f"工具{request.tool_call['name']}调用失败，原因{type(e).__name__}: {e}")
        raise e


@before_model
def log_before_model(
        state: AgentState,          #整个Agent智能体的状态记录
        runtime: Runtime            #记录了整个执行过程中的上下文信息
):     #在模型执行输出日志
    messages = state['messages']
    logger.info(f"[log_before_model]即将调用模型，带有{len(messages)}条消息")
    latest_message = messages[-1]
    content = latest_message.content if isinstance(latest_message.content, str) else str(latest_message.content)
    logger.debug(f"[log_before_model]{type(latest_message).__name__}: {content.strip()[:200]}")
    return None


@dynamic_prompt     #每一次在生成提示词之前调用此函数
def report_prompt_switch(request: ModelRequest):     #动态切换提示词
    context = request.runtime.context or {}
    is_report = context.get('report', False)
    if is_report:           #报告生成场景，返回报告生成提示词内容
        logger.info("[report_prompt_switch]已切换到报告生成提示词")
        return load_report_prompts()
    return load_system_prompts()

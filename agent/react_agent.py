from langchain.agents import create_agent

from agent.tools.agent_tools import (
    fetch_external_data,
    fill_context_for_report,
    get_current_month,
    get_user_id,
    get_user_location,
    get_weather,
    rag_summarize,
)
from agent.tools.middleware import log_before_model, monitor_tool, report_prompt_switch
from model.factory import chat_model
from utils.prompt_loader import load_system_prompts


class ReactAgent:
    def __init__(self):
        self.agent = create_agent(
            model=chat_model,
            system_prompt=load_system_prompts(),
            tools=[rag_summarize, get_weather, get_user_id, get_user_location,
                   get_current_month, fetch_external_data, fill_context_for_report],
            middleware=[monitor_tool, log_before_model, report_prompt_switch],
            context_schema=dict,     # 让 stream(context={"report": False}) 可用
        )

    def execute_stream(self, query: str):
        input_dict = {
            "messages": [
                {"role": "user", "content": query}
            ]
        }
        for chunk in self.agent.stream(input_dict, stream_mode="values",
                                       context={"report": False}):
            latest_message = chunk["messages"][-1]
            content = latest_message.content
            # 只把模型输出的文字流式吐出来（工具消息、空消息不输出）
            if getattr(latest_message, "type", "") == "ai" and isinstance(content, str) and content.strip():
                yield content.strip() + "\n"

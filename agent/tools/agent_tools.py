import csv
import os
import random

from langchain_core.tools import tool

from rag.rag_service import RagSummarizeService
from utils.cofig_handler import agent_config
from utils.logger_handler import logger
from utils.path_tool import get_abs_path

rag = RagSummarizeService()
user_ids = ["1001", "1002", "1003", "1004", "1005", "1006", "1007", "1008", "1009", "1010"]
month_arr = ["2025-01", "2025-02", "2025-03", "2025-04", "2025-05", "2025-06",
             "2025-07", "2025-08", "2025-09", "2025-10", "2025-11", "2025-12"]
external_data = {}


@tool(description="从向量存储中检索参考资料")
def rag_summarize(query: str) -> str:
    return rag.rag_summarize(query)


@tool(description="获取指定城市的天气，以消息字符串的形式返回")
def get_weather(city: str) -> str:
    return f"城市{city}天气为晴天，气温26摄氏度，空气湿度50%，南风1级，AQI21，最近6小时降雨概率极低"


@tool(description="获取用户所在城市的名称，以纯字符串形式返回")
def get_user_location() -> str:
    return random.choice(["深圳", "合肥", "杭州"])


@tool(description="获取用户的ID，以纯字符串形式返回")
def get_user_id() -> str:
    return random.choice(user_ids)


@tool(description="获取当前月份，以纯字符串形式返回")
def get_current_month() -> str:
    return random.choice(month_arr)


def generate_external_data():
    if not external_data:
        external_data_path = get_abs_path(agent_config["external_data_path"])
        if not os.path.exists(external_data_path):
            raise FileNotFoundError(f"外部数据文件{external_data_path}不存在")
        # 文件带 BOM，用 utf-8-sig；用 csv 模块解析，字段里有引号和逗号也不会切错
        with open(external_data_path, "r", encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                uid = row["用户ID"].strip()
                month = row["时间"].strip()
                external_data.setdefault(uid, {})[month] = {
                    "特征": row["特征"].strip(),
                    "效率": row["清洁效率"].strip(),
                    "耗材": row["耗材"].strip(),
                    "对比": row["对比"].strip(),
                }


@tool(description="从外部系统中获取用户在指定月份的使用记录，以纯字符串形式返回，如果检索到返回空字符串")
def fetch_external_data(user_id: str, month: str) -> str:
    generate_external_data()
    try:
        return str(external_data[user_id][month])
    except KeyError:
        logger.warning(f"[fetch_external_data]未能检索用户:{user_id}在{month}的使用记录")
        return ""


@tool(description="无入参，无返回值，调用后触发中间件自动为报告生成的场景动态注入上下文信息，为后续提示词切换上下文信息")
def fill_context_for_report() -> str:
    return "fill_context_for_report已调用"

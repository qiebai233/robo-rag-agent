"""
yaml
k: v
"""
import yaml
from utils.path_tool import get_abs_path


def load_rag_config(config_path: str = get_abs_path("config/rag.yml"), encoding: str = 'utf-8'):
    with open(config_path, 'r', encoding=encoding) as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def load_chroma_config(config_path: str = get_abs_path("config/chroma.yml"), encoding: str = 'utf-8'):
    with open(config_path, 'r', encoding=encoding) as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def load_prompts_config(config_path: str = get_abs_path("config/prompt.yml"), encoding: str = 'utf-8'):
    with open(config_path, 'r', encoding=encoding) as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def load_agent_config(config_path: str = get_abs_path("config/agent.yml"), encoding: str = 'utf-8'):
    with open(config_path, 'r', encoding=encoding) as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def load_pdf_config(config_path: str = get_abs_path("config/pdf.yml"), encoding: str = 'utf-8'):
    """PDF 解析配置（图片/表格增强）。

    这个文件是新加的，为了兼容“没有这个配置文件”的老项目，读不到时返回空字典，
    代码里全部用默认值处理（等价于开启增强解析）。
    """
    try:
        with open(config_path, 'r', encoding=encoding) as f:
            return yaml.load(f, Loader=yaml.FullLoader) or {}
    except FileNotFoundError:
        return {}


rag_config = load_rag_config()
chroma_config = load_chroma_config()
prompt_config = load_prompts_config()
agent_config = load_agent_config()
pdf_config = load_pdf_config()

if __name__ == '__main__':
    print(rag_config["chat_model_name"])
    print("pdf_config =", pdf_config)

import hashlib
import os

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document

from utils.logger_handler import logger


def get_file_md5_hex(filepath: str):         #获取文件的md5 16进制字符串
    if not os.path.exists(filepath):
        logger.error(f"[md5计算]文件{filepath}不存在")
        return None
    if not os.path.isfile(filepath):
        logger.error(f"[md5计算]文件{filepath}不是文件")
        return None
    md5_obj = hashlib.md5()
    chunk_size = 4096       #4kb分片，避免文件过大爆内存
    try:
        with open(filepath, 'rb') as f:       #必须二进制读取
            while chunk := f.read(chunk_size):
                md5_obj.update(chunk)
    except Exception as e:
        logger.error(f"计算文件{filepath}MD5失败，{str(e)}")
        return None
    return md5_obj.hexdigest()


def listdir_with_allowed_type(path: str, allowed_types: tuple[str, ...]):        #返回文件夹内（含子文件夹）的文件列表
    files = []
    if not os.path.isdir(path):
        logger.error(f"[listdir_with_allowed_type]{path}不是文件夹")
        return files
    for root_dir, _, filenames in os.walk(path):
        for filename in filenames:
            if filename.endswith(tuple(allowed_types)):    #判断结尾是否符合
                files.append(os.path.join(root_dir, filename))
    return files


def pdf_loader(filepath: str, passwd=None) -> list[Document]:
    """PDF 解析。

    升级后走 utils/pdf_parser.py：正文 + 表格(Markdown) + 图片(通义千问VL 识别成文字)，
    三类内容都会进入向量库；如果 pdf_config 里 enabled: false，则退回原来的纯文字解析。
    """
    from utils.cofig_handler import pdf_config

    if (pdf_config or {}).get("enabled", True):
        try:
            from utils.pdf_parser import parse_pdf
            documents = parse_pdf(filepath)
            if documents:
                return documents
            logger.warning(f"[pdf_loader]{filepath} 增强解析没有拿到内容，回退到纯文本解析")
        except Exception as e:
            logger.error(f"[pdf_loader]{filepath} 增强解析出错，回退到纯文本解析：{e}")

    return PyPDFLoader(filepath, passwd).load()


def txt_loader(filepath: str) -> list[Document]:
    return TextLoader(filepath, encoding="utf-8").load()

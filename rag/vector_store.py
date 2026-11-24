import os

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from model.factory import embed_model
from utils.cofig_handler import chroma_config
from utils.file_handler import (
    get_file_md5_hex,
    listdir_with_allowed_type,
    pdf_loader,
    txt_loader,
)
from utils.logger_handler import logger
from utils.path_tool import get_abs_path


class VectorStoreService():
    def __init__(self):
        self.vector_store = Chroma(
            collection_name=chroma_config["collection_name"],
            embedding_function=embed_model,
            persist_directory=get_abs_path(chroma_config["persist_directory"]),
        )
        self.spliter = RecursiveCharacterTextSplitter(
            chunk_size=chroma_config["chunk_size"],
            chunk_overlap=chroma_config["chunk_overlap"],
            separators=chroma_config["separators"],
            length_function=len,
        )

    def get_retriever(self):
        return self.vector_store.as_retriever(search_kwargs={'k': chroma_config['k']})

    def load_document(self):
        """
        从数据文件内读取文件，转为向量存入向量库
        要计算文件的MD5做去重
        :return: None
        """

        def check_md5_hex(md5_for_check: str):
            md5_file = get_abs_path(chroma_config["md5_hex_store"])
            if not os.path.exists(md5_file):
                open(md5_file, "w", encoding='utf-8').close()
                return False

            with open(md5_file, "r", encoding='utf-8') as f:
                for line in f.readlines():
                    line = line.strip()
                    if line == md5_for_check:
                        return True
                return False

        def save_md5_hex(md5_for_check: str):
            with open(get_abs_path(chroma_config["md5_hex_store"]), "a", encoding='utf-8') as f:
                f.write(md5_for_check + "\n")

        def get_file_documents(read_path: str):    #读文件，拿到所有文件对象
            if read_path.endswith(".txt"):
                return txt_loader(read_path)
            if read_path.endswith(".pdf"):
                return pdf_loader(read_path)
            return []

        allowed_files_path: list[str] = listdir_with_allowed_type(
            get_abs_path(chroma_config["data_path"]),
            tuple(chroma_config["allow_knowledge_file_type"]),
        )
        for path in allowed_files_path:
            md5_hex = get_file_md5_hex(path)
            if check_md5_hex(md5_hex):
                logger.info(f"[加载知识库]{path}已经存在知识库，跳过")
                continue
            try:
                documents: list[Document] = get_file_documents(path)
                if not documents:
                    logger.warning(f"[加载知识库]{path}内没有有效文本，跳过")
                    continue
                spliter_document: list[Document] = self.spliter.split_documents(documents)

                if not spliter_document:
                    logger.warning(f"[加载知识库]{path}内没有有效文本，跳过")
                    continue
                #将内容存入向量库
                self.vector_store.add_documents(spliter_document)
                #记录这个已经处理好的md5，避免重复
                save_md5_hex(md5_hex)
                logger.info(f'[加载知识库]{path} 加载完成，共 {len(spliter_document)} 段')
            except Exception as e:
                #exc_info为True会详细记录报错堆栈，如果为False仅记录错误本身
                logger.error(f"[加载知识库]{path}加载失败：{str(e)}", exc_info=True)
                continue


if __name__ == '__main__':
    vs = VectorStoreService()
    vs.load_document()
    retriever = vs.get_retriever()
    res = retriever.invoke("迷路")
    for f in res:
        print(f.page_content)
        print('-' * 20)

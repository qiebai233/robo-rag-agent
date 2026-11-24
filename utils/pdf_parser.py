# -*- coding: utf-8 -*-
"""
PDF 增强解析器
===============

作用：把一份 PDF 拆成三类 LangChain Document，后续都会被切成 chunk 存进向量库：

  1) 文本  type=text  ：每页正文（会自动剔除表格区域，避免和表格内容重复）
  2) 表格  type=table ：用 PyMuPDF 的 find_tables() 识别，转成 Markdown 表格文本
  3) 图片  type=image ：把内嵌图片提取成 png/jpg，再交给 通义千问VL(qwen3-vl-plus)
                        识别成一段文字描述（文字/表格/示意图都能识别）

为什么要做这件事：
  原来的 PyPDFLoader 只抽“文字层”，PDF 里的图片、截图、表格里的信息全部丢失，
  用户问“我家适合哪种型号”这种需要看参数表/示意图的问题时，检索不到对应内容。
  把图片转成描述文字、把表格转成 Markdown 之后，它们就能像普通文本一样被向量化、被检索到。
"""
import base64
import os

from langchain_core.documents import Document

from utils.cofig_handler import pdf_config
from utils.logger_handler import logger
from utils.path_tool import get_abs_path

try:
    import pymupdf as fitz
except ImportError:  # 没装 pymupdf 时优雅降级，不影响原来的文字模式
    fitz = None


# ---------------------------------------------------------------- 小工具
def _get(key, default):
    """读配置，配置缺失/为 None 时用默认值"""
    value = (pdf_config or {}).get(key, default)
    return default if value is None else value


def _extract_text_from_vl_content(content) -> str:
    """把 DashScope 多模态返回的 content 统一转成字符串"""
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and "text" in item:
                parts.append(str(item["text"]))
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(parts).strip()
    return str(content).strip()


def _table_to_markdown(rows) -> str:
    """把 find_tables().extract() 得到的二维列表转成 Markdown 表格"""
    if not rows:
        return ""
    clean_rows = []
    for row in rows:
        cells = []
        for cell in row:
            text = "" if cell is None else str(cell)
            # 单元格里的换行/竖线会破坏 Markdown 表格，先处理掉
            text = text.replace("\r", " ").replace("\n", " ").replace("|", "\\|").strip()
            cells.append(text)
        clean_rows.append(cells)

    width = max(len(r) for r in clean_rows)
    clean_rows = [r + [""] * (width - len(r)) for r in clean_rows]
    if len(clean_rows) < 2:
        return ""

    mark = "| " + " | ".join(clean_rows[0]) + " |"
    sep = "| " + " | ".join(["---"] * width) + " |"
    body = []
    for row in clean_rows[1:]:
        if all(cell == "" for cell in row):     # 跳过整行空白
            continue
        body.append("| " + " | ".join(row) + " |")
    return "\n".join([mark, sep] + body)


def _page_text_without_tables(page, table_rects) -> str:
    """取本页正文，并剔除表格所在区域（表格会单独转成 Markdown，避免重复）"""
    if not table_rects:
        return (page.get_text("text") or "").strip()

    parts = []
    for block in page.get_text("blocks"):
        # block = (x0, y0, x1, y1, text, block_no, block_type)
        if len(block) < 5:
            continue
        block_rect = fitz.Rect(block[0], block[1], block[2], block[3])
        if any(block_rect.intersects(rect) for rect in table_rects):
            continue
        parts.append(str(block[4]))
    return "\n".join(parts).strip()


def _image_to_data_url(image_path: str) -> str:
    """把本地图片转成 base64 data URL（这样 Windows 路径也不用担心 file:// 兼容性）"""
    ext = os.path.splitext(image_path)[1].lower().lstrip(".") or "png"
    if ext not in ("png", "jpg", "jpeg", "webp", "bmp"):
        ext = "png"
    with open(image_path, "rb") as f:
        raw = f.read()

    # 太大的图先缩小，省 token 也避免超限
    try:
        from PIL import Image
        import io
        if len(raw) > 1_500_000:
            image = Image.open(io.BytesIO(raw))
            image.thumbnail((1600, 1600))
            buffer = io.BytesIO()
            image.convert("RGB").save(buffer, format="JPEG", quality=85)
            raw = buffer.getvalue()
            ext = "jpeg"
    except Exception:
        pass

    return f"data:image/{ext};base64,{base64.b64encode(raw).decode('utf-8')}"


def describe_image(image_path: str) -> str:
    """调 通义千问VL 识别图片，返回文字描述；失败返回空字符串（不影响入库）"""
    api_key = os.getenv("DASHSCOPE_API_KEY") or _get("dashscope_api_key", "")
    if not api_key:
        logger.warning("[pdf解析]没有 DASHSCOPE_API_KEY，跳过图片识别")
        return ""

    model = _get("vlm_model", "qwen3-vl-plus")
    prompt = _get("vlm_prompt", "请完整识别这张图片里的文字和表格，并说明它表达的信息。")
    try:
        import dashscope
        from dashscope import MultiModalConversation
    except Exception as e:
        logger.error(f"[pdf解析]dashscope 不可用：{e}")
        return ""

    dashscope.api_key = api_key
    try:
        response = MultiModalConversation.call(
            model=model,
            messages=[{
                "role": "user",
                "content": [{"image": _image_to_data_url(image_path)}, {"text": prompt}],
            }],
        )
    except Exception as e:
        logger.error(f"[pdf解析]图片识别异常 {os.path.basename(image_path)}：{e}")
        return ""

    if getattr(response, "status_code", None) != 200:
        logger.error(
            f"[pdf解析]图片识别失败 {os.path.basename(image_path)}："
            f"{getattr(response, 'status_code', '')} "
            f"{getattr(response, 'code', '')} {getattr(response, 'message', '')}"
        )
        return ""

    try:
        content = response.output.choices[0].message.content
    except Exception:
        return ""
    result = _extract_text_from_vl_content(content)
    logger.info(f"[pdf解析]图片识别完成：{os.path.basename(image_path)}（{len(result)} 字）")
    return result


def _relative_or_abs(path: str) -> str:
    """图片路径尽量记成相对项目根目录的路径；跨盘符（比如图片目录配到别的盘）时退回绝对路径"""
    try:
        return os.path.relpath(path, get_abs_path("")).replace("\\", "/")
    except ValueError:
        return os.path.abspath(path).replace("\\", "/")


def _save_page_images(doc, page, page_no: int, pdf_stem: str, image_dir: str):
    """把一页里的内嵌图片导出到磁盘，返回 [{path,width,height}]"""
    saved = []
    min_w = int(_get("min_image_width", 80))
    min_h = int(_get("min_image_height", 80))
    max_n = int(_get("max_images_per_page", 5))

    for index, image_info in enumerate(page.get_images(full=True)[:max_n], start=1):
        xref = image_info[0]
        try:
            pix = doc.extract_image(xref)
        except Exception as e:
            logger.warning(f"[pdf解析]第{page_no}页图片导出失败：{e}")
            continue
        if not pix or not pix.get("image"):
            continue
        width, height = pix.get("width", 0), pix.get("height", 0)
        if width < min_w or height < min_h:
            continue
        ext = pix.get("ext", "png")
        if ext not in ("png", "jpg", "jpeg", "webp", "bmp"):
            ext = "png"
        out_path = os.path.join(image_dir, f"{pdf_stem}_p{page_no}_{index}.{ext}")
        with open(out_path, "wb") as f:
            f.write(pix["image"])
        saved.append({"path": out_path, "width": width, "height": height})
    return saved


def _render_page_as_image(page, page_no: int, pdf_stem: str, image_dir: str):
    """扫描件兜底：整页渲染成图片"""
    dpi = int(_get("render_dpi", 150))
    try:
        pix = page.get_pixmap(dpi=dpi)
    except Exception as e:
        logger.warning(f"[pdf解析]第{page_no}页渲染失败：{e}")
        return None
    out_path = os.path.join(image_dir, f"{pdf_stem}_p{page_no}_page.png")
    pix.save(out_path)
    return {"path": out_path, "width": pix.width, "height": pix.height}


# ---------------------------------------------------------------- 主入口
def parse_pdf(filepath: str) -> list[Document]:
    """
    解析一份 PDF，返回 Document 列表（text / table / image 三种 type）。

    注意：返回的 Document 还没有切分，切分交给 VectorStoreService 里的 splitter。
    """
    if fitz is None:
        logger.error("[pdf解析]没有安装 pymupdf，无法使用图片/表格识别（pip install pymupdf）")
        return []
    if not _get("enabled", True):
        return []

    filepath = os.path.abspath(filepath)
    file_name = os.path.basename(filepath)
    pdf_stem = os.path.splitext(file_name)[0]
    image_dir = os.path.join(get_abs_path(_get("image_output_dir", "data/pdf_images")), pdf_stem)
    os.makedirs(image_dir, exist_ok=True)

    docs: list[Document] = []
    try:
        doc = fitz.open(filepath)
    except Exception as e:
        logger.error(f"[pdf解析]打不开 {filepath}：{e}")
        return []

    try:
        for page_no, page in enumerate(doc, start=1):
            base_meta = {"source": filepath, "file_name": file_name, "page": page_no}

            # ---------- 1) 表格 ----------
            table_rects = []
            if _get("extract_tables", True):
                try:
                    found = page.find_tables()
                    tables = getattr(found, "tables", []) or []
                except Exception as e:
                    logger.warning(f"[pdf解析]第{page_no}页表格识别失败：{e}")
                    tables = []
                for table_no, table in enumerate(tables, start=1):
                    try:
                        rows = table.extract()
                        table_bbox = table.bbox
                    except Exception as e:
                        logger.warning(f"[pdf解析]第{page_no}页表格{table_no}解析失败：{e}")
                        continue
                    markdown = _table_to_markdown(rows)
                    if not markdown:
                        continue
                    table_rects.append(fitz.Rect(table_bbox))
                    docs.append(Document(
                        page_content=f"【表格】{file_name} 第{page_no}页 表格{table_no}\n{markdown}",
                        metadata={**base_meta, "type": "table", "table_no": table_no},
                    ))
                    logger.info(f"[pdf解析]{file_name} 第{page_no}页 识别到表格{table_no}（{len(rows)} 行）")

            # ---------- 2) 正文 ----------
            text = _page_text_without_tables(page, table_rects)
            if text and _get("extract_text", True):
                docs.append(Document(
                    page_content=f"【正文】{file_name} 第{page_no}页\n{text}",
                    metadata={**base_meta, "type": "text"},
                ))

            # ---------- 3) 图片 ----------
            if not _get("extract_images", True):
                continue
            images = _save_page_images(doc, page, page_no, pdf_stem, image_dir)

            # 扫描件兜底：这页几乎没有文字、也没有内嵌图片 → 整页渲染（并自动排除已经处理过表格的页）
            if (not images and not text and not table_rects
                    and _get("render_scanned_page", True)):
                rendered = _render_page_as_image(page, page_no, pdf_stem, image_dir)
                if rendered:
                    images = [rendered]
                    logger.info(f"[pdf解析]{file_name} 第{page_no}页疑似扫描件，已整页渲染识别")

            for image_no, image in enumerate(images, start=1):
                rel_path = _relative_or_abs(image["path"])
                content = f"【图片】{file_name} 第{page_no}页 第{image_no}张图片"
                if _get("describe_images", True):
                    description = describe_image(image["path"])
                    if description:
                        content += f"\n图片内容识别：{description}"
                docs.append(Document(
                    page_content=content,
                    metadata={
                        **base_meta,
                        "type": "image",
                        "image_no": image_no,
                        "image_path": rel_path,
                        "image_width": image["width"],
                        "image_height": image["height"],
                    },
                ))
    finally:
        doc.close()

    return docs


if __name__ == '__main__':
    import sys
    from collections import Counter

    target = sys.argv[1] if len(sys.argv) > 1 else get_abs_path("data/扫地机器人100问.pdf")
    result = parse_pdf(target)
    print(f"\n文件：{target}")
    print(f"共解析出 {len(result)} 个 Document，类型统计：{dict(Counter(d.metadata.get('type') for d in result))}")
    for i, d in enumerate(result, start=1):
        preview = d.page_content.replace("\n", " ")[:100]
        print(f"{i:>2}. [{d.metadata.get('type')}] 第{d.metadata.get('page')}页 | {preview} ...")


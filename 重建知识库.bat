@echo off
chcp 65001 >nul
cd /d "%~dp0"

echo ============================================
echo  重建知识库
echo  - 会删除旧的 chroma_db / md5.txt / data\pdf_images
echo  - 重新解析 data 目录（PDF 里的表格和图片会被识别）
echo ============================================
echo.

if exist chroma_db rmdir /s /q chroma_db
if exist md5.txt del md5.txt
if exist data\pdf_images rmdir /s /q data\pdf_images

echo 旧的向量库已清理，开始解析并入库...
echo.
".venv\Scripts\python.exe" -m rag.vector_store

echo.
echo 完成。按任意键退出。
pause >nul

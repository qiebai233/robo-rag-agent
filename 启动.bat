@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo 使用项目虚拟环境启动: %CD%\.venv
".venv\Scripts\python.exe" -m streamlit run app.py
pause

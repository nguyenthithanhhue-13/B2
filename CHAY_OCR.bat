@echo off
chcp 65001 >nul
cd /d "%~dp0"
set "PY=C:\Users\Thanh Hue\.venv\Scripts\python.exe"
if not exist "%PY%" set "PY=python"
rem Khong cho thu vien nao tu len mang (moi mo hinh da nam san trong may)
set HF_HUB_OFFLINE=1
set TRANSFORMERS_OFFLINE=1
set YOLO_OFFLINE=1
echo ============================================================
echo  BAI 2 - DOC CHU TRONG ANH (OCR). Tra loi tung cau, keo tha thu muc roi Enter.
echo ============================================================
"%PY%" ocr_mau.py
echo.
pause

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
echo  BAI 2 - KHOANH VAT THE / TO VUNG (YOLO). Tra loi tung cau, keo tha file roi Enter.
echo ============================================================
"%PY%" yolo_mau.py
echo.
pause

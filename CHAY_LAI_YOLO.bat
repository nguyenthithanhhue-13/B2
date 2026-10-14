@echo off
chcp 65001 >nul
cd /d "%~dp0"
set "PY=C:\Users\Thanh Hue\.venv\Scripts\python.exe"
if not exist "%PY%" set "PY=python"
rem Khong cho thu vien nao tu len mang (moi mo hinh da nam san trong may)
set HF_HUB_OFFLINE=1
set TRANSFORMERS_OFFLINE=1
set YOLO_OFFLINE=1
echo  YOLO - chay lai dung cau hinh cu.
if not "%~1"=="" (set "CH=%~1") else (set /p "CH=Keo tha file cau_hinh.json vao day roi bam Enter: ")
set "CH=%CH:"=%"
"%PY%" yolo_mau.py --lai "%CH%"
echo.
pause

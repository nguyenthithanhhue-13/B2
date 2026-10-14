@echo off
chcp 65001 >nul
cd /d "%~dp0"
set "PY=C:\Users\Thanh Hue\.venv\Scripts\python.exe"
if not exist "%PY%" set "PY=python"
rem Khong cho thu vien nao tu len mang (moi mo hinh da nam san trong may)
set HF_HUB_OFFLINE=1
set TRANSFORMERS_OFFLINE=1
echo ============================================================
echo  BAI 2 - tra loi vai cau hoi, may tu hoc va ghi file nop.
echo  Moi cau hoi: keo tha file vao cua so nay, hoac go cau tra loi, roi bam Enter.
echo ============================================================
"%PY%" bai2.py
echo.
pause

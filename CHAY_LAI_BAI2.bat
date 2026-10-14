@echo off
chcp 65001 >nul
cd /d "%~dp0"
set "PY=C:\Users\Thanh Hue\.venv\Scripts\python.exe"
if not exist "%PY%" set "PY=python"
rem Khong cho thu vien nao tu len mang (moi mo hinh da nam san trong may)
set HF_HUB_OFFLINE=1
set TRANSFORMERS_OFFLINE=1
echo ============================================================
echo  BAI 2 - chay lai dung cau hinh cu (sau khi sua code), khong phai tra loi lai.
echo ============================================================
if not "%~1"=="" (set "CH=%~1") else (set /p "CH=Keo tha file cau_hinh.json (trong ket_qua\ten_lan_chay) vao day roi bam Enter: ")
set "CH=%CH:"=%"
"%PY%" bai2.py --lai "%CH%"
echo.
pause

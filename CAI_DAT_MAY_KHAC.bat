@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo ============================================================
echo  CAI DAT tren may KHAC (khong phai may Hue). Can MANG, mat 20-40 phut.
echo  Can co Python 3.13 (python.org) - khi cai nho tick 'Add python.exe to PATH'.
echo ============================================================
python --version || (echo Chua co Python. Cai tu python.org roi chay lai file nay. & pause & exit /b)
python -m pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
python -m pip install --no-deps easyocr==1.7.2
echo.
echo Tai cac mo hinh (doc anh, YOLO, OCR, doc cau, dich)... khoang 6 GB
python tai_mo_hinh_cv.py
echo.
echo XONG. Neu dong nao ghi [LOI] thi chay lai file nay.
pause

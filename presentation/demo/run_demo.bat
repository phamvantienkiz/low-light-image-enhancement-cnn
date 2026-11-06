@echo off
chcp 65001 >nul
echo ====================================================================
echo  🚀 KHỞI ĐỘNG ỨNG DỤNG DEMO CỤC BỘ LLIE DARKIR-EDGE (CS2203 - UIT)
echo ====================================================================

cd /d "%~dp0"

:: Kiểm tra môi trường ảo cục bộ .venv ở thư mục gốc dự án
set VENV_PYTHON=..\..\.venv\Scripts\python.exe
set LOCAL_VENV_PYTHON=.venv\Scripts\python.exe

if exist "%LOCAL_VENV_PYTHON%" (
    echo [OK] Tìm thấy môi trường ảo tại presentation\demo\.venv
    set PYTHON_EXEC=%LOCAL_VENV_PYTHON%
) else if exist "%VENV_PYTHON%" (
    echo [OK] Tìm thấy môi trường ảo tại gốc dự án (.venv)
    set PYTHON_EXEC=%VENV_PYTHON%
) else (
    echo [CẢNH BÁO] Không tìm thấy môi trường ảo .venv cục bộ.
    echo Đang sử dụng python trong hệ thống hiện tại...
    set PYTHON_EXEC=python
)

echo.
echo [*] Đang kiểm tra thư viện và khởi động máy chủ Demo...
echo [*] Giao diện web sẽ tự động mở tại: http://127.0.0.1:7860
echo.

"%PYTHON_EXEC%" app.py

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [LỖI] Ứng dụng dừng với mã lỗi %ERRORLEVEL%.
    echo Nếu thiếu thư viện, hãy cài đặt bằng: pip install -r requirements.txt
    pause
)

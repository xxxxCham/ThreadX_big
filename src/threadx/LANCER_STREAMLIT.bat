@echo off
REM Tuer les instances Streamlit existantes
taskkill /F /IM streamlit.exe 2>nul

cd /d "D:\ThreadX_big"
call .venv\Scripts\activate.bat
set PYTHONPATH=D:\ThreadX_big\src
set THREADX_DEBUG=0
rem Silence all logs for performance testing (set to 0 to re-enable)
set THREADX_SILENCE_LOGS=1
streamlit run src\threadx\streamlit_app.py --server.port=8504

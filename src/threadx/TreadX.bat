@echo off
REM Tuer les instances Streamlit existantes
taskkill /F /IM streamlit.exe 2>nul

cd "D:\ThreadX_big"
call .venv\Scripts\activate.bat
set ollama serve
set PYTHONPATH=D:\ThreadX_big\src
set THREADX_DEBUG=1
rem Silence all logs for performance testing (set to 0 to re-enable)
set THREADX_SILENCE_LOGS=0
streamlit run src\threadx\streamlit_app.py --server.port=8501


@echo off
REM ===============================================
REM ThreadX - Lanceur RAPIDE (sans vérifications)
REM ===============================================
REM Utiliser UNIQUEMENT si tout est déjà configuré
REM ===============================================

title ThreadX - Lancement Rapide

echo.
echo Lancement ThreadX...
echo.

REM Lancement direct
cd /d "%~dp0"
start /min "" "C:\Program Files\BraveSoftware\Brave-Browser\Application\brave.exe" "http://localhost:8501"
".venv\Scripts\python.exe" -m streamlit run "src\threadx\ui\page_backtest_optimization.py" --server.port=8501 --server.headless=true --browser.gatherUsageStats=false --theme.base=dark

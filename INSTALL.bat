@echo off
REM ===============================================
REM ThreadX - Installation Première Utilisation
REM ===============================================
REM À lancer UNE SEULE FOIS avant la première utilisation
REM ===============================================

title ThreadX - Installation

setlocal EnableDelayedExpansion

echo.
echo ================================================
echo    ThreadX - Installation Premiere Utilisation
echo ================================================
echo.
echo Ce script va:
echo  1. Verifier Python
echo  2. Creer l'environnement virtuel
echo  3. Installer toutes les dependances
echo.
echo ================================================
echo.
pause

REM ==========================
REM 1. Vérification Python
REM ==========================
echo.
echo [1/3] Verification Python...

python --version >nul 2>&1
if errorlevel 1 (
    echo [ERREUR] Python n'est pas installe ou pas dans le PATH
    echo.
    echo Telechargez Python 3.12+ sur: https://www.python.org/downloads/
    echo Assurez-vous de cocher "Add Python to PATH" durant l'installation
    pause
    exit /b 1
)

for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo [OK] Python %PYTHON_VERSION% trouve
echo.

REM ==========================
REM 2. Création environnement
REM ==========================
echo [2/3] Creation environnement virtuel...

if exist ".venv\" (
    echo [INFO] Environnement .venv existe deja
    choice /C ON /M "Voulez-vous le recreer (O=Oui, N=Non)"
    if errorlevel 2 (
        echo [INFO] Conservation de l'environnement existant
        goto INSTALL_DEPS
    )
    echo [INFO] Suppression ancien environnement...
    rmdir /s /q ".venv"
)

echo [INFO] Creation de .venv...
python -m venv .venv

if errorlevel 1 (
    echo [ERREUR] Echec creation environnement virtuel
    pause
    exit /b 1
)

echo [OK] Environnement virtuel cree
echo.

REM ==========================
REM 3. Installation dépendances
REM ==========================
:INSTALL_DEPS
echo [3/3] Installation des dependances...
echo.
echo [INFO] Cela peut prendre 5-10 minutes selon votre connexion...
echo.

if not exist "requirements.txt" (
    echo [ERREUR] Fichier requirements.txt introuvable
    pause
    exit /b 1
)

REM Mise à jour pip
echo [INFO] Mise a jour de pip...
".venv\Scripts\python.exe" -m pip install --upgrade pip --quiet

if errorlevel 1 (
    echo [ATTENTION] Echec mise a jour pip (non critique)
) else (
    echo [OK] pip mis a jour
)

REM Installation requirements
echo.
echo [INFO] Installation des packages...
echo [INFO] Patience, cela peut prendre plusieurs minutes...
echo.

".venv\Scripts\python.exe" -m pip install -r requirements.txt

if errorlevel 1 (
    echo.
    echo [ERREUR] Echec installation des requirements
    echo.
    echo Essayez manuellement:
    echo   .\.venv\Scripts\activate
    echo   pip install -r requirements.txt
    pause
    exit /b 1
)

echo.
echo [OK] Toutes les dependances sont installees
echo.

REM ==========================
REM Vérification finale
REM ==========================
echo ================================================
echo    Verification finale...
echo ================================================
echo.

".venv\Scripts\python.exe" -c "import streamlit, pandas, numpy, plotly, numba; print('[OK] Tous les packages critiques sont installes')"

if errorlevel 1 (
    echo [ATTENTION] Certains packages manquent peut-etre
    echo Mais l'installation de base est terminee
)

echo.
echo ================================================
echo    INSTALLATION TERMINEE !
echo ================================================
echo.
echo Vous pouvez maintenant lancer l'application avec:
echo   - Launch_ThreadX_App.bat (recommande)
echo   - ThreadX_Menu.bat (menu interactif)
echo   - Launch_FAST.bat (lancement rapide)
echo.
echo ================================================
echo.
pause

REM Proposer de lancer l'application
echo.
choice /C ON /M "Voulez-vous lancer l'application maintenant"
if errorlevel 2 goto END

echo.
echo Lancement de ThreadX...
timeout /t 2 >nul
call "%~dp0Launch_ThreadX_App.bat"

:END
exit /b 0

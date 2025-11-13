@echo off
REM Open_VSCode_Terminal.bat
REM Ouvre VS Code sur le workspace ThreadX et tente d'ouvrir le terminal intégré (qui exécutera l'activation automatique)

nREM 1) Ouvrir VS Code sur le dossier
"%ProgramFiles%\Microsoft VS Code\Code.exe" -r "D:\ThreadX_big" 2>nul || code -r "D:\ThreadX_big"

REM 2) Tenter d'ouvrir le terminal intégré via l'option --command (si supportée par votre version de code)
"%ProgramFiles%\Microsoft VS Code\Code.exe" --command workbench.action.terminal.toggleTerminal 2>nul || code --command workbench.action.terminal.toggleTerminal 2>nul

REM Note: si la commande --command n'est pas supportée, appuyez sur Ctrl+` dans VS Code pour ouvrir le terminal intégré;
REM le terminal intégré utilisera le profil PowerShell configuré dans .vscode/settings.json et exécutera automatiquement activate_threadx.ps1.

exit /b 0

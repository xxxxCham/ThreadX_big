# Script PowerShell pour redémarrer Streamlit proprement
# Nettoie les caches Python et force le rechargement des modules

Write-Host "🧹 Nettoyage des caches Python..." -ForegroundColor Yellow

# Supprimer __pycache__ dans le projet
Get-ChildItem -Path "D:\ThreadX_big" -Include "__pycache__" -Recurse -Directory | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
Write-Host "   ✓ Cache Python nettoyé" -ForegroundColor Green

# Supprimer cache Streamlit
$streamlitCache = "$env:USERPROFILE\.streamlit\cache"
if (Test-Path $streamlitCache) {
    Remove-Item -Path $streamlitCache -Recurse -Force -ErrorAction SilentlyContinue
    Write-Host "   ✓ Cache Streamlit nettoyé" -ForegroundColor Green
}

# Tuer processus Streamlit existants
Get-Process -Name streamlit -ErrorAction SilentlyContinue | Stop-Process -Force
Write-Host "   ✓ Processus Streamlit arrêtés" -ForegroundColor Green

Write-Host ""
Write-Host "🚀 Lancement de Streamlit..." -ForegroundColor Cyan
Write-Host "   URL: http://localhost:8501" -ForegroundColor Gray
Write-Host ""

# Lancer Streamlit
Set-Location "D:\ThreadX_big"
streamlit run src\threadx\ui\app.py

@echo off
REM Script para ejecutar Dashboard de Métricas en Windows
REM Endurance Lab AI

echo 🚀 Iniciando Dashboard de Métricas - Endurance Lab AI
echo ============================================================

REM Verificar Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Error: Python no está instalado o no está en PATH
    pause
    exit /b 1
)

REM Verificar archivo
if not exist "app\metrics_dashboard.py" (
    echo ❌ Error: No se encuentra app\metrics_dashboard.py
    echo    Asegúrate de ejecutar este script desde la raíz del proyecto
    pause
    exit /b 1
)

echo ✅ Verificaciones completadas
echo.
echo 📊 Configuración del Dashboard:
echo    - Puerto: 8502
echo    - URL: http://localhost:8502
echo    - Archivo: app\metrics_dashboard.py
echo.
echo 🔄 Iniciando servidor...
echo ============================================================
echo 🌐 Dashboard iniciado exitosamente!
echo 📱 Abre tu navegador en: http://localhost:8502
echo ⏹️  Presiona Ctrl+C para detener el servidor
echo ============================================================
echo.

REM Ejecutar Streamlit
python -m streamlit run app\metrics_dashboard.py --server.port=8502 --server.address=localhost

echo.
echo ⏹️  Dashboard detenido
pause
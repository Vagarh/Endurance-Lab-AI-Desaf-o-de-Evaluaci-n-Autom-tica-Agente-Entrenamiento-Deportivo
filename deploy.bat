@echo off
REM Endurance Lab AI - Production Deployment Script for Windows
REM Author: Juan Felipe Cardona Arango

echo 🚀 Iniciando despliegue de Endurance Lab AI...

REM Check if Docker is installed
docker --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Docker no está instalado. Por favor instala Docker primero.
    pause
    exit /b 1
)

REM Check if Docker Compose is installed
docker-compose --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Docker Compose no está instalado. Por favor instala Docker Compose primero.
    pause
    exit /b 1
)

REM Check if .env file exists
if not exist .env (
    echo [WARNING] Archivo .env no encontrado. Copiando desde production.env...
    copy production.env .env
    echo [WARNING] Por favor edita el archivo .env con tus configuraciones antes de continuar.
    set /p answer="¿Has configurado el archivo .env? (y/N): "
    if /i not "%answer%"=="y" (
        echo [ERROR] Configura el archivo .env y ejecuta el script nuevamente.
        pause
        exit /b 1
    )
)

echo [INFO] Verificando configuración...

REM Create necessary directories
if not exist vectorstore mkdir vectorstore
if not exist mlruns mkdir mlruns
if not exist ssl mkdir ssl
if not exist logs mkdir logs

echo [INFO] Construyendo imagen Docker...
docker-compose build

echo [INFO] Verificando que el vectorstore existe...
if not exist vectorstore\index.faiss (
    echo [WARNING] Vectorstore no encontrado. Creando...
    docker-compose run --rm endurance-lab-ai python create_vectorstore.py
)

echo [INFO] Iniciando servicios...
docker-compose up -d

echo [INFO] Esperando que los servicios estén listos...
timeout /t 10 /nobreak >nul

echo [INFO] Verificando estado de los servicios...
python health_check.py

echo.
echo 🎉 Despliegue completado!
echo.
echo 📱 Accede a la aplicación en:
echo    - Directo: http://localhost:8501
echo    - Via Nginx: http://localhost
echo.
echo 🔧 Comandos útiles:
echo    - Ver logs: docker-compose logs -f
echo    - Parar servicios: docker-compose down
echo    - Reiniciar: docker-compose restart
echo.
echo ¡Endurance Lab AI está listo para usar! 🏆
pause
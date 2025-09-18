#!/bin/bash

# Endurance Lab AI - Production Deployment Script
# Author: Juan Felipe Cardona Arango

set -e  # Exit on any error

echo "🚀 Iniciando despliegue de Endurance Lab AI..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    print_error "Docker no está instalado. Por favor instala Docker primero."
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    print_error "Docker Compose no está instalado. Por favor instala Docker Compose primero."
    exit 1
fi

# Check if .env file exists
if [ ! -f .env ]; then
    print_warning "Archivo .env no encontrado. Copiando desde production.env..."
    cp production.env .env
    print_warning "Por favor edita el archivo .env con tus configuraciones antes de continuar."
    read -p "¿Has configurado el archivo .env? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_error "Configura el archivo .env y ejecuta el script nuevamente."
        exit 1
    fi
fi

# Validate OPENAI_API_KEY
if ! grep -q "OPENAI_API_KEY=sk-" .env; then
    print_error "OPENAI_API_KEY no está configurado correctamente en .env"
    exit 1
fi

print_status "Verificando configuración..."

# Create necessary directories
mkdir -p vectorstore mlruns ssl logs

print_status "Construyendo imagen Docker..."
docker-compose build

print_status "Verificando que el vectorstore existe..."
if [ ! -d "vectorstore/index.faiss" ] && [ ! -f "vectorstore/index.faiss" ]; then
    print_warning "Vectorstore no encontrado. Creando..."
    docker-compose run --rm endurance-lab-ai python create_vectorstore.py
fi

print_status "Iniciando servicios..."
docker-compose up -d

# Wait for services to be ready
print_status "Esperando que los servicios estén listos..."
sleep 10

# Health check
print_status "Verificando estado de los servicios..."
if curl -f http://localhost:8501/_stcore/health > /dev/null 2>&1; then
    print_success "✅ Aplicación está funcionando correctamente"
else
    print_error "❌ La aplicación no responde. Verificando logs..."
    docker-compose logs endurance-lab-ai
    exit 1
fi

if curl -f http://localhost/health > /dev/null 2>&1; then
    print_success "✅ Nginx está funcionando correctamente"
else
    print_warning "⚠️ Nginx puede tener problemas. Verificando logs..."
    docker-compose logs nginx
fi

print_success "🎉 Despliegue completado exitosamente!"
echo
echo "📱 Accede a la aplicación en:"
echo "   - Directo: http://localhost:8501"
echo "   - Via Nginx: http://localhost"
echo
echo "🔧 Comandos útiles:"
echo "   - Ver logs: docker-compose logs -f"
echo "   - Parar servicios: docker-compose down"
echo "   - Reiniciar: docker-compose restart"
echo "   - Actualizar: git pull && docker-compose build && docker-compose up -d"
echo
echo "📊 Monitoreo:"
echo "   - Estado de contenedores: docker-compose ps"
echo "   - Uso de recursos: docker stats"
echo
print_success "¡Endurance Lab AI está listo para usar! 🏆"
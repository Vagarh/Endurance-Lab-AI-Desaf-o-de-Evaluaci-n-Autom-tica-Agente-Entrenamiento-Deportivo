# 🚀 Guía de Despliegue para Producción

## 📋 Preparación Completada

Tu proyecto **Endurance Lab AI** está ahora completamente preparado para producción con:

### ✅ Archivos de Producción Creados

1. **`docker-compose.yml`** - Orquestación de contenedores
2. **`Dockerfile`** (mejorado) - Imagen optimizada para producción
3. **`nginx.conf`** - Proxy reverso con seguridad y rate limiting
4. **`deploy.sh`** - Script automatizado de despliegue
5. **`health_check.py`** - Verificación de estado del sistema
6. **`requirements-prod.txt`** - Dependencias optimizadas para producción
7. **`production.env`** - Template de variables de entorno
8. **`.dockerignore`** - Optimización de imagen Docker

---

## 🚀 Despliegue Rápido

### Opción 1: Despliegue Automatizado (Recomendado)

```bash
# 1. Hacer el script ejecutable
chmod +x deploy.sh

# 2. Ejecutar despliegue
./deploy.sh
```

### Opción 2: Despliegue Manual

```bash
# 1. Configurar variables de entorno
cp production.env .env
# Editar .env con tu OPENAI_API_KEY

# 2. Construir y ejecutar
docker-compose build
docker-compose up -d

# 3. Verificar estado
python health_check.py
```

---

## 🌐 Acceso a la Aplicación

Una vez desplegado, accede a:

- **Aplicación principal**: http://localhost:8501
- **Via Nginx**: http://localhost (con rate limiting y seguridad)
- **Health check**: http://localhost/health

---

## 🔧 Comandos de Gestión

### Monitoreo
```bash
# Ver estado de contenedores
docker-compose ps

# Ver logs en tiempo real
docker-compose logs -f

# Ver uso de recursos
docker stats

# Verificar salud del sistema
python health_check.py
```

### Mantenimiento
```bash
# Reiniciar servicios
docker-compose restart

# Parar servicios
docker-compose down

# Actualizar aplicación
git pull
docker-compose build
docker-compose up -d
```

### Backup
```bash
# Backup del vectorstore
tar -czf vectorstore_backup.tar.gz vectorstore/

# Backup de métricas MLflow
tar -czf mlruns_backup.tar.gz mlruns/
```

---

## 🔒 Configuración de Seguridad

### Variables de Entorno Críticas
```bash
# Requeridas
OPENAI_API_KEY=tu_clave_openai

# Recomendadas para producción
SECRET_KEY=clave_secreta_aleatoria
ALLOWED_HOSTS=tu-dominio.com,localhost
ENVIRONMENT=production
```

### HTTPS (Producción)
Para habilitar HTTPS:

1. Obtén certificados SSL (Let's Encrypt recomendado)
2. Coloca los certificados en `./ssl/`
3. Descomenta la sección HTTPS en `nginx.conf`
4. Actualiza `ALLOWED_HOSTS` con tu dominio

---

## 📊 Monitoreo y Métricas

### Health Checks Automáticos
- **Docker**: Health checks integrados cada 30s
- **Nginx**: Rate limiting (10 req/s por IP)
- **Aplicación**: Endpoint `/_stcore/health`

### Logs
```bash
# Logs de aplicación
docker-compose logs endurance-lab-ai

# Logs de Nginx
docker-compose logs nginx

# Logs del sistema
docker-compose logs
```

---

## 🚀 Escalabilidad

### Recursos Recomendados

#### Desarrollo/Testing
- **CPU**: 2 cores
- **RAM**: 4GB
- **Disco**: 10GB

#### Producción Pequeña (1-100 usuarios)
- **CPU**: 4 cores
- **RAM**: 8GB
- **Disco**: 50GB

#### Producción Media (100-1000 usuarios)
- **CPU**: 8 cores
- **RAM**: 16GB
- **Disco**: 100GB
- **Load Balancer**: Recomendado

### Optimizaciones
```bash
# Aumentar workers de Streamlit
docker-compose up --scale endurance-lab-ai=3

# Configurar load balancer
# (Actualizar nginx.conf con múltiples upstreams)
```

---

## 🔧 Troubleshooting

### Problemas Comunes

#### 1. "OPENAI_API_KEY no configurado"
```bash
# Verificar .env
cat .env | grep OPENAI_API_KEY

# Configurar si falta
echo "OPENAI_API_KEY=tu_clave" >> .env
```

#### 2. "Vectorstore no encontrado"
```bash
# Crear vectorstore
docker-compose run --rm endurance-lab-ai python create_vectorstore.py
```

#### 3. "Puerto 8501 en uso"
```bash
# Verificar procesos
lsof -i :8501

# Cambiar puerto en docker-compose.yml
ports:
  - "8502:8501"  # Usar puerto 8502
```

#### 4. "Nginx no inicia"
```bash
# Verificar configuración
nginx -t -c nginx.conf

# Ver logs específicos
docker-compose logs nginx
```

---

## 📈 Próximos Pasos

### Mejoras Inmediatas
1. **SSL/HTTPS** para producción real
2. **Dominio personalizado** 
3. **Backup automatizado**
4. **Monitoreo avanzado** (Prometheus/Grafana)

### Funcionalidades Futuras
1. **Autenticación de usuarios**
2. **API REST** para integraciones
3. **App móvil**
4. **Integración con Strava/Garmin**

---

## 🎯 Estado Final

✅ **Proyecto completamente preparado para producción**

- Arquitectura escalable con Docker
- Seguridad implementada (Nginx + rate limiting)
- Monitoreo y health checks
- Scripts de despliegue automatizado
- Documentación completa

**¡Tu Endurance Lab AI está listo para conquistar el mundo del deporte! 🏆**

---

*Preparado por: Juan Felipe Cardona Arango*  
*Fecha: Septiembre 2025*  
*Versión: 1.0 Production Ready*
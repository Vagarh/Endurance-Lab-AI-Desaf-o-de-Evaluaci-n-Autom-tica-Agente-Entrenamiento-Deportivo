# 🎨 Assets - Endurance Lab AI

Esta carpeta contiene todos los recursos estáticos del proyecto.

## 📋 Estructura

### 🖼️ Imágenes
- `images/` - Imágenes del proyecto
  - `logos/` - Logotipos y marcas
  - `screenshots/` - Capturas de pantalla
  - `icons/` - Iconos y elementos gráficos

### 🎨 Estilos
- `styles/` - Archivos CSS y estilos
- `themes/` - Temas personalizados

### 📄 Documentos
- `docs/` - PDFs y documentos estáticos
- `templates/` - Plantillas de documentos

## 🔗 Uso

Los assets se referencian desde la aplicación usando rutas relativas:

```python
from pathlib import Path
ASSETS_DIR = Path(__file__).parent / "assets"
```
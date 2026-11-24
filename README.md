# Análisis Comparativo DMO vs. Imagen Percibida (XidooTopic)

Sistema de análisis de lenguaje natural y visualización interactiva (2 pantallas × 4 cuadrantes) para contrastar la imagen proyectada por una DMO frente a la imagen percibida en fuentes UGC.

## Estructura del Proyecto

- `main.py`: Pipeline completo (limpieza, lematización, *embeddings* con `xidootopic`, detección de *outliers*, sentimiento, modelado de tópicos, análisis comparativo y de precio).
- `dashboard_html.py`: Librería dedicada exclusivamente a la generación del tablero interactivo HTML.
- `data/`: Incluye **3 datasets sintéticos cortos de ejemplo** listos para probar el sistema sin datos de terceros:
  - `data/dmo_ejemplo.csv` (columna `comentario`, 18 registros tipo guía oficial DMO)
  - `data/ugc_resenas_ejemplo.csv` (columna `texto`, 32 registros tipo reseñas UGC)
  - `data/ugc_redes_ejemplo.csv` (columna `texto`, 22 registros tipo redes sociales ES/EN)
- `requirements.txt`: Dependencias de Python.

Al ejecutarse, `main.py` crea automáticamente el directorio de salida (`resultados/` junto al script, o el que indiques con `-o`) con:
- `resultados/csv/`: matrices procesadas, *outliers*, n-gramas, brechas y resumen JSON.
- `resultados/reportes/`: reporte Markdown con métricas clave.
- `resultados/tablero_2pantallas_8puntos.html`: tablero interactivo final.

## Instalación

```bash
pip install -r requirements.txt
python -m spacy download es_core_news_sm
python -m spacy download en_core_web_sm
```

## Uso

### 1. Prueba rápida con los 3 datasets de ejemplo incluidos
```bash
python main.py
```

### 2. Con paleta alternativa (`okabe-ito` o `viridis`) y directorio de salida personalizado
```bash
python main.py -p viridis -o mi_corrida
```

### 3. Múltiples CSVs propios (UGC y DMO) con columnas independientes
Todas las fuentes `--dmo_dataset` se fusionan en una sola fuente DMO (`--dmo_label`, por defecto `sectur`), mientras que cada CSV en `--ugc_dataset` conserva su fuente para análisis individual:

```bash
python main.py \
  --dmo_dataset data/dmo_ejemplo.csv \
  --dmo_text_col comentario \
  --ugc_dataset data/ugc_resenas_ejemplo.csv data/ugc_redes_ejemplo.csv \
  --ugc_text_col texto texto \
  -l es -p okabe-ito
```

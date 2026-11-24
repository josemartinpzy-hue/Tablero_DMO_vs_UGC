# Tablero Decisional DMO vs. Imagen Percibida (UGC)

**Autor:** José Martín Páramo Zárate  
**Programa:** Posgrado LIDIA 2026 — Visualización de Información

Sistema de análisis de lenguaje natural y visualización interactiva (**2 pantallas × 4 cuadrantes**) para contrastar la imagen proyectada por una DMO (*Destination Management Organization*, ej. SECTUR) frente a la imagen percibida en fuentes UGC (*User-Generated Content* como reseñas o redes sociales).

---

## Estructura del Proyecto

- `main.py`: Pipeline completo de análisis (limpieza, lematización ES/EN, *embeddings* densos con `xidootopic`, detección de *outliers* y n-gramas, análisis de sentimiento, modelado de tópicos con respaldo por frecuencia para particiones pequeñas, comparativo DMO vs. UGC y análisis del eje precio/valor).
- `dashboard_html.py`: Librería dedicada exclusivamente a la generación del tablero interactivo HTML (`PALETTES`, nubes SVG y gráficas Plotly).
- `data/`: Incluye **3 datasets sintéticos cortos de ejemplo** listos para probar el sistema:
  - `data/dmo_ejemplo.csv` (columna `comentario`, 18 registros tipo guía oficial DMO)
  - `data/ugc_resenas_ejemplo.csv` (columna `texto`, 32 registros tipo reseñas UGC)
  - `data/ugc_redes_ejemplo.csv` (columna `texto`, 22 registros tipo redes sociales ES/EN)
- `requirements.txt`: Dependencias de Python.

Al ejecutarse, `main.py` crea automáticamente el directorio de salida (`resultados/` junto al script, o el que indiques con `-o`) con:
- `resultados/csv/`: matrices procesadas, *outliers*, n-gramas, brechas temáticas y resumen JSON.
- `resultados/reportes/`: reporte Markdown con métricas clave.
- `resultados/tablero_2pantallas_8puntos.html`: tablero interactivo final.

---

## Instalación

```bash
pip install -r requirements.txt
python -m spacy download es_core_news_sm
python -m spacy download en_core_web_sm
```

---

## Ejemplos de Ejecución

### 1. Prueba rápida con los 3 datasets sintéticos de ejemplo
```bash
python main.py
```

### 2. Cambiar paleta de colores (`okabe-ito` o `viridis`) y carpeta de salida
```bash
python main.py -p viridis -o mi_corrida
```

### 3. Todos los datasets con la misma columna de texto (ej. `texto`)
Cuando todos los CSV comparten el mismo nombre de columna, basta pasar un solo nombre en `--ugc_text_col` y `--dmo_text_col`:
```bash
python main.py --dmo_dataset data/dmo_oficial.csv --dmo_text_col texto --ugc_dataset data/ugc_resenas_ejemplo.csv data/ugc_redes_ejemplo.csv --ugc_text_col texto
```

### 4. Dos datasets con la misma columna y uno diferente
- **2 UGC con columna `texto` y 1 DMO con columna `comentario`:**
```bash
python main.py --dmo_dataset data/dmo_ejemplo.csv --dmo_text_col comentario --ugc_dataset data/ugc_resenas_ejemplo.csv data/ugc_redes_ejemplo.csv --ugc_text_col texto -l es -p okabe-ito
```
- **2 UGC con columnas distintas (`pregunta` y `texto`) y 1 DMO con `comentario`** *(se indican en el mismo orden de los archivos)*:
```bash
python main.py --dmo_dataset data/dmo_ejemplo.csv --dmo_text_col comentario --ugc_dataset data/encuesta.csv data/ugc_redes_ejemplo.csv --ugc_text_col pregunta texto -l es -p okabe-ito
```

### 5. Múltiples archivos DMO y múltiples archivos UGC con etiquetas personalizadas
Todas las fuentes en `--dmo_dataset` se fusionan en una sola fuente DMO (`--dmo_label`, por defecto `sectur`), mientras que cada CSV en `--ugc_dataset` conserva su propia etiqueta para análisis individual:
```bash
python main.py --dmo_dataset data/dmo_1.csv data/dmo_2.csv --dmo_text_col comentario texto -m sectur --ugc_dataset data/ugc_resenas_ejemplo.csv data/ugc_redes_ejemplo.csv --ugc_text_col texto --ugc_label resenas redes -p viridis
```

### 6. Un solo CSV ya unificado (con columna de texto y columna de fuente)
```bash
python main.py -d datos_unificados.csv -c texto -s fuente -m sectur -l es -p okabe-ito
```

### 7. Ajuste de idioma, umbral de partición pequeña, outliers y muestreo rápido
```bash
python main.py -l en --min_partition_size 25 --contamination 0.05 --sample_limit 500 -t "Mi Análisis DMO vs UGC"
```

---

## Ayuda del CLI (`python main.py -h`)

```text
usage: main.py [-h] [-d DATASET] [-c TEXT_COLUMN] [-s SOURCE_COLUMN]
               [-m DMO_LABEL] [-l {es,en,fr}] [-t REPORT_TITLE]
               [-p {okabe-ito,viridis}] [--embedding_model EMBEDDING_MODEL]
               [--min_partition_size MIN_PARTITION_SIZE]
               [--contamination CONTAMINATION] [--sample_limit SAMPLE_LIMIT]
               [--ugc_dataset UGC_DATASET [UGC_DATASET ...]]
               [--ugc_text_col UGC_TEXT_COL [UGC_TEXT_COL ...]]
               [--ugc_label UGC_LABEL [UGC_LABEL ...]]
               [--dmo_dataset DMO_DATASET [DMO_DATASET ...]] [-o OUTPUT_DIR]
               [--dmo_text_col DMO_TEXT_COL [DMO_TEXT_COL ...]]

Sistema de Análisis Comparativo (XidooTopic): Imagen Proyectada (DMO) vs. Imagen Percibida.

options:
  -h, --help            show this help message and exit
  -d, --dataset DATASET
                        Ruta al archivo CSV unificado o crudo.
  -c, --text_column TEXT_COLUMN
                        Nombre de la columna que contiene los comentarios textuales (default: texto)
  -s, --source_column SOURCE_COLUMN
                        Nombre de la columna que identifica la fuente del comentario (default: fuente)
  -m, --dmo_label DMO_LABEL
                        Valor/etiqueta que identifica a la DMO en la columna de fuentes (default: sectur)
  -l, --language {es,en,fr}
                        Idioma objetivo del análisis (es: Español, en: Inglés, fr: Francés). Default: es
  -t, --report_title REPORT_TITLE
                        Título a desplegar en reportes y visualizaciones generadas.
  -p, --palette {okabe-ito,viridis}
                        Paleta de colores accesible apta para daltonismo. Default: okabe-ito
  --embedding_model EMBEDDING_MODEL
                        Modelo HuggingFace para xidootopic.TextEmbedder. Default: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
  --min_partition_size MIN_PARTITION_SIZE
                        Umbral mínimo para modelado de tópicos; por debajo usa frecuencia léxica. Default: 30
  --contamination CONTAMINATION
                        Proporción esperada de outliers en IsolationForestOutlierRemover. Default: 0.05
  --sample_limit SAMPLE_LIMIT
                        Límite opcional de muestreo estratificado por fuente para pruebas rápidas.
  --ugc_dataset UGC_DATASET [UGC_DATASET ...]
                        Uno o más CSVs de comentarios UGC (requiere --dmo_dataset).
  --ugc_text_col UGC_TEXT_COL [UGC_TEXT_COL ...]
                        Columna(s) de texto en --ugc_dataset (1 global o 1 por archivo; default: texto).
  --ugc_label UGC_LABEL [UGC_LABEL ...]
                        Etiqueta(s) de fuente para --ugc_dataset (1 por archivo; por defecto usa el nombre del archivo).
  --dmo_dataset DMO_DATASET [DMO_DATASET ...]
                        Uno o más CSVs de comentarios DMO (se fusionan en una sola fuente --dmo_label).
  -o, --output_dir OUTPUT_DIR
                        Directorio de salida donde se crearán los resultados (default: ./resultados junto a main.py).
  --dmo_text_col DMO_TEXT_COL [DMO_TEXT_COL ...]
                        Columna(s) de texto en --dmo_dataset (1 global o 1 por archivo; default: texto).
```

---

## Créditos y Referencias

- **XidooTopic** ([PyPI](https://pypi.org/project/xidootopic/) · [GitHub](https://github.com/JoseAdrianRodriguezGonzalez/xidooTOPIC)) — Autor: **Angel Diaz Pacheco** (`xidootopic==0.1.0`), utilizado para *embeddings* (`TextEmbedder`), reducción UMAP (`UMAPReducer`), detección de *outliers* (`IsolationForestOutlierRemover`), grafo KNN + clustering Leiden (`KNNGraphBuilder`, `LeidenClusterer`, `LeidenResolutionOptimizer`, `ClusterFilter`, `ClusterMerger`) y extracción de palabras clave y documentos representativos.
- **Análisis de sentimiento:** `nlptown/bert-base-multilingual-uncased-sentiment`.
- **Paletas accesibles (CVD-Safe):** **Okabe-Ito** (Nature Methods, 2011 / Color Universal Design) y **Viridis**.

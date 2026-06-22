# TFM-ML-Benchmark

Repositorio del TFM para comparar modelos de machine learning sobre el dataset SUPPORT2, con dos ramas principales:

- `df1`: grupo bio.
- `df2`: grupo tech.

El proyecto incluye el preprocesado de datos, analisis exploratorio, entrenamiento de modelos, y el manuscrito final en LaTeX.

## Estructura del repositorio

- `preprocessing/`: scripts en RMarkdown para generar los datasets procesados.
- `eda/`: analisis exploratorios y graficas auxiliares.
- `models/`: notebooks RMarkdown para entrenar y evaluar los modelos.
- `scripts/`: utilidades compartidas entre los notebooks.
- `data/raw/`: datos originales.
- `data/processed/`: datos ya limpios y listos para modelado.
- `outputs/`: resultados exportados, como tablas o recopilaciones de literatura.
- `manuscript/`: documento final en LaTeX.

## Flujo de trabajo

1. Preprocesar los datos con los archivos de `preprocessing/`.
2. Revisar exploracion y validaciones en `eda/`.
3. Entrenar modelos desde `models/`.
4. Generar o actualizar el manuscrito en `manuscript/`.

## Dependencias

El proyecto usa dos entornos distintos:

- R para el preprocesado, EDA, modelado y manuscrito.
- Python para el script de busqueda bibliografica en `scripts/search.py`.

### Dependencias de R

Para evitar conflictos con otras instalaciones de R, el proyecto instala los paquetes en una libreria local dentro del repositorio: `.R/library`.

Instalacion:

```bash
Rscript scripts/install_r_dependencies.R
```

Ese script lee `requirements-r.txt`, crea la libreria local si no existe y solo instala los paquetes que falten.

### Dependencias de Python

Recomendacion:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

En Windows:

```bash
.venv\Scripts\activate
pip install -r requirements.txt
```

## Reproducibilidad

El mecanismo recomendado es:

1. Clonar el repositorio.
2. Instalar los paquetes de R con `Rscript scripts/install_r_dependencies.R`.
3. Crear y activar el entorno virtual de Python.
4. Instalar las dependencias de Python con `pip install -r requirements.txt`.
5. Ejecutar los notebooks o scripts desde la raiz del proyecto.

La configuracion de RStudio esta en `TFM-ML-Benchmark.Rproj`, y el archivo `.Rprofile` del proyecto prioriza la libreria local cuando existe.

## Modelado

Los notebooks principales de modelado son:

- `models/bio_model_train.Rmd`
- `models/bio_model_train_downsampling.Rmd`
- `models/bio_model_train_no_cv.Rmd`
- `models/tech_model_train.Rmd`
- `models/tech_model_train_cut_columns.Rmd`
- `models/tech_model_train_no_cv.Rmd`

Todos usan las funciones compartidas de `scripts/modeling_utils_tidymodels.R`.

## Preprocesado

Los archivos de preprocesado principales son:

- `preprocessing/support2_ds1_proc.Rmd`
- `preprocessing/support2_ds2_proc.Rmd`
- `preprocessing/support2_proc.Rmd`

Estos generan los conjuntos ya limpios que luego se usan en `models/`.

## Manuscrito

El manuscrito final vive en `manuscript/main.tex` y se apoya en las secciones de `manuscript/sections/`.

Si quieres recompilarlo manualmente, normalmente basta con entrar en `manuscript/` y ejecutar:

```bash
latexmk -pdf main.tex
```

## Notas

- Evita editar a mano los archivos en `data/processed/` si provienen del flujo de preprocesado.
- Los archivos generados por LaTeX, HTML o PDF se excluyen en `.gitignore`.
- Si se agregan nuevas dependencias, actualiza `requirements-r.txt` o `requirements.txt` segun corresponda.

# Laboratorio de Nowcasting de Precipitacion

Este material esta pensado para estudiantes que no necesariamente programan todos los dias. La idea es ejecutar un flujo pequeno y reproducible para entender un problema central del nowcasting de precipitacion:

> un pronostico puede tener buen error promedio, pero fallar justo donde mas importa: lluvia intensa, nucleos localizados y desplazamiento espacial.

El laboratorio usa 5 secuencias de radar IDEAM en formato `.npy`, cada una con forma `(25, 128, 128)`. Usaremos los primeros 13 cuadros como contexto y los ultimos 12 cuadros como futuro a pronosticar.

## Estructura del curso

```text
nowcasting_course_lab/
├── README.md
├── environment.yml
├── inference_requirements.txt
├── config/
│   └── model_demo_paths.yaml
├── course_utils/
│   ├── data.py
│   ├── metrics.py
│   ├── model_inference.py
│   ├── palette.py
│   └── plotting.py
├── scripts/
│   ├── download_assets.py
│   ├── evaluate_predictions.py
│   ├── make_persistence_predictions.py
│   └── run_model_inference.py
├── notebooks/
│   ├── 01_walkthrough_nowcasting_practico.ipynb
│   ├── 02_tarea_nowcasting_metricas.ipynb
│   ├── 03_comparacion_modelos_post_inferencia.ipynb
│   ├── 04_tarea_mini_proyecto_nowcasting_avanzado.ipynb
│   ├── 05_tarea_piura_sophy_comparacion_interpretacion.ipynb
│   └── nowcasting_metrics_lab.ipynb
├── data/
│   └── samples/
├── outputs/
│   └── predictions/
│       ├── persistence/
│       ├── earthformer/
│       ├── cascast/
│       └── model/
└── checkpoints/
```

## 1. Instalar Python de forma sencilla

La opcion recomendada es instalar Miniconda o Mambaforge. Si ya tienes Anaconda, Miniconda, Mambaforge o Python funcionando, puedes saltar esta parte.

Opcion recomendada:

1. Descarga Miniconda desde <https://docs.conda.io/en/latest/miniconda.html>.
2. Instala Miniconda con las opciones por defecto.
3. Abre una terminal nueva.

En Linux o macOS, la terminal suele llamarse `Terminal`. En Windows, puedes usar `Anaconda Prompt`.

## 2. Crear el ambiente del curso

Desde la carpeta `nowcasting_course_lab`, ejecuta:

```bash
conda env create -f environment.yml
conda activate nowcasting-course-lab
python -m ipykernel install --user --name nowcasting-course-lab --display-name "Nowcasting Course Lab"
```

Si el ambiente ya existe y quieres actualizarlo:

```bash
conda env update -f environment.yml --prune
conda activate nowcasting-course-lab
```

Este ambiente es liviano y sirve para visualizacion, persistencia y metricas. No instala PyTorch ni CUDA.

## 2b. Extras opcionales para inferencia

No necesitas crear otro ambiente. Usa el mismo `nowcasting-course-lab` e instala extras solo si vas a correr EarthFormer/CasCast:

```bash
conda activate nowcasting-course-lab

# Opcion A: PyTorch con CUDA 11.8, recomendado si tienes NVIDIA.
conda install -c pytorch -c nvidia pytorch=2.0.1 torchvision pytorch-cuda=11.8

# Opcion B: PyTorch CPU-only, mas lento pero valido para pruebas/inferencia pequena.
# conda install -c pytorch pytorch=2.0.1 torchvision cpuonly

# Paquetes extra usados por CasCast.
pip install -r inference_requirements.txt
```

Si ya instalaste otro kernel por accidente, no pasa nada. Puedes seguir usando `nowcasting-course-lab` en Jupyter:

```bash
python -m ipykernel install --user --name nowcasting-course-lab --display-name "Nowcasting Course Lab"
```

Luego prueba una corrida pequena desde la linea de comandos. `--device auto` usa NVIDIA si existe y CPU si no:

```bash
python scripts/run_model_inference.py --stage earthformer --smoke --device auto
python scripts/run_model_inference.py --stage all --smoke --device auto
```

Si solo tienes CPU, empieza asi:

```bash
python scripts/run_model_inference.py --stage earthformer --smoke --device cpu --cpu-threads 8
python scripts/run_model_inference.py --stage all --smoke --device cpu --ddim-steps 2 --cpu-threads 8
```

Para procesar los 5 casos:

```bash
python scripts/run_model_inference.py --stage all --sample all --ddim-steps 20 --device auto
```

Nota: EarthFormer pesa alrededor de 100 MB y deberia ser razonable en CPU. CasCast pesa varios GB y puede caber en RAM, pero la inferencia por difusion es lenta en CPU porque ejecuta muchos pasos de denoising. Para CPU, usa primero `--ddim-steps 2` o `--ddim-steps 5`.

Si ves un aviso como `NVIDIA driver on your system is too old`, el script usara CPU con `--device auto`. Para forzar CPU y evitar confusiones:

```bash
python scripts/run_model_inference.py --stage earthformer --smoke --device cpu
```

Si ves un error relacionado con `hf_cache_home` de `huggingface_hub`, el wrapper del curso ya incluye una compatibilidad para el `diffusers` antiguo que viene dentro de CasCast. Actualiza este repo/archivo y vuelve a ejecutar el comando; no deberia ser necesario bajar de version `huggingface_hub`.

Las predicciones se guardan en:

```text
outputs/predictions/earthformer/
outputs/predictions/cascast/
```

Despues de generar predicciones, ejecuta la evaluacion comparativa:

```bash
python scripts/evaluate_predictions.py --sample all
```

Esto crea:

```text
outputs/evaluation/per_lead_continuous_metrics.csv
outputs/evaluation/per_lead_event_metrics.csv
outputs/evaluation/per_file_summary.csv
outputs/evaluation/overall_summary.csv
outputs/evaluation/figures/
```

## 3. Descargar las muestras desde Hugging Face

Las muestras estan publicadas aqui:

<https://huggingface.co/datasets/andrexandrex322/ideam-nowcasting-samples/tree/main/samples>

Para descargarlas automaticamente:

```bash
python scripts/download_assets.py
```

El script guardara los 5 archivos `.npy` en:

```text
data/samples/
```

Tambien revisara que cada secuencia tenga forma `(25, 128, 128)`.

## 4. Crear un pronostico base de persistencia

Antes de usar modelos profundos, usaremos un pronostico muy simple:

> persistencia = repetir el ultimo cuadro observado 12 veces.

Este pronostico base no es sofisticado, pero es excelente para aprender metricas porque permite comparar desplazamiento, suavizado y errores por intensidad.

Ejecuta:

```bash
python scripts/make_persistence_predictions.py
```

Los resultados quedaran en:

```text
outputs/predictions/persistence/
```

## 5. Abrir el notebook

Ejecuta:

```bash
jupyter lab
```

Luego abre:

```text
notebooks/01_walkthrough_nowcasting_practico.ipynb
```

Selecciona el kernel `Nowcasting Course Lab` si Jupyter lo solicita.

## 6. Orden recomendado de notebooks

1. `notebooks/01_walkthrough_nowcasting_practico.ipynb`
   - Demostracion progresiva para clase.
   - Carga un caso, grafica la tormenta, compara persistencia y calcula RMSE/CSI.
   - Incluye una seccion avanzada opcional para EarthFormer + Autoencoder + CasCast.

2. `notebooks/02_tarea_nowcasting_metricas.ipynb`
   - Tarea guiada para estudiantes.
   - Tiene formulas, pistas y celdas `TODO` para completar.
   - No depende de `course_utils.metrics`: los estudiantes implementan sus propias funciones y luego construyen un pipeline por archivo.

3. `notebooks/nowcasting_metrics_lab.ipynb`
   - Version resuelta o notebook de referencia para instructor.
   - Usa las mismas utilidades compartidas que los otros notebooks.

4. `notebooks/03_comparacion_modelos_post_inferencia.ipynb`
   - Se usa despues de correr `scripts/run_model_inference.py`.
   - Compara persistencia, EarthFormer y CasCast con las mismas metricas.
   - Resume metricas por archivo, por modelo y por umbral.

5. `notebooks/04_tarea_mini_proyecto_nowcasting_avanzado.ipynb`
   - Mini-proyecto de 10 horas.
   - Incluye auditoria del dataset, comparacion de modelos, pooling espacial, metricas de objetos, extremos, baselines opcionales e incertidumbre.
   - Material de profundizacion opcional, no evaluado en el modulo corto.

6. `notebooks/05_tarea_piura_sophy_comparacion_interpretacion.ipynb`
   - Segunda tarea evaluada.
   - Usa unicamente datos Sophy/Piura.
   - Continua la tarea `02` con comparacion de modelos, resumen por archivo e interpretacion de extremos simples.

## 7. Que haras en el laboratorio

El laboratorio guia paso a paso:

1. Cargar una secuencia de precipitacion.
2. Separar `inputs = sequence[:13]` y `target = sequence[13:25]`.
3. Cargar una prediccion. Si no hay predicciones de modelo, se usa persistencia.
4. Visualizar entrada, objetivo, prediccion y error absoluto.
5. Calcular metricas continuas por tiempo de pronostico:
   - MAE
   - RMSE
   - Bias
   - Correlacion de Pearson
6. Calcular metricas de eventos de lluvia usando umbrales:
   - 0.5 mm/h: lluvia ligera
   - 2.0 mm/h: lluvia moderada
   - 5.0 mm/h: lluvia fuerte
   - 10.0 mm/h: lluvia intensa
7. Interpretar por que RMSE no basta para evaluar lluvia intensa.

## 8. Utilidades compartidas

Los notebooks usan `course_utils/` para mantener consistencia:

- `palette.py`: paleta discreta de lluvia en mm/h y paleta de error.
- `data.py`: carga de muestras, limpieza de NaN, split 13->12 y predicciones.
- `metrics.py`: MAE, RMSE, bias, correlacion, CSI, POD, FAR, F1 para notebooks resueltos o instructor.
- `plotting.py`: figuras compartidas para eventos, metricas y paneles target/prediccion.
- `model_inference.py`: demo avanzado opcional con checkpoints y GPU.
- `evaluation.py`: evaluacion comparativa de predicciones guardadas.

Esto evita copiar y pegar codigo entre notebooks.

## 9. Opcion avanzada: checkpoints del modelo

Los checkpoints de EarthFormer, Autoencoder y CasCast estan publicados aqui:

<https://huggingface.co/andrexandrex322/ideam-nowcasting-earthformer-cascast/tree/main>

Son archivos grandes. El checkpoint de difusion pesa varios GB, por eso no se requieren para el laboratorio basico.

Si el instructor quiere descargarlos para una extension avanzada:

```bash
python scripts/download_assets.py --checkpoints
```

Los archivos quedaran en:

```text
checkpoints/
```

Este notebook basico no ejecuta inferencia completa EarthFormer + Autoencoder + Difusion. Su objetivo principal es entender la evaluacion de nowcasting.

La configuracion del demo avanzado esta en:

```text
config/model_demo_paths.yaml
```

Por defecto apunta a:

```text
../codigo_github/Repo_CasCast
```

En el walkthrough, cambia `RUN_ADVANCED_MODEL_DEMO = True` solo si estas en un ambiente con PyTorch, CUDA y las dependencias de CasCast. Para uso mas reproducible, preferimos los scripts:

```bash
python scripts/run_model_inference.py --stage all --sample all --ddim-steps 20 --device auto
python scripts/evaluate_predictions.py --sample all
```

## 10. Entregables sugeridos

Cada estudiante entrega:

1. Notebook ejecutado: `nowcasting_metrics_lab.ipynb`.
2. Una figura con entrada, objetivo, prediccion y error.
3. Una tabla de MAE, RMSE, bias y correlacion por tiempo de pronostico.
4. Una tabla de CSI, POD, FAR y F1 por umbral.
5. Una reflexion corta de 300 a 500 palabras.

## 11. Rubrica sugerida

| Componente | Puntos |
|---|---:|
| Carga y visualizacion de datos | 20 |
| Metricas continuas implementadas correctamente | 20 |
| Metricas de eventos implementadas correctamente | 25 |
| Interpretacion por tiempo y umbral | 25 |
| Claridad y reproducibilidad | 10 |
| Total | 100 |

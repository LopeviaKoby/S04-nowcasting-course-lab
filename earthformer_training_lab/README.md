# EarthFormer Training Lab

Este modulo es una version docente del entrenamiento EarthFormer usado por el
pipeline IDEAM/CasCast.

## Relacion con el repo original

En el repo original, el flujo empieza en:

```text
run_ideam_train.sh
```

y llama:

```text
train.py -c configs/sevir_used/EarthFormer_ideam.yaml
```

Las piezas principales del flujo original son:

```text
configs/sevir_used/EarthFormer_ideam.yaml
datasets/ideam_used.py
networks/earthformer_xy.py
models/non_ar_model.py
models/model.py
utils/builder.py
```

Para el curso se dejo una version mas pequena:

```text
config/earthformer_ideam_course.yaml
data.py
model.py
train.py
architectures/earthformer_xy.py
```

## Que se mantiene

- La arquitectura real `EarthFormer_xy`.
- La convencion 13 cuadros de entrada -> 12 cuadros futuros.
- La normalizacion de lluvia por `max_rain = 60.0`.
- La perdida tipo `WeightedMSELoss`.
- El guardado de checkpoints `.pth`.

## Que se simplifica

- No se usa DDP.
- No se usa `MultiPoolBatchSampler`.
- No se usa logging distribuido ni Ceph.
- Las metricas de validacion se reducen a RMSE, MAE y bias para explicar el ciclo.

El objetivo es pedagogico: que los estudiantes puedan leer el entrenamiento de
principio a fin antes de volver al pipeline completo de investigacion.

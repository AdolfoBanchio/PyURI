# PyURI

Estructura del proyecto para la implementacion en pytorch del Tap Withdrawal Circuit (TWC) utilizando el modelo Fiuri como la dinamica neuronal. Todos los modulos se encuentran bajo el directorio src/.

- Ariel : Implementación original
- fiuri: Implementacion en pytorch del modelo Fiuri y sus conexiones sinapticas
- mlp: definicion de una red tradicional MLP para ser usada como critico en el algoritmo DDPG/TD3.
- td3: Implementacion del algoritmo TD3 para el entrenamiento del TWC+Fiuri en entornos continuos.
- twc: funciones que definen su interaccion con el entorno.
- utils: utilidades varias para el proyecto
  
Se desarrollo un test que compara los valores de salida entre la implementacion original en Ariel y ambas implementaciones en pytorch (version normal y version suave).

```
python3 tests/twc_validation.py
```
### Scripts de entrenamiento

- Para entrenar el TWC+Fiuri utilizando TD3 en el entorno MCC se puede ejecutar el siguiente comando: 

```
python3 scripts/twc_mcc_td3.py <ruta a config.json>
```

Donde el archivo .json debe contener los campos de configuracion de hiperparametros para el entrenamiento con TD3. Siguiendo la clase TD3Config definida en src/td3_flat/td3_flat.py

Para comenzar la busqueda de hiperparametros con optuna se debe correr el siguiente comando:

Ahora mismo este script busca resolver el MCC y encola 5 configuraciones de "calentamiento" para el pruner de optuna.

```
python3 scripts/twc_mcc_td3_optuna.py
```

Para evaluar los modelos obtenidos luego de un entrenamiento:

```
python3 scripts/evaluate_twc_mcc.py <ruta a carpeta creada de la corrida>
```

Dicha ruta se genera en el directorio out/runs/td3_mcc_twc.
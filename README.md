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

Esto guardara los resultados del test en out/tests/twc_validation.

**Decisiones de diseño tomadas a la hora te integrar TWC en pytorch:**

El TWC esta implementado dentro del modulo twc. En twc_builder.py se enceuntra la calse TWC que define el modulo de pytorch con la estructura del TWC. La estructura tiene dos "modos" de ejecicion. Uno donde el estado es manejado por si mismo y de manera interna, y otro donde el estado es manejado externamente y se le pasa como parametro. Esto es necesario para poder realizar entrenamiento haciendo uso de BPTT y TD3.

Ya que durante el entrenamiento tenemos un episodio activo, donde nuestro agente va realizando acciones y recibiendo observaciones del entorno aqui es donde queremos que el estado se conserve a lo largo de la ejecucion del episodio, pero ademas en cada paso de tiempo queremos poder realizar un paso de actualizacion utilizando BPTT, que requiere de correr el modelo por una secuencia de largo (BURN_IN + SEQ_LEN) y luego calcular los gradientes y realizar backpropagation. Lo ideal es que esta secuencia no nos interrumpa el estado interno generado por la red en el episodio activo. Por lo tanto en este paso se utiliza el modo donde el estado es manejado externamente.

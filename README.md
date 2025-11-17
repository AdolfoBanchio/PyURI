# PyURI

Implementacion en pytorch del Tap Withrawal Circuit (TWC) que utiliza el modelo Fiuri como el modelo de la dinamica neuronal.

La implementación original esta en el modulo 'Ariel' dentro de src.

La integracion del TWC+Fiuri en pytorch se divide en dos modulos.

1. Fiuri: Implementacion de la dinamica neuronal Fiuri en pytorch y sus respectivos tipos de conexiones sinapticas.
2. TWC: Implementacion del Tap Withdrawal Circuit (TWC) en pytorch utilizando el modelo Fiuri como la dinamica neuronal.


Decisiones de diseño tomadas a la hora te integrar Fiuri en pytorch:
- El estado interno/externo de cada neurona es manejado por el modulo que contenga a la neurona. Es decir, para poder hacer un paso de una capa neuronal Fiuri esta recibe su estado actual como un parametro y lo retorna actualizado.
- Las conexiones sinapticas inhibitorias y exitatorias son representadas como una capa densa tradicional entre dos capas Fiuri. Pero contiene una mascara para definir que conexiones estan activas y cuales no. (Estas mascaras son constantes y no se actualizan durante el entrenamiento).
- Las 'gap junctions' son representadas como una conexion dispersa que contiene los indices de las neuronas que estan conectadas entre si. (Estas conexiones son constantes y no se actualizan durante el entrenamiento). Como estas conexiones su valor depende de los estados internos, los valores finales que influyen en cada neurona se calculan dentro de la capa neuronal Fiuri.
- La funcion que define el estado externo de una neurona originalemente corresponde a una funcion ReLU. Donde ReLU(Sn -Tn) == max(0, Sn -Tn). Para esta implementacion donde neceistamos que los gradientes fluyan correctamente y no se queden atascados en 0. Como note que la condicion Sn > Tn no ocurre con demasiada frecuencia, entonces el gradiente para el parametro Tn se vuelve 0 la mayoria del tiempo. Para solucionar esto, se utiliza la funcion suavizada LeakyReLU en lugar de ReLU. De esta manera, los gradientes pueden fluir correctamente a traves del parametro Tn en cualquier condicion.
- Para facilitar la integracion con pytorch y reducir los costos computacionales, a la hora de actualizar el estado intenro, en lugar de realizar la comparacion exacta Sn == En, se los compara con un cierto margen de tolerancia (epsilon).

Se desarrollo un test que compara los valores de salida entre la implementacion original en Ariel y la implementacion en pytorch.

```
python3 tests/twc_validation.py 
```

Esto guardara los resultados del test en out/tests/twc_validation.

Decisiones de diseño tomadas a la hora te integrar TWC en pytorch:
El TWC esta implementado dentro del modulo twc. En twc_builder.py se enceuntra la calse TWC que define el modulo de pytorch con la estructura del TWC.

- El modelo completo es el encargado de manejar los estados internos y externos de todas las neuronas Fiuri que componen el TWC.
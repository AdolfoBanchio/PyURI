# PyURI

Estructura del proyecto para la implementacion en pytorch del Tap Withdrawal Circuit (TWC) utilizando el modelo Fiuri como la dinamica neuronal. Todos los modulos se encuentran bajo el directorio src/.

- Ariel : Implementación original
- ddpg: Implementacion del algoritmo DDPG para el entrenamiento del TWC+Fiuri en entornos continuos.
- fiuri: Implementacion en pytorch del modelo Fiuri y sus conexiones sinapticas
- mlp: definicion de una red tradicional MLP para ser usada como critico en el algoritmo DDPG/TD3.
- td3: Implementacion del algoritmo TD3 para el entrenamiento del TWC+Fiuri en entornos continuos.
- twc: Implementacion en pytorch del Tap Withdrawal Circuit (TWC) utilizando el modelo Fiuri como la dinamica neuronal. Asi como tambien las funciones que definen su interaccion con el entorno.
- utils: utilidades varias para el proyecto (funciones de ayuda, logging, etc).
Implementacion en pytorch del Tap Withrawal Circuit (TWC) que utiliza el modelo Fiuri como el modelo de la dinamica neuronal.


**Decisiones de diseño tomadas a la hora te integrar Fiuri en pytorch:**

- El estado interno/externo de cada neurona es manejado por el modulo que contenga a la neurona. Es decir, para poder hacer un paso de una capa neuronal Fiuri esta recibe su estado actual como un parametro y lo retorna actualizado.
- Las conexiones sinapticas inhibitorias y exitatorias son representadas como una capa densa tradicional entre dos capas Fiuri. Pero contiene una mascara para definir que conexiones estan activas y cuales no. (Estas mascaras son constantes y no se actualizan durante el entrenamiento).
- Las 'gap junctions' son representadas como una conexion dispersa que contiene los indices de las neuronas que estan conectadas entre si. (Estas conexiones son constantes y no se actualizan durante el entrenamiento). Como estas conexiones su valor depende de los estados internos, los valores finales que influyen en cada neurona se calculan dentro de la capa neuronal Fiuri.
- La funcion que define el estado externo de una neurona originalemente corresponde a una funcion ReLU. Donde ReLU(Sn -Tn) == max(0, Sn -Tn). Para esta implementacion donde neceistamos que los gradientes fluyan correctamente y no se queden atascados en 0. Como note que la condicion Sn > Tn no ocurre con demasiada frecuencia, entonces el gradiente para el parametro Tn se vuelve 0 la mayoria del tiempo. Para solucionar esto, se utiliza la funcion suavizada LeakyReLU en lugar de ReLU. De esta manera, los gradientes pueden fluir correctamente a traves del parametro Tn en cualquier condicion.
- Para facilitar la integracion con pytorch y reducir los costos computacionales, a la hora de actualizar el estado intenro, en lugar de realizar la comparacion exacta Sn == En, se los compara con un cierto margen de tolerancia (epsilon).
- Debido a que la especificacion en pytorch de la dinamica neuronal genera muchas 'ramas' de ejecucion esto genero que muchas veces los gradientes de los parametross (D y T) se volvieran casi nulos ya que solo fluye gradiente a  traves de ellos cuando la rama donde pertenecen se activa. Por lo tanto en el mismo modulo fiuri, se implemento una version alternativa de la dinamica neuronal, una version 'suave'. Donde se busca reemplazar las operaciones de comparacion exacta (==, >) por operaciones continuas que permitan el flujo de gradientes en todo momento. (surrogate gradients o gradientes sustitutos). Esta version alternativa se puede activar al crear el TWC con el parametro use_v2=True.
  
Se desarrollo un test que compara los valores de salida entre la implementacion original en Ariel y ambas implementaciones en pytorch (version normal y version suave).

```
python3 tests/twc_validation.py
```

Esto guardara los resultados del test en out/tests/twc_validation.

**Decisiones de diseño tomadas a la hora te integrar TWC en pytorch:**

El TWC esta implementado dentro del modulo twc. En twc_builder.py se enceuntra la calse TWC que define el modulo de pytorch con la estructura del TWC. La estructura tiene dos "modos" de ejecicion. Uno donde el estado es manejado por si mismo y de manera interna, y otro donde el estado es manejado externamente y se le pasa como parametro. Esto es necesario para poder realizar entrenamiento haciendo uso de BPTT y TD3.

Ya que durante el entrenamiento tenemos un episodio activo, donde nuestro agente va realizando acciones y recibiendo observaciones del entorno aqui es donde queremos que el estado se conserve a lo largo de la ejecucion del episodio, pero ademas en cada paso de tiempo queremos poder realizar un paso de actualizacion utilizando BPTT, que requiere de correr el modelo por una secuencia de largo (BURN_IN + SEQ_LEN) y luego calcular los gradientes y realizar backpropagation. Lo ideal es que esta secuencia no nos interrumpa el estado interno generado por la red en el episodio activo. Por lo tanto en este paso se utiliza el modo donde el estado es manejado externamente.

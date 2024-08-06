# FWI acústico incorporando redes neuronales convolucionales y un modelo de difusión regularizado

### Paquetes requeridos

Se recomienda el uso de Pip (https://pip.pypa.io/en/stable/) para la instalación de cada uno de los paquetes así como 
el uso de un ambiente virtual como venv (https://docs.python.org/es/3/library/venv.html) para el manejo de los mismos

-> Devito 4.8.8 (https://www.devitoproject.org/). **Tutoriales y ejemplos en: https://github.com/devitocodes/transform22**

-> Numpy 2.0.0 (instalado con devito, sino instalar según https://numpy.org/)

-> Matplotlib 3.9.1 (https://matplotlib.org/stable/install/index.html)

-> Scipy 1.14.0 (https://scipy.org/)

-> Simpy 1.12.1 (https://simpy.readthedocs.io/en/latest/simpy_intro/installation.html)

## Teoría

La inversión de onda completa (FWI) es una metodología para la construcción de modelos de velocidad del subsuelo a partir de 
datos sísmicos observados adquiridos en campo y un modelo de velocidad incial a partir del cual modelan datos sísmicos sintéticos ($d_{i}^{syn}$)
los cuales son comparados con los datos observados ($d_{i}^{obs}$) mediante una función de costo (generalmente la norma l2 de los residuos)
que es reducida iterativamente. Esto se logra mediante el cálculo del gradiente el cual consiste en la derivada de la función de costo e indica la dirección que debe tomar el modelo
en orden de reducir dicha función. El gradiente es sumado o restado, según convenga, al modelo inicial (en la primera iteración) generando un modelo actualizado a partir del cual
se repite todos el ciclo en bucles hasta lograr la minimización de la función de costo:

$x(m)=\frac{1}{2} \sum_{i=0}^{n_s} \lvert \lvert d_{i}^{syn}-d_{i}^{obs} \lvert \lvert _2 ^2$


El objetivo de la inversión es minimizar esta función para que los datos sintéticos concuerden con los datos reales y por tanto el modelo invertido final sea lo más
parecido posible al modelo verdadero (desconocido en situaciones reales). La generación de los datos sintéticos se basa en el modelamiento directo.

![FWIp](https://github.com/user-attachments/assets/e01a5930-5f47-47ae-a5a2-9edcdbdb01a8)

### Modelamiento directo (forward modelling)

En sísmica de reflexión existe un conjunto de fuentes $s$ que se accionan generando ondas sísmicas de cuerpo (P y S) y superficiales que se propagan por el subsuelo y por superficie
, respectivamente. Las ondas de cuerpo, tras chocar con cada una de las interfaces o estratos de las estructuras geológicas del subsuelo, se devuelven hasta superficie en donde se 
situa la geometría de adquisición (receptores) que recopilan la información sísmica. Con esto, es posible generar datos sísmicos conocidos como shot gathers cuyas dimensiones son
tiempo x distancia horizontal y consisten en la conjunción de múltiples trazas.

**Shot gather**

![imagen](https://github.com/user-attachments/assets/2ec687f8-be77-432e-80da-9b44ecc1b0e0)

**Método sísmico**

![imagen](https://github.com/user-attachments/assets/9a65b31f-75ea-418a-89bc-f58b89aaf190)

El modelamiento directo consiste en la resolución de la ecuación de onda acústica:

$\bigtriangledown^2 p-\frac{1}{c^2}\frac{\partial^2{p}}{\partial{t^2}}=s$

donde $p$ es el campo de presión (datos sísmicos), $c$ es la velocidad del medio (a veces, se representa la ecuación en terminos de lentitud $m=\frac{1}{c²}$, $t$ es el tiempo y 
$s$ es el factor fuente. 

Existen distintos ecuaciones matemáticas para realizar el modelamiento/simulación siendo el método de diferencias finitas uno de los más usados.

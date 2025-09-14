# Proyecto_final_mlops
1) Descripción corta

Construimos un modelo para detectar billetes falsos usando el dataset Banknote Authentication de UCI. El dataset ya viene en formato tabular (no hay imágenes): cada fila representa un billete y sus columnas son estadísticos calculados sobre la textura del billete (a partir de una transformada wavelet).

2) Problema y objetivo de negocio

Dolor del negocio: las entidades pueden sufrir pérdidas por aceptar billetes falsos en caja o cajeros.
Objetivo: reducir el riesgo operativo y las pérdidas mediante un clasificador que ayude a identificar billetes falsos.

3) Dataset y fuente

Nombre: Banknote Authentication

Fuente (UCI): https://archive.ics.uci.edu/ml/datasets/banknote+authentication

Descarga directa (CSV):
https://archive.ics.uci.edu/ml/machine-learning-databases/00267/data_banknote_authentication.txt

Tamaño aproximado: 1.372 filas × 5 columnas

4) Significado de columnas

| Columna      | ¿Qué mide? (intuitivo)                         | Pista típica para fraude                                          |
| ------------ | ---------------------------------------------- | ----------------------------------------------------------------- |
| **variance** | **Contraste/variación** de la textura          | Auténticos suelen tener **> 0**; falsos, **< 0**                  |
| **skewness** | **Inclinación** de la distribución (cola)      | Auténticos: **positiva** (cola derecha); falsos: **0 o negativa** |
| **curtosis** | **Pico** y **colas largas** (valores extremos) | **Alta** aparece más en **falsos**                                |
| **entropy**  | **Desorden/variedad** de la textura            | Muy parecida en ambos; **no separa sola**                         |
| **class**    | Etiqueta objetivo                              | **0 = auténtico**, **1 = falso**                                  |


variance:mide cuánta dispersión tienen los coeficientes, valores altos → mucha variación/contraste en la textura; valores bajos → textura más uniforme.
skewness:mide si la distribución es simétrica o si tiene cola más larga a un lado, mucha asimetría = presencia de coeficientes atípicos (bordes/texturas raras).
curtosis:mide cuán “picuda” es la distribución y qué tan pesadas son sus colas, curtosis alta sugiere patrones muy marcados en la textura (posible señal de falsificación).
entropy: mide complejidad de los coeficientes,más cerca de 0 = mayor entropía.

5) Resultados del EDA

Tamaño y calidad de datos

Filas: 1.372 · Columnas: 5

Nulos: 0 (no se requieren imputaciones)

|     Clase | Significado |    Conteo |         % |
| --------: | ----------- | --------: | --------: |
|         0 | Auténtico   |   **762** | **55.5%** |
|         1 | Falso       |   **610** | **44.5%** |
| **Total** | —           | **1.372** |  **100%** |

variance:Los billetes auténticos suelen tener variance positiva (mayores a 0) y Los billetes falsos suelen tener variance negativa (menores a 0).
skewness:Los billetes auténticos suelen tener asimetría hacia la derecha,los falsos suelen tener asimetría neutra o hacia la izquierda  
curtosis:Los billetes falsos presentan más casos con curtosis muy alta —es decir, picos muy marcados y valores extremos en la textura—, mientras que los auténticos rara vez alcanzan esos extremos. 
entropía: tienen cajas muy parecidas, no se distingue bien entre auténticos y falsos: los dos grupos se comportan casi igual.

#conclusión: 
#variance > 0 y skewness > 0 → más probable Auténtico.
#variance < 0 y/o curtosis alta → más probable Falso.
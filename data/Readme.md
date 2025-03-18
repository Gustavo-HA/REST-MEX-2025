# Rest-Mex 2025
# Sentiment Analysis Magical Mexican Towns Training Corpus
-*- coding: utf-8 -*-

## English
The dataset is located entirely in the file `Rest-Mex_2025_train.csv`.
The dataset consists of 208,051 rows (70% of the original dataset. The remaining 30% will be used as a test set), one for each opinion. Each row contains 6 columns:

1. **Title**: The title that the tourist assigned to their opinion. Data type: Text.
2. **Review**: The opinion issued by the tourist. Data type: Text.
3. **Polarity**: The label representing the sentiment polarity of the opinion. Data type: [1, 2, 3, 4, 5].
4. **Town**: The town where the review is focused. Data type: Text.
5. **Region**: The region (state in Mexico) where the town is located. Data type: Text. This feature is not intended to be classified but is provided as additional information that could be leveraged in classification models.
6. **Type**: The type of place the review refers to. Data type: [Hotel, Restaurant, Attractive].

### Polarity Explanation
The polarity score ranges from 1 (most dissatisfied) to 5 (most satisfied) and can be interpreted as follows:

1. Very bad
2. Bad
3. Neutral
4. Good
5. Very good

### Statistics:

#### Polarity
```
Class | Instances
------------------
    1 |   5,441
    2 |   5,496
    3 |  15,519
    4 |  45,034
    5 | 136,561
------------------
Total | 208,051
```

#### Type
```
     Class | Instances
-----------------------
     Hotel |  51,410
Restaurant |  86,720
Attractive |  69,921
-----------------------
     Total | 208,051
```

#### Top 40 Selected Towns
```
    Town                           Count
----------------------------------------
 1. Tulum                         45,345
 2. Isla Mujeres                  29,826
 3. San Cristóbal de las Casas    13,060
 4. Valladolid                    11,637
 5. Bacalar                       10,822
 6. Palenque                       9,512
 7. Sayulita                       7,337
 8. Valle de Bravo                 5,959
 9. Teotihuacan                    5,810
10. Loreto                         5,525
11. Todos Santos                   4,600
12. Pátzcuaro                      4,454
13. Taxco                          4,201
14. Tlaquepaque                    4,041
15. Ajijic                         3,752
16. Tequisquiapan                  3,627
17. Metepec                        3,532
18. Tepoztlán                      3,445
19. Cholula                        2,790
20. Tequila                        2,650
21. Orizaba                        2,521
22. Izamal                         2,041
23. Creel                          1,786
24. Ixtapan de la Sal              1,696
25. Zacatlán                       1,602
26. Huasca de Ocampo               1,509
27. Mazunte                        1,466
28. Xilitla                        1,458
29. Atlixco                        1,444
30. Malinalco                      1,429
31. Bernal                         1,252
32. Tepotzotlán                    1,013
33. Cuetzalan                         996
34. Chiapa de Corzo                  960
35. Parras                            953
36. Dolores Hidalgo                   909
37. Coatepec                          818
38. Cuatro Ciénegas                   788
39. Real de Catorce                   760
40. Tapalpa                           725
----------------------------------------
Total | 208,051
```

### Important
All participants who registered for Rest-Mex 2025 and those who access this data (regardless of participation in the forum) agree to use the data obtained from REST-MEX exclusively for academic and research purposes. Any other use of the data is at their own risk.

---

## Spanish (Español)
El conjunto de datos se encuentra completamente en el archivo `Rest-Mex_2025_train.csv`.
El conjunto de datos consta de 208,051 renglones (el 70% del conjunto de datos original. El otro 30% será utilizado como conjunto de prueba), uno por cada opinión. Cada renglón contiene 6 columnas:

1. **Title**: El título que el propio turista le otorgó a su opinión. Tipo de dato: Texto.
2. **Review**: La opinión emitida por el turista. Tipo de dato: Texto.
3. **Polarity**: La etiqueta que representa la polaridad de la opinión. Tipo de dato: [1, 2, 3, 4, 5].
4. **Town**: El pueblo mágico sobre el cual se emite la opinión. Tipo de dato: Texto.
5. **Region**: La región (estado de la república mexicana) donde se ubica el pueblo mágico. Tipo de dato: Texto. Esta característica no será clasificada, pero se proporciona como información adicional que podría ser aprovechada en los modelos de clasificación.
6. **Type**: El tipo de lugar del cual se emite la opinión. Tipo de dato: [Hotel, Restaurant, Attractive].

### Explicación de la Polaridad
La polaridad varía de 1 (mayor grado de insatisfacción) hasta 5 (mayor grado de satisfacción). Se puede interpretar de la siguiente manera:

1. Muy malo
2. Malo
3. Neutral
4. Bueno
5. Muy bueno

### Importante
Todos los participantes que se registraron en Rest-Mex 2025 y aquellos que accedan a estos datos (sin importar que participen en el foro o no) aceptan utilizar los datos obtenidos de REST-MEX exclusivamente con fines académicos y de investigación. Cualquier otro uso de los datos será bajo su propia responsabilidad.


[English Version](README.md)

# Proyecto 3 - Clasificación de Dígitos Manuscritos con una CNN (MNIST)

## Objetivo

Construir y entrenar una Red Neuronal Convolucional (CNN) con PyTorch para clasificar dígitos manuscritos del conjunto de datos MNIST.

En comparación con el proyecto 2, que usaba un Perceptrón Multicapa y trabajaba directamente sobre píxeles aplanados, este proyecto conserva la estructura espacial de la imagen y usa capas convolucionales para aprender patrones visuales locales como bordes, trazos y curvas.

El objetivo no fue solamente obtener un buen clasificador, sino entender por qué las CNN son una mejor opción que una red totalmente conectada para datos de imagen.

---

# Dataset

El modelo se entrena y se evalúa sobre el dataset **MNIST**.

- 60,000 imágenes de entrenamiento
- 10,000 imágenes de prueba
- Imágenes en escala de grises
- Resolución: **28 × 28 píxeles**
- 10 clases (dígitos del 0 al 9)

Cada imagen se transforma en un tensor con forma:

```
1 × 28 × 28
```

A diferencia del MLP del proyecto 2, la imagen **no se aplana al inicio**. La CNN conserva la disposición espacial para que las capas convolucionales trabajen sobre vecindades locales de píxeles.

---

# Arquitectura del Modelo

La red implementada es una CNN pequeña construida con `nn.Sequential`.

```
Entrada (1×28×28)
↓
Conv2d (1 → 16, kernel=3)
↓
ReLU
↓
MaxPool2d (2)
↓
Conv2d (16 → 32, kernel=3)
↓
ReLU
↓
MaxPool2d (2)
↓
Flatten
↓
Linear (800 → 128)
↓
ReLU
↓
Linear (128 → 10)
↓
Logits
```

La diferencia principal con el proyecto 2 es que las primeras capas ya hacen extracción de características.

- `Conv2d` aprende filtros que detectan patrones visuales locales.
- `MaxPool2d` reduce el tamaño espacial y conserva las respuestas más fuertes.
- `Flatten` convierte los mapas de características en un vector solo después de que la parte convolucional ya extrajo información útil.

Eso hace que el modelo sea más eficiente para imágenes que un MLP puro.

---

# ¿Por qué usar CNN y no MLP?

El MLP del proyecto 2 trabaja con píxeles aplanados, así que trata a la imagen como un vector largo y no aprovecha la estructura 2D de los dígitos.

En cambio, esta CNN:

- Conserva la localidad espacial en las primeras capas.
- Aprende filtros reutilizables en lugar de una conexión independiente para cada píxel.
- Usa menos parámetros que una red totalmente conectada de capacidad comparable.
- Suele generalizar mejor en tareas de imágenes.

Esa es la idea central que deja más clara este proyecto.

---

# Flujo de Entrenamiento

El ciclo de entrenamiento sigue el mismo flujo general de PyTorch que en el proyecto 2, pero ahora aplicado a un modelo convolucional.

1. Poner el modelo en modo entrenamiento con `model.train()`.
2. Cargar un batch de imágenes y etiquetas desde el `DataLoader`.
3. Limpiar los gradientes anteriores con `optimizer.zero_grad()`.
4. Ejecutar el forward pass a través de la CNN.
5. Calcular la pérdida con `CrossEntropyLoss`.
6. Ejecutar `loss.backward()` para obtener los gradientes.
7. Actualizar los parámetros con `optimizer.step()`.
8. Guardar loss y accuracy por época.

La diferencia importante respecto al proyecto 2 es que no solo se optimizan pesos densos, sino también kernels convolucionales que aprenden características visuales directamente.

---

# Función de Pérdida

El modelo utiliza:

```python
nn.CrossEntropyLoss()
```

Tal como en el proyecto 2, esta pérdida es la correcta para clasificación multiclase con logits crudos.

Combina:

- Softmax
- Negative Log Likelihood

en una sola operación numéricamente estable.

---

# Optimizador

El entrenamiento usa Adam:

```python
optim.Adam(model.parameters(), lr=0.001)
```

En comparación con el SGD usado en el proyecto 2, Adam adapta la tasa de aprendizaje para cada parámetro y suele converger más rápido y de forma más suave en este tipo de CNN.

---

# Rendimiento

Después de entrenar durante 10 épocas, el modelo alcanza una precisión muy alta sobre MNIST.

Las curvas de entrenamiento muestran una caída rápida del loss y un aumento estable de la accuracy:

![Pérdida y accuracy de entrenamiento](images/loss_acc_hty.png)

El rendimiento final es consistente con una CNN bien entrenada sobre MNIST, y la matriz de confusión muestra que la mayoría de los errores aparecen solo entre dígitos visualmente similares.

![Matriz de confusión](images/conf_matrix.png)

---

# Qué Aprendió la CNN

Una de las partes más útiles del proyecto fue visualizar lo que la red aprendió internamente.

La primera capa convolucional aprende filtros parecidos a bordes y trazos:

![Filtros aprendidos](images/filtros.png)

Cuando esos filtros se aplican sobre una imagen de ejemplo, los mapas de activación resaltan distintas partes del dígito:

![Mapas de activación de ejemplo](images/filtro_ej.png)

Eso ayuda a entender cómo la red pasa de píxeles crudos a una representación más abstracta de la forma.

---

# Evaluación

La etapa de evaluación sigue la misma idea que en el proyecto 2, pero aquí es especialmente útil inspeccionar la confianza de la CNN y cómo cambia su salida según el dígito.

El notebook evalúa el modelo sobre el loader completo de test y también sobre imágenes individuales.

- Accuracy completa sobre test
- Predicción de una sola imagen
- Logits crudos
- Probabilidades Softmax
- Matriz de confusión

Esta combinación permite verificar no solo que el modelo acierta, sino también que está aprendiendo representaciones útiles.

---

# Conceptos Aprendidos

Este proyecto introdujo varios conceptos importantes de CNN y reforzó el flujo de trabajo de PyTorch visto en el proyecto 2:

- Capas convolucionales
- Filtros aprendibles
- Campos receptivos locales
- Extracción espacial de características
- Max pooling
- Mapas de activación
- Flatten después de la convolución
- Diseño de clasificadores CNN
- Logits para clasificación multiclase
- CrossEntropy loss
- Optimizador Adam
- Modo entrenamiento vs modo evaluación
- Seguimiento de accuracy
- Análisis con matriz de confusión
- Visualización de filtros y activaciones

---

# Comparación con el Proyecto 2

El proyecto 2 y el proyecto 3 resuelven el mismo problema de clasificación en MNIST, pero con sesgos inductivos distintos.

- El proyecto 2 usa un MLP y parte de píxeles aplanados.
- El proyecto 3 usa una CNN y conserva la estructura de la imagen.
- El proyecto 2 aprende relaciones densas globales de forma directa.
- El proyecto 3 primero aprende patrones visuales locales y luego los combina en características de nivel dígito.
- El proyecto 2 es más simple como línea base.
- El proyecto 3 es un enfoque más natural y potente para clasificación de imágenes.

Ver ambos lado a lado hace mucho más clara la ventaja de la convolución.

---

# Tecnologías

- Python
- PyTorch
- TorchVision
- NumPy
- Matplotlib
- Scikit-learn

---

##### Ramiro Alvarez

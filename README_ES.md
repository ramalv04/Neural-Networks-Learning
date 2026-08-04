# Neural Networks Learning

Este repositorio es mi ruta de aprendizaje de redes neuronales y Deep Learning. Lo uso para ir reuniendo proyectos pequeños a medida que avanzo desde los fundamentos hacia arquitecturas más avanzadas.

Cada proyecto se enfoca en una idea, un dataset o un flujo de entrenamiento específico. El objetivo no es solo construir modelos, sino también entender qué pasa dentro de ellos y cómo se conectan los conceptos entre distintas implementaciones.

## Proyectos

### 01. XOR desde Cero

[README del proyecto](01-XOR-From-Scratch/README_ES.md)

Mi primer proyecto fue una red neuronal construida completamente desde cero con NumPy. Resuelve el problema XOR y muestra las ideas centrales detrás del forward propagation, el cálculo de la pérdida, la backpropagation, el descenso por gradiente y la actualización de parámetros sin usar un framework de Deep Learning.

### 02. MLP para MNIST con PyTorch

[README del proyecto](02-MNIST-MLP/README_ES.md)

El segundo proyecto da el salto a PyTorch y entrena un Perceptrón Multicapa sobre MNIST. Introduce tensores, DataLoader, mini-batches, ReLU, logits, CrossEntropy loss, SGD y diferenciación automática con Autograd.

### 03. CNN para MNIST

[README del proyecto](03-MNIST-CNN/README_ES.md)

El tercer proyecto construye una Red Neuronal Convolucional para MNIST. Extiende la idea del proyecto 2 con capas convolucionales y de pooling, agrega visualización de filtros y mapas de activación, y muestra por qué las CNN son una mejor opción que una red totalmente conectada para datos de imagen.

## Para qué sirve este repositorio

- Practicar los fundamentos de redes neuronales paso a paso.
- Comparar implementaciones manuales con entrenamiento usando frameworks.
- Construir intuición sobre ciclos de entrenamiento, optimización y evaluación.
- Mantener un registro claro de lo que voy aprendiendo en cada etapa.

## Tecnologías usadas hasta ahora

- Python
- NumPy
- PyTorch
- TorchVision
- Matplotlib
- Scikit-learn

## Estructura del repositorio

```text
01-XOR-From-Scratch/
02-MNIST-MLP/
03-MNIST-CNN/
```

---

Si querés ver este resumen en inglés, abrí [README.md](README.md).
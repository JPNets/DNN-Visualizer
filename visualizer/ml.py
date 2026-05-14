from __future__ import annotations

import math
import random
from typing import Callable, List, Tuple

import numpy as np

ActivationFn = Callable[[np.ndarray], np.ndarray]
ActivationDerivFn = Callable[[np.ndarray], np.ndarray]


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def sigmoid_derivative(x: np.ndarray) -> np.ndarray:
    s = sigmoid(x)
    return s * (1.0 - s)


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)


def relu_derivative(x: np.ndarray) -> np.ndarray:
    return np.where(x > 0.0, 1.0, 0.1)


def tanh(x: np.ndarray) -> np.ndarray:
    return np.tanh(x)


def tanh_derivative(x: np.ndarray) -> np.ndarray:
    return 1.0 - np.tanh(x) ** 2


def leaky_relu(x: np.ndarray) -> np.ndarray:
    return np.where(x > 0.0, x, x * 0.12)


def leaky_relu_derivative(x: np.ndarray) -> np.ndarray:
    return np.where(x > 0.0, 1.0, 0.12)


def gelu(x: np.ndarray) -> np.ndarray:
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3)))


def gelu_derivative(x: np.ndarray) -> np.ndarray:
    c = math.sqrt(2.0 / math.pi)
    t = np.tanh(c * (x + 0.044715 * x ** 3))
    return 0.5 * (1.0 + t) + 0.5 * x * (1.0 - t ** 2) * c * (1.0 + 0.134145 * x ** 2)


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e / np.sum(e, axis=1, keepdims=True)


ACTIVATIONS = {
    'linear': (lambda x: x, lambda x: np.ones_like(x)),
    'sigmoid': (sigmoid, sigmoid_derivative),
    'relu': (relu, relu_derivative),
    'tanh': (tanh, tanh_derivative),
    'leaky_relu': (leaky_relu, leaky_relu_derivative),
    'gelu': (gelu, gelu_derivative),
    'softmax': (softmax, lambda x: np.ones_like(x))
}


DATASET_OPTIONS = ['XOR', 'Moons', 'Circles', 'Spiral', 'Iris']


def normalize(points: np.ndarray) -> np.ndarray:
    min_vals = points.min(axis=0)
    max_vals = points.max(axis=0)
    return (points - min_vals) / (max_vals - min_vals + 1e-8) * 2.0 - 1.0


def pca_projection(points: np.ndarray, components: int = 2) -> np.ndarray:
    if points.shape[1] <= components:
        return points[:, :components]
    mean_centered = points - np.mean(points, axis=0, keepdims=True)
    cov = np.cov(mean_centered, rowvar=False)
    _, vectors = np.linalg.eigh(cov)
    top_vectors = vectors[:, -components:]
    projected = mean_centered @ top_vectors
    return projected


def create_dataset(name: str, samples: int = 200) -> Tuple[np.ndarray, np.ndarray]:
    if name == 'XOR':
        data = np.array([[x, y] for x in [0.1, 0.9] for y in [0.1, 0.9]], dtype=np.float32)
        labels = np.array([[0, 1] if int(round(x)) ^ int(round(y)) else [1, 0] for x, y in data], dtype=np.float32)
        jitter = np.random.randn(*data.shape) * 0.12
        data = normalize(data + jitter)
        return data, labels

    def random_circle(radius: float, noise: float = 0.08) -> np.ndarray:
        angle = random.random() * 2 * math.pi
        return np.array([math.cos(angle) * radius, math.sin(angle) * radius]) + np.random.randn(2) * noise

    if name == 'Moons':
        points = []
        labels = []
        for idx in range(samples):
            angle = math.pi * (idx / samples) * 2.0
            if idx % 2 == 0:
                points.append([math.cos(angle), math.sin(angle)])
                labels.append([1, 0])
            else:
                points.append([1.0 - math.cos(angle), -math.sin(angle) + 0.3])
                labels.append([0, 1])
        return normalize(np.array(points, dtype=np.float32)), np.array(labels, dtype=np.float32)

    if name == 'Circles':
        points = []
        labels = []
        for idx in range(samples):
            radius = 0.4 if idx < samples / 2 else 0.8
            points.append(random_circle(radius))
            labels.append([1, 0] if idx < samples / 2 else [0, 1])
        return normalize(np.array(points, dtype=np.float32)), np.array(labels, dtype=np.float32)

    if name == 'Spiral':
        points = []
        labels = []
        for idx in range(samples):
            label = 0 if idx < samples / 2 else 1
            radius = (idx / samples) * 1.3
            angle = radius * 4.0 * math.pi + (0 if label == 0 else math.pi)
            points.append([math.cos(angle) * radius, math.sin(angle) * radius])
            labels.append([1, 0] if label == 0 else [0, 1])
        return normalize(np.array(points, dtype=np.float32)), np.array(labels, dtype=np.float32)

    if name == 'Iris':
        raw = np.array([
            [5.1, 3.5, 1, 0, 0],
            [6.2, 3.4, 1, 0, 0],
            [5.8, 2.7, 1, 0, 0],
            [7.0, 3.2, 0, 1, 0],
            [6.4, 3.2, 0, 1, 0],
            [6.9, 3.1, 0, 1, 0],
            [6.3, 2.7, 0, 1, 0],
            [6.5, 3.0, 0, 1, 0],
            [6.7, 3.1, 0, 0, 1],
            [5.6, 2.5, 0, 0, 1],
            [6.1, 2.8, 0, 0, 1],
            [5.8, 2.7, 0, 0, 1]
        ], dtype=np.float32)
        data = normalize(raw[:, :2])
        labels = raw[:, 2:]
        return data, labels

    return np.zeros((0, 2), dtype=np.float32), np.zeros((0, 2), dtype=np.float32)


class NeuralNetwork:
    def __init__(self, layer_sizes: List[int], activations: List[str]) -> None:
        self.layer_sizes = layer_sizes
        self.activations = activations
        self.weights: List[np.ndarray] = []
        self.biases: List[np.ndarray] = []
        self.activations_cache: List[np.ndarray] = []
        self.pre_activations: List[np.ndarray] = []
        self.loss_history: list[float] = []
        self.accuracy_history: list[float] = []
        self.gradient_history: list[float] = []
        self.reset()
        self.loss = 0.0
        self.accuracy = 0.0
        self.gradient_norm = 0.0

    def reset(self) -> None:
        self.weights = []
        self.biases = []
        self.activations_cache = []
        self.pre_activations = []
        self.loss_history = []
        self.accuracy_history = []
        self.gradient_history = []
        for i in range(len(self.layer_sizes) - 1):
            fan_in = self.layer_sizes[i]
            fan_out = self.layer_sizes[i + 1]
            limit = math.sqrt(6.0 / max(1, fan_in + fan_out))
            self.weights.append(np.random.uniform(-limit, limit, (fan_in, fan_out)).astype(np.float32))
            self.biases.append(np.zeros((1, fan_out), dtype=np.float32))

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        activation = inputs.copy()
        self.activations_cache = [activation]
        self.pre_activations = []
        for i, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            z = activation @ weight + bias
            self.pre_activations.append(z)
            if self.activations[i] == 'softmax' and i == len(self.weights) - 1:
                activation = softmax(z)
            else:
                activation = ACTIVATIONS[self.activations[i]][0](z)
            self.activations_cache.append(activation)
        return activation

    def predict_proba(self, inputs: np.ndarray) -> np.ndarray:
        activation = inputs.copy()
        for i, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            z = activation @ weight + bias
            if self.activations[i] == 'softmax' and i == len(self.weights) - 1:
                activation = softmax(z)
            else:
                activation = ACTIVATIONS[self.activations[i]][0](z)
        return activation

    def hidden_representation(self, inputs: np.ndarray) -> np.ndarray:
        activation = inputs.copy()
        hidden_states = []
        for i, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            z = activation @ weight + bias
            if self.activations[i] == 'softmax' and i == len(self.weights) - 1:
                activation = softmax(z)
            else:
                activation = ACTIVATIONS[self.activations[i]][0](z)
            hidden_states.append(activation)
        if len(hidden_states) > 1:
            return hidden_states[-2]
        return hidden_states[0]

    def compute_loss(self, predictions: np.ndarray, labels: np.ndarray) -> float:
        clipped = np.clip(predictions, 1e-8, 1.0 - 1e-8)
        return float(-np.mean(np.sum(labels * np.log(clipped), axis=1)))

    def train_batch(self, inputs: np.ndarray, labels: np.ndarray, learning_rate: float) -> None:
        outputs = self.forward(inputs)
        loss = self.compute_loss(outputs, labels)
        self.loss = loss
        predicted = np.argmax(outputs, axis=1)
        truth = np.argmax(labels, axis=1)
        self.accuracy = float(np.mean(predicted == truth) * 100.0)

        gradients = outputs - labels
        grad_norm_total = 0.0
        for i in reversed(range(len(self.weights))):
            activation = self.activations_cache[i]
            z = self.pre_activations[i]
            if self.activations[i] != 'softmax' or i != len(self.weights) - 1:
                grad_act = gradients * ACTIVATIONS[self.activations[i]][1](z)
            else:
                grad_act = gradients
            grad_w = activation.T @ grad_act / inputs.shape[0]
            grad_b = np.mean(grad_act, axis=0, keepdims=True)
            gradients = grad_act @ self.weights[i].T
            self.weights[i] -= learning_rate * grad_w
            self.biases[i] -= learning_rate * grad_b
            grad_norm_total += float(np.linalg.norm(grad_w) + np.linalg.norm(grad_b))
        self.gradient_norm = float(grad_norm_total)
        self.loss_history.append(self.loss)
        self.accuracy_history.append(self.accuracy)
        self.gradient_history.append(self.gradient_norm)

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        outputs = self.forward(inputs)
        return np.argmax(outputs, axis=1)

    def decision_grid(self, resolution: int = 50) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        xs = np.linspace(-1.2, 1.2, resolution)
        ys = np.linspace(-1.2, 1.2, resolution)
        grid = np.stack(np.meshgrid(xs, ys), axis=-1).reshape(-1, 2)
        outputs = self.forward(grid)
        labels = np.argmax(outputs, axis=1).astype(np.int32)
        return grid[:, 0], grid[:, 1], labels

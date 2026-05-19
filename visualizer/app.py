from __future__ import annotations

import math
from typing import Optional

import numpy as np
from PySide6.QtCore import QMargins, QTimer, Qt, QPropertyAnimation, QEasingCurve, QPoint
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QScrollArea,
    QSlider,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
    QToolTip,
    QDialog,
    QCheckBox,
    QTextEdit,
)

from visualizer.ml import (
    DATASET_OPTIONS,
    NeuralNetwork,
    ACTIVATIONS,
    create_dataset,
    pca_projection,
    get_dataset_description,
    get_activation_description,
)


class NetworkCanvas(QWidget):
    def __init__(self, network: NeuralNetwork, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.network = network
        self.phase = 0.0
        self.rotation_x = -18.0
        self.rotation_y = 35.0
        self.zoom = 3.0
        self.dragging = False
        self.last_mouse_pos = None
        self.setMinimumHeight(420)
        self.setAutoFillBackground(True)
        self.setStyleSheet('background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #050812, stop:1 #091424); border-radius: 20px;')
        self.setCursor(Qt.OpenHandCursor)

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.RightButton:
            self.dragging = True
            self.last_mouse_pos = event.position()
            self.setCursor(Qt.ClosedHandCursor)

    def mouseMoveEvent(self, event) -> None:
        if self.dragging and self.last_mouse_pos is not None:
            delta = event.position() - self.last_mouse_pos
            self.rotation_y += float(delta.x()) * 0.35
            self.rotation_x = float(max(-80.0, min(80.0, self.rotation_x + delta.y() * 0.35)))
            self.last_mouse_pos = event.position()
            self.update()

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.RightButton:
            self.dragging = False
            self.last_mouse_pos = None
            self.setCursor(Qt.OpenHandCursor)

    def wheelEvent(self, event) -> None:
        delta = event.angleDelta().y() / 120.0
        self.zoom = max(0.4, min(3.0, self.zoom + delta * 0.15))
        self.update()

    def _rotate_point(self, x: float, y: float, z: float) -> tuple[float, float, float]:
        rx = math.radians(self.rotation_x)
        ry = math.radians(self.rotation_y)
        cosx, sinx = math.cos(rx), math.sin(rx)
        cosy, siny = math.cos(ry), math.sin(ry)
        y2 = y * cosx - z * sinx
        z2 = y * sinx + z * cosx
        x2 = x * cosy + z2 * siny
        z3 = -x * siny + z2 * cosy
        return x2, y2, z3

    def _project(self, x: float, y: float, z: float) -> tuple[float, float, float]:
        x2, y2, z2 = self._rotate_point(x, y, z)
        distance = 10.0
        z2 += distance
        scale = self.zoom * min(self.width(), self.height()) * 0.11 / max(1.0, z2)
        return (
            self.width() * 0.5 + x2 * scale,
            self.height() * 0.5 - y2 * scale,
            z2,
        )

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        try:
            painter.setRenderHint(QPainter.Antialiasing)
            painter.fillRect(self.rect(), QColor(8, 10, 18))
            layer_count = len(self.network.layer_sizes)
            nodes_by_layer: list[list[tuple[float, float, float]]] = []

            for layer_index, size in enumerate(self.network.layer_sizes):
                layer_x = (layer_index - (layer_count - 1) / 2.0) * 4.0
                if size <= 1:
                    rows = cols = 1
                else:
                    rows = math.ceil(math.sqrt(size))
                    cols = math.ceil(size / rows)
                layer_nodes: list[tuple[float, float, float]] = []
                for neuron_index in range(size):
                    row = neuron_index // cols
                    col = neuron_index % cols
                    y = (row - (rows - 1) / 2.0) * 1.4
                    z = (col - (cols - 1) / 2.0) * 1.4
                    layer_nodes.append((layer_x, y, z))
                nodes_by_layer.append(layer_nodes)

            connections: list[tuple[float, tuple[float, float], tuple[float, float], float]] = []
            for layer_index, weights in enumerate(self.network.weights):
                for src in range(weights.shape[0]):
                    for tgt in range(weights.shape[1]):
                        src_pos = nodes_by_layer[layer_index][src]
                        tgt_pos = nodes_by_layer[layer_index + 1][tgt]
                        p1x, p1y, z1 = self._project(*src_pos)
                        p2x, p2y, z2 = self._project(*tgt_pos)
                        connections.append(((z1 + z2) * 0.5, (p1x, p1y), (p2x, p2y), float(weights[src, tgt])))

            connections.sort(key=lambda item: item[0], reverse=True)
            for _, start, end, strength in connections:
                alpha = int(min(200, max(40, abs(strength) * 220)))
                color = QColor(81, 185, 255, alpha) if strength >= 0 else QColor(255, 130, 180, alpha)
                pen = QPen(color, max(1.0, 1.2 + abs(strength) * 1.8))
                painter.setPen(pen)
                painter.drawLine(int(start[0]), int(start[1]), int(end[0]), int(end[1]))

            node_positions: list[tuple[float, float, float, float]] = []
            for layer_index, layer_nodes in enumerate(nodes_by_layer):
                for neuron_index, node_pos in enumerate(layer_nodes):
                    activation_value = 0.0
                    if hasattr(self.network, 'activations_cache') and self.network.activations_cache and layer_index < len(self.network.activations_cache):
                        a = self.network.activations_cache[layer_index]
                        if neuron_index < a.shape[1]:
                            activation_value = float(a[0, neuron_index])
                    px, py, depth = self._project(*node_pos)
                    node_positions.append((depth, px, py, activation_value))

            node_positions.sort(key=lambda item: item[0], reverse=True)
            for depth, px, py, activation_value in node_positions:
                radius = 10 + abs(activation_value) * 14
                glow = QColor(68, 180, 255, 60)
                painter.setBrush(glow)
                painter.setPen(Qt.NoPen)
                painter.drawEllipse(int(px - radius * 0.8), int(py - radius * 0.8), int(radius * 1.6), int(radius * 1.6))
                painter.setBrush(QColor(94, 205, 255, 220))
                painter.setPen(QPen(QColor(232, 244, 255, 220), 1.4))
                painter.drawEllipse(int(px - radius / 2), int(py - radius / 2), int(radius), int(radius))

            painter.setPen(QPen(QColor(215, 230, 255), 1))
            painter.setFont(QFont('Segoe UI', 12, QFont.Weight.DemiBold))
            painter.drawText(20, 30, 'Neural architecture display')
            painter.setFont(QFont('Segoe UI', 10, QFont.Weight.Normal))
            painter.setPen(QColor(180, 196, 215, 180))
            painter.drawText(22, 50, 'Right-click + drag to rotate | Mouse wheel to zoom')
            self.phase += 0.05
        finally:
            painter.end()


class DecisionCanvas(QWidget):
    def __init__(self, network: NeuralNetwork, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.network = network
        self.dataset_x = np.zeros((0, 2), dtype=np.float32)
        self.dataset_y = np.zeros((0, 2), dtype=np.float32)
        self.setMinimumHeight(320)
        self.setAutoFillBackground(True)
        self.setStyleSheet('background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #09111e, stop:1 #06101b); border-radius: 20px;')

    def set_data(self, x: np.ndarray, y: np.ndarray) -> None:
        self.dataset_x = x
        self.dataset_y = y
        self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        try:
            painter.setRenderHint(QPainter.Antialiasing)
            rect = self.rect().marginsRemoved(QMargins(16, 16, 16, 16))
            painter.setBrush(QColor(12, 19, 35, 230))
            painter.setPen(Qt.NoPen)
            painter.drawRoundedRect(rect, 18, 18)

            if self.dataset_x.size:
                resolution = 38
                xs = np.linspace(-1.2, 1.2, resolution)
                ys = np.linspace(-1.2, 1.2, resolution)
                grid = np.stack(np.meshgrid(xs, ys), axis=-1).reshape(-1, 2)
                decisions = self.network.predict_proba(grid)
                decision_labels = np.argmax(decisions, axis=1)
                palette = [QColor(36, 83, 129, 45), QColor(74, 47, 102, 45), QColor(125, 39, 72, 45)]

                for idx, (gx, gy) in enumerate(grid):
                    label = decision_labels[idx]
                    painter.setBrush(palette[label % len(palette)])
                    painter.setPen(Qt.NoPen)
                    px = rect.x() + ((gx + 1.2) / 2.4) * rect.width()
                    py = rect.y() + ((1.2 - gy) / 2.4) * rect.height()
                    painter.drawRect(int(px), int(py), int(rect.width() / resolution) + 1, int(rect.height() / resolution) + 1)

                hidden = self.network.hidden_representation(self.dataset_x)
                latent = pca_projection(hidden)
                if latent.shape[1] == 2:
                    norm_x = (latent[:, 0] - latent[:, 0].min()) / (latent[:, 0].ptp() + 1e-8)
                    norm_y = (latent[:, 1] - latent[:, 1].min()) / (latent[:, 1].ptp() + 1e-8)
                    for idx in range(latent.shape[0]):
                        label = int(np.argmax(self.dataset_y[idx]))
                        palette = [QColor(84, 212, 255, 170), QColor(214, 146, 255, 170), QColor(255, 148, 179, 170)]
                        painter.setBrush(palette[label % len(palette)])
                        painter.setPen(QPen(QColor(255, 255, 255, 140), 1))
                        px = rect.x() + norm_x[idx] * rect.width()
                        py = rect.y() + (1 - norm_y[idx]) * rect.height()
                        painter.drawEllipse(int(px - 4), int(py - 4), 8, 8)

                for idx, point in enumerate(self.dataset_x):
                    label = int(np.argmax(self.dataset_y[idx]))
                    palette = [QColor(103, 203, 255), QColor(186, 154, 255), QColor(255, 130, 170)]
                    painter.setPen(QPen(QColor(255, 255, 255, 200), 1))
                    painter.setBrush(palette[label % len(palette)])
                    px = rect.x() + ((point[0] + 1.2) / 2.4) * rect.width()
                    py = rect.y() + ((1.2 - point[1]) / 2.4) * rect.height()
                    painter.drawEllipse(int(px - 5), int(py - 5), 10, 10)

            painter.setPen(QPen(QColor(202, 224, 255), 1))
            painter.setFont(QFont('Segoe UI', 12, QFont.Weight.DemiBold))
            painter.drawText(rect.x() + 16, rect.y() + 28, 'Decision boundary + latent projection')
        finally:
            painter.end()


class MathDialog(QDialog):
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle('Explain the Math')
        self.setMinimumSize(540, 420)
        layout = QVBoxLayout(self)
        txt = QTextEdit()
        txt.setReadOnly(True)
        txt.setStyleSheet('background: #091424; color: #e7f1ff; border-radius: 8px; padding:10px;')
        txt.setHtml(
            '<h2>How the network computes a prediction</h2>'
            '<p>Each neuron computes a weighted sum plus a bias, then an activation function:</p>'
            '<p><b>z = w \u00b7 x + b</b></p>'
            '<p><b>a = activation(z)</b></p>'
            '<p>For the output layer with softmax, we turn raw scores into probabilities:</p>'
            '<p><b>softmax(s)_i = exp(s_i) / sum_j exp(s_j)</b></p>'
            '<h3>Plain language</h3>'
            '<ul>'
            '<li><b>Weights</b> decide how important each input is.</li>'
            '<li><b>Bias</b> shifts the activation threshold.</li>'
            '<li><b>Activation</b> controls whether the neuron "fires".</li>'
            '</ul>'
        )
        layout.addWidget(txt)
        close = QPushButton('Close')
        close.clicked.connect(self.accept)
        layout.addWidget(close)


class CompareDialog(QDialog):
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle('Compare Models')
        self.setMinimumSize(960, 680)
        self.setStyleSheet('background: #071224; color: #eaf3ff;')

        # state
        self.left_net = NeuralNetwork([2, 6, 2], ['linear', 'tanh', 'softmax'])
        self.right_net = NeuralNetwork([2, 10, 2], ['linear', 'tanh', 'softmax'])
        self.left_x, self.left_y = create_dataset('XOR')
        self.right_x, self.right_y = create_dataset('CIRCLE')

        layout = QVBoxLayout(self)

        top = QHBoxLayout()

        # Left pane
        left_panel = QVBoxLayout()
        self.left_decision = DecisionCanvas(self.left_net)
        self.left_history = HistoryCanvas()
        self.left_decision.set_data(self.left_x, self.left_y)
        left_panel.addWidget(self.left_decision, 3)
        left_panel.addWidget(self.left_history, 1)

        # Right pane
        right_panel = QVBoxLayout()
        self.right_decision = DecisionCanvas(self.right_net)
        self.right_history = HistoryCanvas()
        self.right_decision.set_data(self.right_x, self.right_y)
        right_panel.addWidget(self.right_decision, 3)
        right_panel.addWidget(self.right_history, 1)

        # Wrap panes in subtle frames for polish
        left_frame = QWidget()
        left_frame.setLayout(left_panel)
        left_frame.setStyleSheet('background: rgba(8,14,28,0.35); border-radius: 12px; padding: 10px;')

        right_frame = QWidget()
        right_frame.setLayout(right_panel)
        right_frame.setStyleSheet('background: rgba(8,14,28,0.35); border-radius: 12px; padding: 10px;')

        top.addWidget(left_frame)
        top.addWidget(right_frame)

        layout.addLayout(top)

        # Control bar
        ctrl = QWidget()
        ctrl_layout = QHBoxLayout(ctrl)
        ctrl_layout.setContentsMargins(6, 6, 6, 6)
        ctrl_layout.setSpacing(12)
        self.start_btn = QPushButton('Start')
        self.start_btn.clicked.connect(self._toggle_training)
        self.start_btn.setProperty('primary', True)
        self.reset_btn = QPushButton('Reset')
        self.reset_btn.clicked.connect(self._reset_models)
        ctrl_layout.addWidget(self.start_btn)
        ctrl_layout.addWidget(self.reset_btn)

        ctrl_layout.addStretch()
        self.sync_checkbox = QCheckBox('Train in sync')
        self.sync_checkbox.setChecked(True)
        ctrl_layout.addWidget(self.sync_checkbox)

        layout.addWidget(ctrl)

        footer = QLabel('Tip: adjust network sizes on the main UI and use this dialog to visually compare decision boundaries and training dynamics.')
        footer.setStyleSheet('color: #cfe6ff;')
        layout.addWidget(footer)

        # timer for training both models
        self._timer = QTimer(self)
        self._timer.setInterval(120)
        self._timer.timeout.connect(self._train_step)

        # initial history
        self._reset_models()

    def _toggle_training(self) -> None:
        if self._timer.isActive():
            self._timer.stop()
            self.start_btn.setText('Start')
        else:
            self._timer.start()
            self.start_btn.setText('Pause')

    def _reset_models(self) -> None:
        # reinitialize networks and histories
        self.left_net.reset()
        self.right_net.reset()
        self.left_history.set_history([], [], [])
        self.right_history.set_history([], [], [])
        self.left_decision.update()
        self.right_decision.update()

    def _train_step(self) -> None:
        # train one batch on each model; if sync, use same lr
        lr = 0.06
        try:
            self.left_net.train_batch(self.left_x, self.left_y, lr)
            if self.sync_checkbox.isChecked():
                self.right_net.train_batch(self.right_x, self.right_y, lr)
            else:
                self.right_net.train_batch(self.right_x, self.right_y, lr * 0.9)
        except Exception:
            pass
        self.left_decision.update()
        self.right_decision.update()
        self.left_history.set_history(self.left_net.loss_history, self.left_net.accuracy_history, self.left_net.gradient_history)
        self.right_history.set_history(self.right_net.loss_history, self.right_net.accuracy_history, self.right_net.gradient_history)


class SmoothScrollArea(QScrollArea):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._anim: QPropertyAnimation | None = None

    def wheelEvent(self, event) -> None:
        # Smooth-scroll by animating the vertical scrollbar
        delta = event.angleDelta().y()
        if delta == 0:
            return
        step = -int(delta / 2)
        bar = self.verticalScrollBar()
        start = bar.value()
        end = max(bar.minimum(), min(bar.maximum(), start + step))
        if self._anim is not None and self._anim.state() == QPropertyAnimation.Running:
            self._anim.stop()
        self._anim = QPropertyAnimation(bar, b'value', self)
        self._anim.setStartValue(start)
        self._anim.setEndValue(end)
        self._anim.setDuration(240)
        self._anim.setEasingCurve(QEasingCurve.InOutCubic)
        self._anim.start()


class HistoryCanvas(QWidget):
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.loss_history: list[float] = []
        self.acc_history: list[float] = []
        self.grad_history: list[float] = []
        self.setMinimumHeight(260)
        self.setAutoFillBackground(True)
        self.setStyleSheet('background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #08111f, stop:1 #07131e); border-radius: 20px;')

    def set_history(self, loss: list[float], accuracy: list[float], grad: list[float]) -> None:
        self.loss_history = loss
        self.acc_history = accuracy
        self.grad_history = grad
        self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        try:
            painter.setRenderHint(QPainter.Antialiasing)
            rect = self.rect().marginsRemoved(QMargins(16, 16, 16, 16))
            painter.setBrush(QColor(10, 16, 28, 220))
            painter.setPen(Qt.NoPen)
            painter.drawRoundedRect(rect, 18, 18)

            max_points = max(len(self.loss_history), len(self.acc_history), len(self.grad_history), 2)
            x_positions = [rect.x() + i * rect.width() / max(1, max_points - 1) for i in range(max_points)]

            def draw_series(data, color, width):
                if not data:
                    return
                normalized = np.array(data, dtype=np.float32)
                normalized = (normalized - normalized.min()) / (normalized.ptp() + 1e-8)
                points = [
                    (x_positions[i], rect.bottom() - normalized[i] * rect.height())
                    for i in range(len(normalized))
                ]
                painter.setPen(QPen(color, width))
                if len(points) == 1:
                    painter.setBrush(color)
                    painter.drawEllipse(int(points[0][0] - 2), int(points[0][1] - 2), 4, 4)
                else:
                    for i in range(len(points) - 1):
                        painter.drawLine(int(points[i][0]), int(points[i][1]), int(points[i + 1][0]), int(points[i + 1][1]))

            draw_series(self.loss_history, QColor(94, 149, 255, 220), 2)
            draw_series(self.acc_history, QColor(147, 232, 255, 200), 2)
            draw_series(self.grad_history, QColor(246, 128, 182, 200), 2)

            painter.setPen(QPen(QColor(212, 228, 255, 120), 1))
            for grid_line in range(1, 4):
                y = rect.y() + grid_line * rect.height() / 4
                painter.drawLine(rect.x(), int(y), rect.right(), int(y))

            painter.setPen(QPen(QColor(212, 228, 255), 1))
            painter.setFont(QFont('Segoe UI', 12, QFont.Weight.DemiBold))
            painter.drawText(rect.x() + 16, rect.y() + 28, 'Training history curves')

            # Draw legend
            legend_x = rect.right() - 170
            legend_y = rect.y() + 8
            painter.setFont(QFont('Segoe UI', 9))
            painter.setPen(QPen(QColor(200, 220, 255), 200))
            painter.drawText(legend_x + 20, legend_y + 12, 'Loss')
            painter.setPen(QPen(QColor(147, 232, 255), 1))
            painter.setBrush(QColor(94, 149, 255, 200))
            painter.drawRect(legend_x, legend_y + 4, 12, 8)
            painter.setPen(QPen(QColor(200, 220, 255), 200))
            painter.drawText(legend_x + 20, legend_y + 30, 'Accuracy')
            painter.setPen(QPen(QColor(147, 232, 255), 1))
            painter.setBrush(QColor(147, 232, 255, 200))
            painter.drawRect(legend_x, legend_y + 22, 12, 8)
            painter.drawText(legend_x + 20, legend_y + 48, 'Grad norm')
            painter.setPen(QPen(QColor(246, 128, 182), 1))
            painter.setBrush(QColor(246, 128, 182, 200))
            painter.drawRect(legend_x, legend_y + 40, 12, 8)
            # Y-axis labels
            painter.setPen(QPen(QColor(190, 210, 235), 160))
            painter.setFont(QFont('Segoe UI', 9))
            painter.drawText(rect.x() + 8, rect.y() + 14, '1.0')
            painter.drawText(rect.x() + 8, rect.bottom() - 6, '0.0')
        finally:
            painter.end()

    def mouseMoveEvent(self, event) -> None:
        # Show tooltip with nearest values
        if not (self.loss_history or self.acc_history or self.grad_history):
            return
        rect = self.rect().marginsRemoved(QMargins(16, 16, 16, 16))
        max_points = max(len(self.loss_history), len(self.acc_history), len(self.grad_history), 1)
        x_positions = [rect.x() + i * rect.width() / max(1, max_points - 1) for i in range(max_points)]
        # find closest x
        mx = event.position().x()
        idx = min(range(len(x_positions)), key=lambda i: abs(x_positions[i] - mx))
        parts = []
        if idx < len(self.loss_history):
            parts.append(f'Loss: {self.loss_history[idx]:.4f}')
        if idx < len(self.acc_history):
            parts.append(f'Acc: {self.acc_history[idx]:.1f}%')
        if idx < len(self.grad_history):
            parts.append(f'Grad: {self.grad_history[idx]:.3f}')
        if parts:
            QToolTip.showText(self.mapToGlobal(QPoint(int(mx), int(event.position().y()))), '\n'.join(parts), self)


class ScatterCanvas(QWidget):
    def __init__(self, network: NeuralNetwork, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.network = network
        self.dataset_x = np.zeros((0, 2), dtype=np.float32)
        self.dataset_y = np.zeros((0, 2), dtype=np.float32)
        self.setMinimumHeight(360)
        self.setAutoFillBackground(True)
        self.setStyleSheet('background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #07111f, stop:1 #091424); border-radius: 20px;')

    def set_data(self, x: np.ndarray, y: np.ndarray) -> None:
        self.dataset_x = x
        self.dataset_y = y
        self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        rect = self.rect().marginsRemoved(QMargins(16, 16, 16, 16))
        painter.setBrush(QColor(13, 23, 45, 220))
        painter.setPen(Qt.NoPen)
        painter.drawRoundedRect(rect, 18, 18)

        if self.dataset_x.size:
            px = rect.x()
            py = rect.y()
            w = rect.width()
            h = rect.height()
            x_norm = (self.dataset_x[:, 0] + 1.2) / 2.4
            y_norm = (self.dataset_x[:, 1] + 1.2) / 2.4
            for idx, point in enumerate(self.dataset_x):
                label = int(np.argmax(self.dataset_y[idx]))
                palette = [QColor(92, 196, 255), QColor(165, 133, 250), QColor(244, 114, 182)]
                color = palette[label % len(palette)]
                color.setAlpha(220)
                painter.setPen(Qt.NoPen)
                painter.setBrush(color)
                x = px + x_norm[idx] * w
                y = py + (1 - y_norm[idx]) * h
                painter.drawEllipse(int(x - 5), int(y - 5), 10, 10)

        painter.setPen(QPen(QColor(145, 178, 255, 120), 1))
        for line in range(1, 5):
            y = rect.y() + line * rect.height() / 6
            painter.drawLine(rect.x(), int(y), rect.right(), int(y))
            x = rect.x() + line * rect.width() / 6
            painter.drawLine(int(x), rect.y(), int(x), rect.bottom())

        painter.setPen(QPen(QColor(212, 228, 255), 1))
        painter.setFont(QFont('Segoe UI', 11, QFont.Weight.DemiBold))
        painter.drawText(rect.x() + 14, rect.y() + 28, 'Dataset & decision field')


class NeuralVisualizerWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle('Deep Neural Network Visualizer')
        self.setMinimumSize(1220, 780)
        self.setStyleSheet('background: #0d1626; color: #eef3ff;')
        self.network = NeuralNetwork([2, 6, 2], ['linear', 'tanh', 'softmax'])
        self.active_dataset = 'XOR'
        self.dataset_x, self.dataset_y = create_dataset(self.active_dataset)
        self.running = False
        self.learning_rate = 0.08
        self.lesson_active = False
        self.lesson_step_index = 0
        self.lesson_steps = [
            {
                'title': 'Input layer',
                'text': 'The network first looks at the dataset points. Each point is a pair of numbers that describe one example before it enters the model.',
            },
            {
                'title': 'Hidden neurons',
                'text': 'Hidden neurons combine the inputs using weights and a bias. This is where the network learns to recognize patterns.',
            },
            {
                'title': 'Activation function',
                'text': 'An activation function decides whether a neuron should pass its signal forward. It helps the network learn non-linear shapes.',
            },
            {
                'title': 'Output prediction',
                'text': 'The output layer turns the neuron signals into a final guess for the class. Softmax makes the result act like a probability.',
            },
            {
                'title': 'Training and learning',
                'text': 'Training means the network checks how wrong it is and adjusts weights to make better guesses next time.',
            },
            {
                'title': 'Decision boundary',
                'text': 'The decision boundary shows how the network separates different classes in the input space. As training improves, the boundary becomes more accurate.',
            },
        ]

        self.dataset_selector = QComboBox()
        self.dataset_selector.addItems(DATASET_OPTIONS)
        self.dataset_selector.setCurrentText(self.active_dataset)
        self.dataset_selector.currentTextChanged.connect(self.on_dataset_changed)
        self.dataset_selector.setToolTip('Pick a dataset that the network will learn from. Each shape shows a different challenge.')

        self.lr_slider = QSlider(Qt.Horizontal)
        self.lr_slider.setRange(1, 40)
        self.lr_slider.setValue(int(self.learning_rate * 1000))
        self.lr_slider.valueChanged.connect(self.on_lr_changed)
        self.lr_slider.setToolTip('Learning rate controls how big each change is while the network learns. Too large can jump around, too small can learn slowly.')

        self.hidden_neurons_spinner = QSpinBox()
        self.hidden_neurons_spinner.setRange(2, 64)
        self.hidden_neurons_spinner.setValue(12)
        self.hidden_neurons_spinner.valueChanged.connect(self.on_architecture_changed)
        self.hidden_neurons_spinner.setToolTip('More neurons allow the network to learn more complex patterns, but simpler shapes can learn faster.')

        self.hidden_neurons_spinner_2 = QSpinBox()
        self.hidden_neurons_spinner_2.setRange(2, 64)
        self.hidden_neurons_spinner_2.setValue(8)
        self.hidden_neurons_spinner_2.valueChanged.connect(self.on_architecture_changed)
        self.hidden_neurons_spinner_2.setToolTip('This second hidden layer adds another level of pattern recognition for more complex decisions.')

        self.activation_selector = QComboBox()
        self.activation_selector.addItems([key for key in ACTIVATIONS.keys() if key != 'linear'])
        self.activation_selector.setCurrentText('tanh')
        self.activation_selector.currentTextChanged.connect(self.on_architecture_changed)
        self.activation_selector.setToolTip('Choose how each neuron decides to pass a value forward. This affects learning behavior.')

        self.train_button = QPushButton('Start training')
        self.train_button.clicked.connect(self.on_toggle_training)
        self.train_button.setToolTip('Start or pause training. When training, the network learns from the current dataset.')
        self.reset_button = QPushButton('Reset network')
        self.reset_button.clicked.connect(self.on_reset)
        self.reset_button.setToolTip('Reset the network weights and training history to start fresh.')

        self.math_button = QPushButton('Explain the Math')
        self.math_button.clicked.connect(self.show_math_dialog)
        self.math_button.setToolTip('Open a concise explanation of the math behind the network.')

        self.compare_button = QPushButton('Compare models')
        self.compare_button.clicked.connect(self.open_compare_dialog)
        self.compare_button.setToolTip('Open a side-by-side comparison of two networks.')

        self.lesson_button = QPushButton('Start lesson')
        self.lesson_button.clicked.connect(self.toggle_lesson_mode)
        self.lesson_button.setToolTip('Start a guided lesson that explains the neural network step by step.')
        self.prev_step_button = QPushButton('Previous step')
        self.prev_step_button.clicked.connect(self.prev_lesson_step)
        self.prev_step_button.setEnabled(False)
        self.prev_step_button.setToolTip('Go back to the previous lesson step.')
        self.next_step_button = QPushButton('Next step')
        self.next_step_button.clicked.connect(self.next_lesson_step)
        self.next_step_button.setEnabled(False)
        self.next_step_button.setToolTip('Advance to the next lesson step.')
        self.lesson_status_label = QLabel('Lesson mode is off.')
        self.lesson_status_label.setWordWrap(True)
        self.lesson_status_label.setStyleSheet('color: #b5c5ff;')

        self.tip_title = QLabel('Quick explanation')
        self.tip_title.setFont(QFont('Segoe UI', 11, QFont.Weight.DemiBold))
        self.tip_title.setStyleSheet('color: #dbe4ff;')
        self.tip_label = QLabel('This app helps you understand how a neural network learns from examples. Use the guided lesson to walk through the core steps.')
        self.tip_label.setWordWrap(True)
        self.tip_label.setStyleSheet('background: rgba(255, 255, 255, 0.04); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 16px; padding: 14px; color: #dfe7ff;')

        self.loss_label = QLabel('Loss: 0.000')
        self.accuracy_label = QLabel('Accuracy: 0.0%')
        self.grad_label = QLabel('Grad norm: 0.000')

        self.explanation_title = QLabel('How this network thinks')
        self.explanation_title.setFont(QFont('Segoe UI', 12, QFont.Weight.DemiBold))
        self.explanation_title.setStyleSheet('color: #dce6ff;')
        self.explanation_label = QLabel()
        self.explanation_label.setWordWrap(True)
        self.explanation_label.setStyleSheet('background: rgba(22, 34, 58, 0.92); color: #dce6ff; padding: 14px; border-radius: 16px;')
        self.explanation_label.setMinimumHeight(160)
        self.explanation_label.setToolTip('A simple explanation of the current dataset, settings, and training behavior.')

        self.canvas = NetworkCanvas(self.network)
        self.canvas.setToolTip('The neural architecture view shows neurons and weighted connections. Brighter glows mean stronger activations.')
        self.scatter = ScatterCanvas(self.network)
        self.scatter.setToolTip('The dataset view shows the input points and their class labels.')
        self.decision_canvas = DecisionCanvas(self.network)
        self.decision_canvas.setToolTip('The decision boundary view shows how the network separates the different classes in the input space.')
        self.history_canvas = HistoryCanvas()
        self.history_canvas.setToolTip('The history chart tracks loss, accuracy, and gradient strength as training progresses.')

        self.build_ui()

        self.timer = QTimer(self)
        self.timer.setInterval(120)
        self.timer.timeout.connect(self.training_step)

        self.update_dataset()
        self.update_metrics()

    def build_ui(self) -> None:
        sidebar = QWidget()
        sidebar.setObjectName('sidebar')
        sidebar.setStyleSheet(
            '#sidebar { background: rgba(14, 20, 39, 0.95); border-radius: 24px; }'
            'QLabel { color: #e7ecff; }'
            'QComboBox, QSpinBox, QSlider, QPushButton { background: rgba(24, 32, 55, 0.95); color: #eef3ff; border: 1px solid rgba(143, 178, 255, 0.18); border-radius: 14px; padding: 10px; }'
            'QComboBox::drop-down { border: none; }'
            'QPushButton { min-height: 42px; }'
            'QPushButton:hover { background: rgba(63, 108, 255, 0.95); }'
            'QPushButton:pressed { background: rgba(44, 82, 212, 0.98); }'
            'QPushButton[primary="true"] { background: qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 #4b9bff, stop:1 #2f6fe6); color: #ffffff; border: none; font-weight: 600; }'
            'QPushButton[primary="true"]:hover { background: qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 #5eaaff, stop:1 #3c7ff0); }'
            'QCheckBox { color: #cfe6ff; }'
            'QSlider::groove:horizontal { height: 10px; background: rgba(108, 137, 205, 0.25); border-radius: 5px; }'
            'QSlider::handle:horizontal { width: 16px; background: #5da1ff; border-radius: 8px; margin: -3px 0; }'
        )
        side_layout = QVBoxLayout(sidebar)
        side_layout.setContentsMargins(26, 26, 26, 26)
        side_layout.setSpacing(20)

        header = QLabel('Control panel')
        header.setFont(QFont('Segoe UI', 16, QFont.Weight.Bold))
        side_layout.addWidget(header)
        side_layout.addWidget(QLabel('Dataset'))
        side_layout.addWidget(self.dataset_selector)
        side_layout.addWidget(QLabel('Hidden layer 1 size'))
        side_layout.addWidget(self.hidden_neurons_spinner)
        side_layout.addWidget(QLabel('Hidden layer 2 size'))
        side_layout.addWidget(self.hidden_neurons_spinner_2)
        side_layout.addWidget(QLabel('Activation function'))
        side_layout.addWidget(self.activation_selector)
        side_layout.addWidget(QLabel('Learning rate'))
        side_layout.addWidget(self.lr_slider)
        side_layout.addWidget(self.train_button)
        side_layout.addWidget(self.reset_button)
        side_layout.addWidget(self.math_button)
        side_layout.addWidget(self.compare_button)
        side_layout.addWidget(QLabel('Guided lesson'))
        side_layout.addWidget(self.lesson_button)
        lesson_control_bar = QWidget()
        lesson_control_layout = QHBoxLayout(lesson_control_bar)
        lesson_control_layout.setContentsMargins(0, 0, 0, 0)
        lesson_control_layout.setSpacing(10)
        lesson_control_layout.addWidget(self.prev_step_button)
        lesson_control_layout.addWidget(self.next_step_button)
        side_layout.addWidget(lesson_control_bar)
        side_layout.addWidget(self.lesson_status_label)
        side_layout.addSpacing(18)
        self.loss_label.setStyleSheet('color: #b5c5ff;')
        self.accuracy_label.setStyleSheet('color: #b5c5ff;')
        self.grad_label.setStyleSheet('color: #b5c5ff;')
        self.loss_label.setToolTip('Loss shows how wrong the network is overall. Lower is better.')
        self.accuracy_label.setToolTip('Accuracy shows what percent of examples the network currently predicts correctly.')
        self.grad_label.setToolTip('Gradient norm measures how strong the learning signal is while the network updates.')
        side_layout.addWidget(self.loss_label)
        side_layout.addWidget(self.accuracy_label)
        side_layout.addWidget(self.grad_label)
        side_layout.addSpacing(12)
        side_layout.addWidget(self.explanation_title)
        side_layout.addWidget(self.explanation_label)
        side_layout.addSpacing(18)
        side_layout.addWidget(self.tip_title)
        side_layout.addWidget(self.tip_label)
        side_layout.addStretch()

        content = QWidget()
        content.setStyleSheet('background: transparent;')
        layout = QHBoxLayout(content)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(24)
        layout.addWidget(sidebar, 0)

        sidebar.setMinimumWidth(300)
        sidebar.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)

        main_area = QWidget()
        main_area.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        main_layout = QVBoxLayout(main_area)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(24)

        top_row = QWidget()
        top_row_layout = QHBoxLayout(top_row)
        top_row_layout.setContentsMargins(0, 0, 0, 0)
        top_row_layout.setSpacing(20)
        top_row_layout.addWidget(self.canvas, 2)

        right_stack = QWidget()
        right_layout = QVBoxLayout(right_stack)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(20)
        right_layout.addWidget(self.decision_canvas)
        right_layout.addWidget(self.scatter)
        top_row_layout.addWidget(right_stack, 1)

        self.canvas.setMinimumHeight(460)
        self.decision_canvas.setMinimumHeight(240)
        self.scatter.setMinimumHeight(240)
        self.history_canvas.setMinimumHeight(240)

        main_layout.addWidget(top_row, 3)
        main_layout.addWidget(self.history_canvas, 1)
        layout.addWidget(main_area, 1)

        wrapper = QWidget()
        wrapper_layout = QVBoxLayout(wrapper)
        wrapper_layout.setContentsMargins(24, 24, 24, 24)
        wrapper_layout.setSpacing(24)

        title = QLabel('Deep Neural Network Visualizer')
        title.setFont(QFont('Segoe UI', 24, QFont.Weight.Bold))
        subtitle = QLabel('Explore forward propagation, training dynamics, and decision boundaries in a cinematic desktop app.')
        subtitle.setWordWrap(True)
        subtitle.setStyleSheet('color: #a9b7d1;')
        wrapper_layout.addWidget(title)
        wrapper_layout.addWidget(subtitle)
        wrapper_layout.addWidget(content)

        scroll_area = SmoothScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(wrapper)
        scroll_area.setFrameShape(QScrollArea.NoFrame)
        scroll_area.setStyleSheet('background: transparent; border: none;')

        window_widget = QWidget()
        window_layout = QVBoxLayout(window_widget)
        window_layout.setContentsMargins(0, 0, 0, 0)
        window_layout.addWidget(scroll_area)
        self.setCentralWidget(window_widget)

    def on_dataset_changed(self, value: str) -> None:
        self.active_dataset = value
        self.update_dataset()

    def on_lr_changed(self, value: int) -> None:
        self.learning_rate = max(0.001, value / 1000.0)
        self.update_metrics()

    def on_architecture_changed(self, value: int) -> None:
        self.network = NeuralNetwork(
            [
                2,
                self.hidden_neurons_spinner.value(),
                self.hidden_neurons_spinner_2.value(),
                self.dataset_y.shape[1],
            ],
            ['linear', self.activation_selector.currentText(), self.activation_selector.currentText(), 'softmax'],
        )
        self.canvas.network = self.network
        self.scatter.network = self.network
        self.on_reset()

    def on_toggle_training(self) -> None:
        if self.running:
            self.timer.stop()
            self.running = False
            self.train_button.setText('Start training')
        else:
            self.running = True
            self.timer.start()
            self.train_button.setText('Pause training')
        self.train_button.setProperty('primary', True)
        self.update_metrics()

    def on_reset(self) -> None:
        self.network.reset()
        self.running = False
        self.timer.stop()
        self.train_button.setText('Start training')
        self.update_metrics()
        self.canvas.update()
        self.scatter.update()
        self.decision_canvas.update()
        self.history_canvas.set_history([], [], [])

    def update_dataset(self) -> None:
        self.dataset_x, self.dataset_y = create_dataset(self.active_dataset)
        if self.dataset_y.ndim == 1:
            self.dataset_y = np.eye(np.max(self.dataset_y) + 1)[self.dataset_y]
        self.network = NeuralNetwork(
            [
                2,
                self.hidden_neurons_spinner.value(),
                self.hidden_neurons_spinner_2.value(),
                self.dataset_y.shape[1],
            ],
            ['linear', self.activation_selector.currentText(), self.activation_selector.currentText(), 'softmax'],
        )
        self.canvas.network = self.network
        self.scatter.network = self.network
        self.decision_canvas.network = self.network
        self.scatter.set_data(self.dataset_x, self.dataset_y)
        self.decision_canvas.set_data(self.dataset_x, self.dataset_y)
        self.on_reset()

    def training_step(self) -> None:
        self.network.train_batch(self.dataset_x, self.dataset_y, self.learning_rate)
        self.update_metrics()
        self.scatter.update()
        self.canvas.update()
        self.decision_canvas.update()
        self.history_canvas.set_history(self.network.loss_history, self.network.accuracy_history, self.network.gradient_history)

    def update_metrics(self) -> None:
        self.loss_label.setText(f'Loss: {self.network.loss:.4f}')
        self.accuracy_label.setText(f'Accuracy: {self.network.accuracy:.1f}%')
        self.grad_label.setText(f'Grad norm: {self.network.gradient_norm:.3f}')
        self.update_explanation()

    def toggle_lesson_mode(self) -> None:
        self.lesson_active = not self.lesson_active
        if self.lesson_active:
            self.lesson_button.setText('Stop lesson')
            self.lesson_step_index = 0
            self.prev_step_button.setEnabled(False)
            self.next_step_button.setEnabled(len(self.lesson_steps) > 1)
            self.lesson_status_label.setText(f'Step 1 of {len(self.lesson_steps)}: {self.lesson_steps[0]["title"]}')
        else:
            self.lesson_button.setText('Start lesson')
            self.prev_step_button.setEnabled(False)
            self.next_step_button.setEnabled(False)
            self.lesson_status_label.setText('Lesson mode is off.')
        self.update_explanation()

    def prev_lesson_step(self) -> None:
        self.set_lesson_step(self.lesson_step_index - 1)

    def next_lesson_step(self) -> None:
        self.set_lesson_step(self.lesson_step_index + 1)

    def set_lesson_step(self, index: int) -> None:
        self.lesson_step_index = max(0, min(index, len(self.lesson_steps) - 1))
        self.prev_step_button.setEnabled(self.lesson_step_index > 0)
        self.next_step_button.setEnabled(self.lesson_step_index < len(self.lesson_steps) - 1)
        step = self.lesson_steps[self.lesson_step_index]
        self.lesson_status_label.setText(f'Step {self.lesson_step_index + 1} of {len(self.lesson_steps)}: {step["title"]}')
        self.update_explanation()

    def update_explanation(self) -> None:
        if self.lesson_active:
            step = self.lesson_steps[self.lesson_step_index]
            explanation_text = (
                f'<b>{step["title"]}</b><br>'
                f'{step["text"]}<br><br>'
                f'Use the Next and Previous buttons to move through the lesson.'
            )
            self.explanation_label.setText(explanation_text)
            return

        dataset_desc = get_dataset_description(self.active_dataset)
        activation_desc = get_activation_description(self.activation_selector.currentText())
        status = 'training' if self.running else 'paused'
        explanation_text = (
            f'<b>Dataset:</b> {self.active_dataset} — {dataset_desc}<br>'
            f'<b>Hidden neurons:</b> {self.hidden_neurons_spinner.value()} — more neurons let the network learn more detailed patterns.<br>'
            f'<b>Activation:</b> {self.activation_selector.currentText()} — {activation_desc}<br>'
            f'<b>Learning rate:</b> {self.learning_rate:.3f} — how big each change is when the network learns.<br><br>'
            f'<b>Training status:</b> {status}. Loss measures overall error, accuracy shows how many examples are correct, and gradient norm shows how strong the weight updates are.<br><br>'
            f'<b>Visual guide:</b> the network view shows neurons firing, the decision boundary shows how the model splits the input space, and the history chart shows how the network improves over time.'
        )
        self.explanation_label.setText(explanation_text)

    def show_math_dialog(self) -> None:
        dlg = MathDialog(self)
        dlg.exec()

    def open_compare_dialog(self) -> None:
        dlg = CompareDialog(self)
        dlg.exec()

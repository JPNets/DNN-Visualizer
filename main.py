from PySide6.QtWidgets import QApplication
from visualizer.app import NeuralVisualizerWindow
import sys


def main() -> int:
    app = QApplication(sys.argv)
    window = NeuralVisualizerWindow()
    window.show()
    return app.exec()


if __name__ == '__main__':
    raise SystemExit(main())

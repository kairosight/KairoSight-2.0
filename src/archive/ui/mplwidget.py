# -*- coding: utf-8 -*-
# Adapted from examples in "Matplotlib for Python Developers", Sandro Tosi, Packt Publishing
# https://www.packtpub.com/sites/default/files/sample_chapters/7900-matplotlib-for-python-developers-sample-chapter-6-embedding-matplotlib-in-qt-4.pdf

# Python Qt5 bindings for GUI objects
from PyQt5 import QtGui
from PyQt5.QtWidgets import QApplication, QMainWindow, QMenu, QVBoxLayout, QSizePolicy, QMessageBox, QWidget, \
    QPushButton
# import the Qt5Agg FigureCanvas object, that binds Figure to
# Qt4Agg backend. It also inherits from QWidget
from matplotlib.backends.backend_qt5agg \
    import FigureCanvasQTAgg as FigureCanvas

# Matplotlib Figure object
from matplotlib.figure import Figure


class MplCanvas(FigureCanvas):
    """Class to represent the FigureCanvas widget"""

    def __init__(self):
        # setup Matplotlib Figure and Axis
        self.fig = Figure()
        self.ax = self.fig.add_subplot(111)
        # initialization of the canvas
        FigureCanvas.__init__(self, self.fig)
        # we define the widget as expandable
        FigureCanvas.setSizePolicy(self,
                                   QSizePolicy.Expanding,
                                   QSizePolicy.Expanding)
        # notify the system of updated policy
        FigureCanvas.updateGeometry(self)


class MplWidget(QWidget):
    """Widget defined in Qt Designer"""

    def __init__(self, parent=None):
        # initialization of Qt MainWindow widget
        super(MplWidget, self).__init__(parent)
        # set the canvas to the Matplotlib widget
        self.canvas = MplCanvas()
        # create a vertical box layout
        self.vbl = QVBoxLayout()
        # add mpl widget to vertical box
        self.vbl.addWidget(self.canvas)
        # set the layout to th vertical box
        self.setLayout(self.vbl)

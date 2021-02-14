

from PyQt5 import QtGui
from PyQt5.QtWidgets import QApplication, QMainWindow, QMenu, QVBoxLayout, QSizePolicy, QMessageBox, QWidget

from pyqtgraph.widgets.GraphicsLayoutWidget import GraphicsLayoutWidget
from pyqtgraph import ImageItem, HistogramLUTItem, HistogramLUTWidget


class GraphicsWidget(QWidget):
    """Widget defined in Qt Designer"""

    def __init__(self, parent=None):
        # initialization of widget
        super(GraphicsWidget, self).__init__(parent)

        # Create a central Graphics Layout Widget
        self.widget = GraphicsLayoutWidget()

        # A plot area (ViewBox + axes) for displaying the image
        self.p1 = self.widget.addPlot()
        # Item for displaying an array of image data stacks
        self.stacks = []
        img = ImageItem()
        self.p1.addItem(img)
        self.stacks.append(img)

        # create a vertical box layout
        self.vbl = QVBoxLayout()
        # add widget to vertical box
        self.vbl.addWidget(self.widget)
        # set the layout to the vertical box
        self.setLayout(self.vbl)

        # Levels/color control with a histogram
        # self.hist = HistogramLUTWidget()
        # self.hist.setImageItem(self.img)
        # parent.horizontalLayout_View.addWidget(self.hist)
        # # self.widget.addWidget(self.hist, 0, 1)
        # self.hist.vb.setMouseEnabled(y=False)  # makes user interaction a little easier

        # Create an array of histograms
        self.histograms = []
        # Levels/color control with a histogram
        hist = HistogramLUTItem()
        hist.vb.setMouseEnabled(y=False)  # makes user interaction a little easier
        hist.setImageItem(img)
        self.widget.addItem(hist)
        self.histograms.append(hist)

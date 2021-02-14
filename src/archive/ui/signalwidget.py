

from PyQt5 import QtCore, QtGui, QtWidgets


class SignalWidget(QtWidgets.QWidget):
    """Widget based on another defined in Qt Designer"""

    def __init__(self, parent=None):
        # initialization of widget
        super(SignalWidget, self).__init__(parent)
        # self.widget = QtWidgets.QWidget()
        self.alpha = 255

        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.sizePolicy().hasHeightForWidth())
        self.setSizePolicy(sizePolicy)
        self.setObjectName("SignalWidget")
        self.horizontalLayout_SignalWidget = QtWidgets.QHBoxLayout(self)
        self.horizontalLayout_SignalWidget.setContentsMargins(-1, 0, -1, 0)
        self.horizontalLayout_SignalWidget.setObjectName("horizontalLayout_SignalWidget")
        self.comboBoxSignal = QtWidgets.QComboBox(self)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.comboBoxSignal.sizePolicy().hasHeightForWidth())
        self.comboBoxSignal.setSizePolicy(sizePolicy)
        self.comboBoxSignal.setObjectName("comboBoxSignal")
        self.horizontalLayout_SignalWidget.addWidget(self.comboBoxSignal)
        self.checkBoxSignal = QtWidgets.QCheckBox(self)
        self.checkBoxSignal.setText("")
        self.checkBoxSignal.setCheckable(True)
        self.checkBoxSignal.setChecked(True)
        self.checkBoxSignal.setObjectName("checkBoxSignal")
        self.horizontalLayout_SignalWidget.addWidget(self.checkBoxSignal)

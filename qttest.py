# from PyQt5 import QtWidgets
# import sys
#
# app = QtWidgets.QApplication(sys.argv)
# window = QtWidgets.QWidget()
# window.show()
# sys.exit(app.exec_())

import sys
import test
from PyQt5.QtWidgets import QApplication, QMainWindow

if __name__ == '__main__':
    app = QApplication(sys.argv)
    MainWindow = QMainWindow()
    ui = test.Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
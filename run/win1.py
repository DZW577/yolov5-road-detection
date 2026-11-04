from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1000, 600)

        # 主部件
        self.centralwidget = QtWidgets.QWidget(MainWindow)

        # 主垂直布局
        self.main_layout = QtWidgets.QVBoxLayout(self.centralwidget)

        # 上半部分：左右显示区 (水平布局)
        self.display_layout = QtWidgets.QHBoxLayout()

        # --- 左侧：原图/视频流 ---
        self.label_original = QtWidgets.QLabel()
        self.label_original.setFrameShape(QtWidgets.QFrame.Box)
        self.label_original.setText("原图 / 视频流")
        self.label_original.setAlignment(QtCore.Qt.AlignCenter)
        self.label_original.setStyleSheet("background-color: #f0f0f0; font-size: 16px;")

        # 🌟 核心修复 1：忽略图片自带的尺寸建议，绝不允许图片撑大 Label
        self.label_original.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Ignored)
        # 🌟 核心修复 2：让图片自动适应 Label 大小
        self.label_original.setScaledContents(True)

        # 🌟 核心修复 3：设置 stretch=1 保证左右五五平分
        self.display_layout.addWidget(self.label_original, stretch=1)

        # --- 右侧：检测结果 ---
        self.label_result = QtWidgets.QLabel()
        self.label_result.setFrameShape(QtWidgets.QFrame.Box)
        self.label_result.setText("检测结果区")
        self.label_result.setAlignment(QtCore.Qt.AlignCenter)
        self.label_result.setStyleSheet("background-color: #f0f0f0; font-size: 16px;")

        # 同样加上尺寸限制
        self.label_result.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Ignored)
        self.label_result.setScaledContents(True)

        self.display_layout.addWidget(self.label_result, stretch=1)

        # 将水平布局加入主布局
        self.main_layout.addLayout(self.display_layout)

        # 下半部分：按钮区 (水平布局)
        self.btn_layout = QtWidgets.QHBoxLayout()

        self.btn_select = QtWidgets.QPushButton("选择图片")
        self.btn_select.setMinimumHeight(40)
        self.btn_select.setStyleSheet("font-size: 14px; font-weight: bold;")

        self.btn_toggle_cam = QtWidgets.QPushButton("打开摄像头")
        self.btn_toggle_cam.setMinimumHeight(40)
        self.btn_toggle_cam.setStyleSheet("font-size: 14px; font-weight: bold;")

        self.btn_detect = QtWidgets.QPushButton("开始检测")
        self.btn_detect.setMinimumHeight(40)
        self.btn_detect.setStyleSheet("background-color: #4CAF50; color: white; font-size: 14px; font-weight: bold;")

        self.btn_layout.addWidget(self.btn_select)
        self.btn_layout.addWidget(self.btn_toggle_cam)
        self.btn_layout.addWidget(self.btn_detect)

        self.main_layout.addLayout(self.btn_layout)

        MainWindow.setCentralWidget(self.centralwidget)

        # 菜单栏和状态栏
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "YOLOv5 道路病害检测系统"))
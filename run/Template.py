import sys
import cv2
import requests
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QFileDialog, QMainWindow, QMessageBox
from win1 import Ui_MainWindow


# --- 异步请求线程，防止发送 Flask 请求时界面卡死 ---
class FlaskRequestThread(QtCore.QThread):
    result_signal = QtCore.pyqtSignal(bytes)  # 成功信号
    error_signal = QtCore.pyqtSignal(str)  # 失败信号

    def __init__(self, image_bytes):
        super().__init__()
        self.image_bytes = image_bytes

    def run(self):
        try:
            files = {'file': ('image.jpg', self.image_bytes, 'image/jpeg')}
            # 向 Flask 后端发送 POST 请求
            response = requests.post('http://127.0.0.1:5000/predict', files=files, timeout=10)
            if response.status_code == 200:
                self.result_signal.emit(response.content)
            else:
                self.error_signal.emit(f"服务端错误: {response.status_code}\n{response.text}")
        except requests.exceptions.ConnectionError:
            self.error_signal.emit("连接失败！请确保 Flask 后端 (app.py) 已启动。")
        except Exception as e:
            self.error_signal.emit(str(e))


# --- 主窗口逻辑 ---
class PyQtMainEntry(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

        # 参数初始化
        self.camera = cv2.VideoCapture(0)
        self.is_camera_opened = False
        self.current_frame_bytes = None  # 存放当前需要检测的图片字节流

        # 定时器用于读取摄像头 (50ms 约等于 20 帧，对 UI 渲染更友好，不易抽搐)
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.query_frame)
        self.timer.setInterval(50)

        # 绑定按钮事件
        self.btn_select.clicked.connect(self.select_image)
        self.btn_toggle_cam.clicked.connect(self.toggle_camera)
        self.btn_detect.clicked.connect(self.detect_image)

    def select_image(self):
        """选择本地图片进行检测"""
        if self.is_camera_opened:
            self.toggle_camera()  # 选图时自动关闭摄像头

        file_path, _ = QFileDialog.getOpenFileName(self, "选择图片", "", "Images (*.png *.jpg *.jpeg *.bmp)")
        if file_path:
            with open(file_path, 'rb') as f:
                self.current_frame_bytes = f.read()

            # 显示原图
            pixmap = QtGui.QPixmap(file_path)
            self.label_original.setPixmap(pixmap.scaled(
                self.label_original.size(),
                QtCore.Qt.KeepAspectRatio,
                QtCore.Qt.SmoothTransformation
            ))
            # 清空之前的结果
            self.label_result.setText("等待检测...")
            self.label_result.setPixmap(QtGui.QPixmap())

    def toggle_camera(self):
        """打开/关闭摄像头"""
        self.is_camera_opened = not self.is_camera_opened
        if self.is_camera_opened:
            self.btn_toggle_cam.setText("关闭摄像头")
            self.btn_toggle_cam.setStyleSheet(
                "background-color: #f44336; color: white; font-weight: bold; font-size: 14px;")
            self.timer.start()
        else:
            self.btn_toggle_cam.setText("打开摄像头")
            self.btn_toggle_cam.setStyleSheet("font-size: 14px; font-weight: bold;")
            self.timer.stop()
            self.label_original.clear()
            self.label_original.setText("原图 / 视频流")
            self.current_frame_bytes = None

    def query_frame(self):
        """定时器回调：读取摄像头帧并显示"""
        ret, frame = self.camera.read()
        if ret:
            # 1. 水平翻转（镜子效果）
            frame = cv2.flip(frame, 1)

            # 2. 保存当前帧的原始字节流给 Flask 检测用（不缩放，保证检测精度）
            self.current_frame_bytes = cv2.imencode('.jpg', frame)[1].tobytes()

            # 3. 转换颜色空间 BGR -> RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame_rgb.shape

            # 4. 防抽搐核心：加入 bytes_per_line，并调用 .copy() 防止内存被 Python 回收
            bytes_per_line = ch * w
            q_image = QtGui.QImage(frame_rgb.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888).copy()

            # 5. 等比例缩放显示，防止画面被拉伸变形
            pixmap = QtGui.QPixmap.fromImage(q_image)
            self.label_original.setPixmap(pixmap.scaled(
                self.label_original.size(),
                QtCore.Qt.KeepAspectRatio,
                QtCore.Qt.SmoothTransformation
            ))

    def detect_image(self):
        """将当前画面/图片发送给 Flask 后端"""
        if self.current_frame_bytes is None:
            QMessageBox.warning(self, "警告", "请先选择一张图片或打开摄像头！")
            return

        self.btn_detect.setText("检测中...")
        self.btn_detect.setEnabled(False)

        # 启动后台线程发送请求
        self.req_thread = FlaskRequestThread(self.current_frame_bytes)
        self.req_thread.result_signal.connect(self.on_detect_success)
        self.req_thread.error_signal.connect(self.on_detect_error)
        self.req_thread.start()

    def on_detect_success(self, image_bytes):
        """处理 Flask 返回的检测结果图"""
        pixmap = QtGui.QPixmap()
        pixmap.loadFromData(image_bytes)
        self.label_result.setPixmap(pixmap.scaled(
            self.label_result.size(),
            QtCore.Qt.KeepAspectRatio,
            QtCore.Qt.SmoothTransformation
        ))
        self.btn_detect.setText("开始检测")
        self.btn_detect.setEnabled(True)

    def on_detect_error(self, error_msg):
        """处理网络异常"""
        QMessageBox.critical(self, "检测失败", error_msg)
        self.btn_detect.setText("开始检测")
        self.btn_detect.setEnabled(True)

    def closeEvent(self, event):
        """重写关闭事件，确保摄像头被安全释放，防止后台一直亮灯"""
        if self.camera.isOpened():
            self.timer.stop()
            self.camera.release()
        event.accept()


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = PyQtMainEntry()
    window.show()
    sys.exit(app.exec_())
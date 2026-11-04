import io
import os
import sys
import cv2
import torch
import numpy as np
from flask import Flask, request, send_file, jsonify, render_template

# ==========================================
# 1. 路径配置
# ==========================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
YOLOV5_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '..', 'yolov5-7.0'))
WEIGHTS_PATH = os.path.join(YOLOV5_DIR, 'runs', 'train', 'mask_exp_4g', 'weights', 'best.pt')

sys.path.append(YOLOV5_DIR)
from models.experimental import attempt_load
from utils.augmentations import letterbox
from utils.general import non_max_suppression, scale_boxes
from utils.plots import Annotator, colors

# 强制使用中文字体
import utils.plots

utils.plots.Annotator.font = 'C:/Windows/Fonts/msyh.ttc'

# ==========================================
# 2. 初始化 Flask 和 加载模型
# ==========================================
app = Flask(__name__)

print("正在加载模型...")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
half = device.type != 'cpu'

model = attempt_load(WEIGHTS_PATH, device=device)
stride = int(model.stride.max())

# 覆盖乱码的 names
names = ['坑槽', '松散', '横向裂纹', '纵向裂纹', '网状裂纹', '横向修补', '纵向修补',
         '块状修补', '网状修补', '标线模糊', '抛洒物', '锥形桶', '圆形桶', '人',
         '方形雨水盖', '圆形雨水盖', '圆形井盖', '护栏损坏']

if half:
    model.half()
print(f"✅ 模型加载成功！运行设备: {device}")


# ==========================================
# 3. 核心推理函数
# ==========================================
def predict_image(image_bytes):
    nparr = np.frombuffer(image_bytes, np.uint8)
    img0 = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    img = letterbox(img0, 640, stride=stride)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()
    img /= 255.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    pred = model(img, augment=False)[0]
    pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45, max_det=1000)

    annotator = Annotator(img0, line_width=3, example=str(names))

    for det in pred:
        if len(det):
            det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], img0.shape).round()
            for *xyxy, conf, cls in reversed(det):
                c = int(cls)
                label = f'{names[c]} {conf:.2f}'
                annotator.box_label(xyxy, label, color=colors(c, True))

    return annotator.result()


# ==========================================
# 4. API 路由
# ==========================================
@app.route('/')
def index():
    # 渲染 templates 文件夹下的 index.html
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def api_predict():
    if 'file' not in request.files:
        return jsonify({"error": "没有找到图片文件"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "文件名为空"}), 400

    try:
        image_bytes = file.read()
        result_img = predict_image(image_bytes)

        is_success, buffer = cv2.imencode(".jpg", result_img)
        io_buf = io.BytesIO(buffer)
        return send_file(io_buf, mimetype='image/jpeg')

    except Exception as e:
        print("推理发生异常:", e)
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    print("Flask 服务正在启动...")
    app.run(host='0.0.0.0', port=5000, debug=False)
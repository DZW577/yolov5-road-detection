import io
import base64
import torch
import cv2
import numpy as np
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # 允许跨域请求

# ==========================================
# 1. 加载 YOLOv5 模型 (使用绝对路径)
# ==========================================
YOLOV5_DIR = r'E:\杂物\道路标签标注\yolov5-7 111\yolov5-7.0'
WEIGHTS_PATH = r'E:\杂物\道路标签标注\yolov5-7 111\yolov5-7.0\runs\train\mask_exp_4g2\weights\best.pt'

print("正在加载模型...")
model = torch.hub.load(YOLOV5_DIR, 'custom', path=WEIGHTS_PATH, source='local')
model.conf = 0.35  # 置信度阈值
model.iou = 0.45  # NMS 阈值
print("✅ 模型加载成功！")

# 🌟 核心修复：强行覆盖模型里存的乱码类别名
names = ['坑槽', '松散', '横向裂纹', '纵向裂纹', '网状裂纹', '横向修补', '纵向修补',
         '块状修补', '网状修补', '标线模糊', '抛洒物', '锥形桶', '圆形桶', '人',
         '方形雨水盖', '圆形雨水盖', '圆形井盖', '护栏损坏']


# 2. 手机端网页路由
@app.route('/')
def index():
    return render_template('index.html')


# 3. 接收图片并推理的路由
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        image_data = data['image']

        # 去掉 Base64 头信息并解码
        img_base64 = image_data.split(',')[1]
        img_bytes = base64.b64decode(img_base64)

        # 转换为 OpenCV 格式
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # YOLO 推理
        results = model(img)
        df = results.pandas().xyxy[0]

        detections = []
        for index, row in df.iterrows():
            cls_id = int(row['class'])
            # 🌟 核心修复：用正确的 names 替换乱码
            class_name = names[cls_id] if cls_id < len(names) else str(row['name'])

            detections.append({
                'xmin': float(row['xmin']),
                'ymin': float(row['ymin']),
                'xmax': float(row['xmax']),
                'ymax': float(row['ymax']),
                'confidence': float(row['confidence']),
                'class_name': class_name
            })

        return jsonify({'status': 'success', 'detections': detections})

    except Exception as e:
        print("推理报错:", e)
        return jsonify({'status': 'error', 'message': str(e)})


if __name__ == '__main__':
    # 关闭 debug 模式，防止模型重复加载爆显存
    app.run(host='0.0.0.0', port=5000, debug=False)
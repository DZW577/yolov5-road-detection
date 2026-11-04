import json
import os

# ================= 配置区域 =================
# 1. 你的10个类别字典（必须和 data.yaml 里的顺序对应）
name2id = {
    '坑槽': 0, '抛洒物': 1, '护栏损坏': 2, '标线模糊不完整': 3, '标志遮挡': 4,
    '横向裂缝': 5, '纵向裂缝': 6, '网状裂缝': 7, '横向修补': 8, '纵向修补': 9
}

# 2. 根据你的截图适配的路径（相对 1.py 所在的目录）
JSON_FOLDER_PATH = r"./photo/train/json"  # 你的json文件夹
TXT_SAVE_PATH = r"./photo/train/labels"  # txt要保存的文件夹


# ============================================

def convert(img_size, box):
    """将 LabelMe 的绝对坐标转为 YOLO 的归一化坐标 [x_center, y_center, width, height]"""
    dw = 1.0 / (img_size[0])
    dh = 1.0 / (img_size[1])

    # 确保取到的是左上角和右下角的坐标（防止反向拉框导致宽高为负数）
    x_min = min(box[0], box[2])
    x_max = max(box[0], box[2])
    y_min = min(box[1], box[3])
    y_max = max(box[1], box[3])

    # 计算中心点坐标和宽高 (修复了原版代码减去1的bug)
    x_center = (x_min + x_max) / 2.0
    y_center = (y_min + y_max) / 2.0
    w = x_max - x_min
    h = y_max - y_min

    # 归一化
    x_center = x_center * dw
    w = w * dw
    y_center = y_center * dh
    h = h * dh

    return (x_center, y_center, w, h)


def decode_json(json_folder, json_name, txt_save_dir):
    # 拼接输出 txt 的路径
    txt_name = os.path.join(txt_save_dir, json_name.replace('.json', '.txt'))
    json_path = os.path.join(json_folder, json_name)

    # 注意：中文标签建议统一使用 utf-8，如果报错可以换成 gbk
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    img_w = data['imageWidth']
    img_h = data['imageHeight']

    with open(txt_name, 'w', encoding='utf-8') as txt_file:
        for i in data['shapes']:
            label_name = i['label']

            # 确保是矩形框且标签在我们的字典里
            if i['shape_type'] == 'rectangle' and label_name in name2id:
                x1 = i['points'][0][0]
                y1 = i['points'][0][1]
                x2 = i['points'][1][0]
                y2 = i['points'][1][1]

                bb = (x1, y1, x2, y2)
                bbox = convert((img_w, img_h), bb)

                # 写入标准 YOLO 格式：类别ID x_center y_center width height
                txt_file.write(f"{name2id[label_name]} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}\n")


if __name__ == "__main__":
    # 如果输出文件夹不存在，自动创建
    if not os.path.exists(TXT_SAVE_PATH):
        os.makedirs(TXT_SAVE_PATH)
        print(f"已创建输出目录: {TXT_SAVE_PATH}")

    # 遍历 json 文件夹
    count = 0
    for json_name in os.listdir(JSON_FOLDER_PATH):
        if json_name.endswith('.json'):  # 只处理 json 文件
            try:
                decode_json(JSON_FOLDER_PATH, json_name, TXT_SAVE_PATH)
                count += 1
            except Exception as e:
                print(f"处理 {json_name} 时出错: {e}")

    print(f"\n转换完成！共处理了 {count} 个 json 文件。")
    print(f"请去 {TXT_SAVE_PATH} 检查生成的 txt 文件。")
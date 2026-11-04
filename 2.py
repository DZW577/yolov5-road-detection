import os
import random
import shutil

# ================= 配置区域 =================
# 你的数据根目录（注意路径末尾不要加斜杠）
BASE_DIR = r"E:\杂物\道路标签标注\yolov5-7 111\photo"

# 训练集和验证集的路径
TRAIN_IMG_DIR = os.path.join(BASE_DIR, "train", "images")
TRAIN_LBL_DIR = os.path.join(BASE_DIR, "train", "labels")
VAL_IMG_DIR = os.path.join(BASE_DIR, "valid", "images")
VAL_LBL_DIR = os.path.join(BASE_DIR, "valid", "labels")

# 划分比例（0.15 代表从 6000 张里抽 15% 即 900 张给验证集）
SPLIT_RATIO = 0.15


# ============================================

def main():
    # 1. 确保验证集文件夹存在
    os.makedirs(VAL_IMG_DIR, exist_ok=True)
    os.makedirs(VAL_LBL_DIR, exist_ok=True)

    # 2. 获取训练集里所有的图片文件
    all_images = [f for f in os.listdir(TRAIN_IMG_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    total_images = len(all_images)
    print(f"训练集共有 {total_images} 张图片。")

    # 3. 计算需要抽取的数量并随机打乱
    num_to_move = int(total_images * SPLIT_RATIO)
    print(f"计划移动 {num_to_move} 张图片到验证集...")

    random.shuffle(all_images)
    images_to_move = all_images[:num_to_move]

    # 4. 开始移动图片和标签
    move_count = 0
    missing_label_count = 0

    for img_name in images_to_move:
        # 获取文件基础名（不带后缀）
        base_name = os.path.splitext(img_name)[0]
        label_name = base_name + ".txt"

        src_img_path = os.path.join(TRAIN_IMG_DIR, img_name)
        src_lbl_path = os.path.join(TRAIN_LBL_DIR, label_name)

        dst_img_path = os.path.join(VAL_IMG_DIR, img_name)
        dst_lbl_path = os.path.join(VAL_LBL_DIR, label_name)

        # 移动图片
        shutil.move(src_img_path, dst_img_path)

        # 移动标签（如果对应的 txt 存在的话）
        if os.path.exists(src_lbl_path):
            shutil.move(src_lbl_path, dst_lbl_path)
            move_count += 1
        else:
            missing_label_count += 1
            # 注意：YOLO 中如果没有 txt，会被当成背景图（负样本），这本身是合法的
            # 但如果你确定每张图都该有标签，那可能是你的标签文件命名或转换有问题

    print("=" * 30)
    print(f"划分完成！")
    print(f"成功移动了 {move_count} 对图片和标签到验证集。")
    if missing_label_count > 0:
        print(f"注意：有 {missing_label_count} 张图片没有找到对应的 txt 标签，已将其作为背景图移入验证集。")
    print("=" * 30)


if __name__ == "__main__":
    main()
import numpy as np
import cv2
import os
from PIL import Image, ImageDraw, ImageFont


def generate_test_images():
    """生成各种测试图像"""
    test_images = []

    # 创建输出目录
    output_dir = "./test_images"
    os.makedirs(output_dir, exist_ok=True)

    # 1. 纯色图像
    black = np.zeros((256, 256, 3), dtype=np.uint8)
    cv2.imwrite(os.path.join(output_dir, "test_black.jpg"), black)
    test_images.append(os.path.join(output_dir, "test_black.jpg"))

    white = np.ones((256, 256, 3), dtype=np.uint8) * 255
    cv2.imwrite(os.path.join(output_dir, "test_white.jpg"), white)
    test_images.append(os.path.join(output_dir, "test_white.jpg"))

    red = np.zeros((256, 256, 3), dtype=np.uint8)
    red[:, :, 2] = 255
    cv2.imwrite(os.path.join(output_dir, "test_red.jpg"), red)
    test_images.append(os.path.join(output_dir, "test_red.jpg"))

    # 2. 渐变图像
    gradient = np.zeros((256, 256, 3), dtype=np.uint8)
    for i in range(256):
        gradient[:, i, 0] = i
        gradient[:, i, 1] = 255 - i
        gradient[:, i, 2] = 128
    cv2.imwrite(os.path.join(output_dir, "test_gradient.jpg"), gradient)
    test_images.append(os.path.join(output_dir, "test_gradient.jpg"))

    # 3. 棋盘格图像
    checkerboard = np.zeros((256, 256, 3), dtype=np.uint8)
    for i in range(0, 256, 32):
        for j in range(0, 256, 32):
            if (i // 32 + j // 32) % 2 == 0:
                checkerboard[i:i + 32, j:j + 32] = [255, 255, 255]
    cv2.imwrite(os.path.join(output_dir, "test_checkerboard.jpg"), checkerboard)
    test_images.append(os.path.join(output_dir, "test_checkerboard.jpg"))

    # 4. 随机噪声图像
    random_noise = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
    cv2.imwrite(os.path.join(output_dir, "test_random.jpg"), random_noise)
    test_images.append(os.path.join(output_dir, "test_random.jpg"))

    # 5. 文本图像
    text_image = np.ones((256, 256, 3), dtype=np.uint8) * 200
    pil_img = Image.fromarray(text_image)
    draw = ImageDraw.Draw(pil_img)
    try:
        font = ImageFont.truetype("arial.ttf", 40)
    except:
        font = ImageFont.load_default()

    draw.text((50, 100), "Hello World", fill=(0, 0, 0), font=font)
    draw.text((30, 150), "Test Image", fill=(255, 0, 0), font=font)
    text_array = np.array(pil_img)
    cv2.imwrite(os.path.join(output_dir, "test_text.jpg"), text_array[:, :, ::-1])
    test_images.append(os.path.join(output_dir, "test_text.jpg"))

    # 6. 简易 Lena-like 图像
    lena_like = np.zeros((256, 256, 3), dtype=np.uint8)
    cv2.circle(lena_like, (128, 128), 60, (255, 255, 255), -1)
    cv2.rectangle(lena_like, (50, 50), (100, 100), (0, 255, 0), -1)
    cv2.rectangle(lena_like, (150, 150), (200, 200), (0, 0, 255), -1)
    cv2.imwrite(os.path.join(output_dir, "test_pattern.jpg"), lena_like)
    test_images.append(os.path.join(output_dir, "test_pattern.jpg"))

    print(f"已生成 {len(test_images)} 张测试图像，保存在 {output_dir}：")
    for img in test_images:
        print("  -", img)

    return test_images


generate_test_images()

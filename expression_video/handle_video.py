import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# 读取视频
cap = cv2.VideoCapture("neutral_ori.mp4")
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 创建 VideoWriter 保存新视频
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter("camera_error.mp4", fourcc, fps, (width, height))

# 设置中文字体（需指定字体文件路径）
font_path = "SimHei.ttf"  # 替换为你的中文字体文件
font = ImageFont.truetype(font_path, 40)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 转换帧为 PIL 图像
    frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(frame_pil)

    # 计算文本尺寸（使用 textbbox）
    text = "致命错误！摄像头断开连接！"
    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]

    # 在右上角添加文字
    draw.text(
        (width - text_width - 20, 20),  # 右上角偏移
        text,
        font=font,
        fill=(0, 0, 0),  # 白色
    )

    # 转换回 OpenCV 格式
    frame_with_text = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)

    # 写入新视频
    out.write(frame_with_text)

cap.release()
out.release()
print("视频处理完成！")

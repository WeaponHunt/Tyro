import cv2
import time
import threading
import os
import numpy as np
from fastapi import FastAPI, Body
from fastapi.responses import JSONResponse
import uvicorn
from dotenv import load_dotenv
import requests

try:
    from PIL import Image, ImageDraw, ImageFont
except Exception:
    Image = None
    ImageDraw = None
    ImageFont = None

from upload_watcher import is_image_path, scan_uploaded_dir, start_watcher, UPLOADED_EXPRESSIONS

# 加载环境变量
load_dotenv()

# 获取表情视频基础路径
VIDEO_BASE_PATH = "./expression_video/"

gaze_mode = False

#显示视线的思路，接收一个post请求，如果注视状态发生变化则改变neutral表情的映射，那这样current_video就也需要改成表情

# 表情对应的视频文件路径配置
expressions = {
    "happy": f"{VIDEO_BASE_PATH}happy.mp4",
    "angry": f"{VIDEO_BASE_PATH}angry.mp4",
    "sad": f"{VIDEO_BASE_PATH}sad.mp4",
    "hatred": f"{VIDEO_BASE_PATH}hatred.mp4",
    "scared": f"{VIDEO_BASE_PATH}scared.mp4",
    "surprised": f"{VIDEO_BASE_PATH}surprised.mp4",
    "more-happy": f"{VIDEO_BASE_PATH}surprised.mp4",
    "neutral": f"{VIDEO_BASE_PATH}neutral_ori.mp4",
    "dizzy": f"{VIDEO_BASE_PATH}dizzy.mp4",
    "evil_smile": f"{VIDEO_BASE_PATH}evil_smile.mp4",
    "nauty_smile": f"{VIDEO_BASE_PATH}evil_smile.mp4",
    "pitying": f"{VIDEO_BASE_PATH}sympathy.mp4",
    "sleep": f"{VIDEO_BASE_PATH}sleep1.mp4",
    "say_hallo": f"{VIDEO_BASE_PATH}say_hallo.mp4",
    "function_display": f"{VIDEO_BASE_PATH}function_express.mp4",
    "camera_error": f"{VIDEO_BASE_PATH}camera_error.mp4",
}

# 默认表情
DEFAULT_EXPRESSION = os.getenv("EXPRESSION_DEFAULT", "neutral")
#current_video = expressions[DEFAULT_EXPRESSION]
current_expression = DEFAULT_EXPRESSION
recording_overlay = False
status_overlay_text = ""
status_overlay_visible = True
heard_overlay_text = ""
heard_overlay_visible = False
status_font = None

# 线程同步，通知播放线程切换视频
stop_event = threading.Event()
state_lock = threading.Lock()

# 实例化 FastAPI 应用
app = FastAPI(title="Expression Player API", description="API 助力 OpenCV 播放表情视频")


@app.get("/expressions")
async def get_available_expressions():
    """
    返回所有可用的表情列表
    """
    return {"expressions": list(expressions.keys()) + list(UPLOADED_EXPRESSIONS.keys())}

@app.post("/if_gaze/{if_gaze}")
async def change_gazemode(if_gaze:str):
    global stop_event,gaze_mode,expressions,VIDEO_BASE_PATH
    if if_gaze == "true":
        if_gaze = True
    else:
        if_gaze = False
    if if_gaze!=gaze_mode:
        gaze_mode = if_gaze
        if gaze_mode:
            expressions['neutral'] = f"{VIDEO_BASE_PATH}neutral.mp4"
        else:
            expressions['neutral'] = f"{VIDEO_BASE_PATH}neutral_notlisten.mp4"
        stop_event.set()



@app.post("/expression/{expression}")
async def change_expression(expression: str):
    """
    设置指定表情
    """
    #global current_video, stop_event
    global current_expression, stop_event

    if expression not in expressions and expression not in UPLOADED_EXPRESSIONS:
        return JSONResponse(
            status_code=404, content={"message": f"Expression '{expression}' not found"}
        )

    # if current_video != expressions[expression]:
    #     current_video = expressions[expression]
    #     stop_event.set()  # 通知播放线程立即切换
    #     return {"message": f"Switching to '{expression}' expression"}
    if current_expression != expression:
        current_expression = expression
        stop_event.set()  # 通知播放线程立即切换
        return {"message": f"Switching to '{expression}' expression"}
    else:
        return {"message": f"Expression '{expression}' is already playing"}


@app.post("/reset")
async def reset_expression():
    """
    重置为默认表情
    """
    #global current_video, stop_event
    global current_expression,stop_event
    #current_video = expressions[DEFAULT_EXPRESSION]
    current_expression = DEFAULT_EXPRESSION
    stop_event.set()  # 通知播放线程切换到默认视频
    return {"message": f"Reset to default expression '{DEFAULT_EXPRESSION}'"}


@app.post("/recording/{recording}")
async def set_recording_overlay(recording: str):
    """
    设置录像提示叠加层，不切换表情视频。
    """
    global recording_overlay
    value = str(recording).strip().lower() in {"1", "true", "yes", "on"}
    with state_lock:
        recording_overlay = value
    return {"recording": recording_overlay}


@app.post("/status")
async def set_status_overlay(payload: dict = Body(default=None)):
    """
    设置机器人状态文字叠加层，不切换表情视频。
    """
    global status_overlay_text, status_overlay_visible
    payload = payload or {}
    text = str(payload.get("status", payload.get("text", "")) or "").strip()
    visible = payload.get("visible", True)
    with state_lock:
        status_overlay_text = text
        status_overlay_visible = bool(visible) and bool(text)
    return {"status": status_overlay_text, "visible": status_overlay_visible}


@app.post("/heard")
async def set_heard_overlay(payload: dict = Body(default=None)):
    """
    设置识别到的用户说话内容叠加层，不切换表情视频。
    """
    global heard_overlay_text, heard_overlay_visible
    payload = payload or {}
    text = str(payload.get("text", payload.get("heard", "")) or "").strip()
    visible = payload.get("visible", True)
    with state_lock:
        heard_overlay_text = text
        heard_overlay_visible = bool(visible) and bool(text)
    return {"text": heard_overlay_text, "visible": heard_overlay_visible}


def run_api_server():
    """
    运行 FastAPI 服务器，使用 uvicorn 启动
    """
    host = os.getenv("EXPRESSION_SERVER_HOST", "0.0.0.0")
    port = int(os.getenv("EXPRESSION_SERVER_PORT", "8001"))
    uvicorn.run(app, host=host, port=port)


def _get_status_font(size: int):
    global status_font
    if ImageFont is None:
        return None

    font_path = os.getenv(
        "EXPRESSION_STATUS_FONT",
        os.path.join(VIDEO_BASE_PATH, "SimHei.ttf"),
    )
    cache_key = (font_path, size)
    if status_font and status_font[0] == cache_key:
        return status_font[1]

    try:
        font = ImageFont.truetype(font_path, size)
    except Exception:
        try:
            font = ImageFont.load_default()
        except Exception:
            return None

    status_font = (cache_key, font)
    return font


def draw_status_overlay(frame):
    """
    在当前表情帧上方叠加机器人状态文字。
    """
    with state_lock:
        enabled = status_overlay_visible
        text = status_overlay_text
    if not enabled or not text:
        return frame

    height, width = frame.shape[:2]
    margin = max(18, int(width * 0.025))

    if Image is None or ImageDraw is None:
        font_scale = max(0.8, min(width, height) / 650.0)
        thickness = max(2, int(font_scale * 2))
        text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        x = max(margin, (width - text_size[0]) // 2)
        y = margin + text_size[1]
        cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
        return frame

    font_size = max(28, int(min(width, height) * 0.06))
    font = _get_status_font(font_size)
    if font is None:
        return frame

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(frame_rgb).convert("RGBA")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    pad_x = max(24, int(font_size * 0.75))
    pad_y = max(14, int(font_size * 0.35))
    box_w = min(width - margin * 2, text_w + pad_x * 2)
    box_h = text_h + pad_y * 2
    box_x = (width - box_w) // 2
    box_y = margin
    radius = max(10, int(box_h * 0.25))

    draw.rounded_rectangle(
        (box_x, box_y, box_x + box_w, box_y + box_h),
        radius=radius,
        fill=(0, 0, 0, 150),
        outline=(255, 255, 255, 110),
        width=2,
    )
    text_x = box_x + (box_w - text_w) // 2
    text_y = box_y + (box_h - text_h) // 2 - bbox[1]
    draw.text((text_x, text_y), text, font=font, fill=(255, 255, 255, 255))

    combined = Image.alpha_composite(image, overlay).convert("RGB")
    return cv2.cvtColor(np.array(combined), cv2.COLOR_RGB2BGR)


def _wrap_text_for_width(draw, text: str, font, max_width: int):
    lines = []
    current = ""
    for ch in text:
        candidate = current + ch
        bbox = draw.textbbox((0, 0), candidate, font=font)
        if bbox[2] - bbox[0] <= max_width or not current:
            current = candidate
            continue
        lines.append(current)
        current = ch
    if current:
        lines.append(current)
    return lines[:3]


def draw_heard_overlay(frame):
    """
    在当前表情帧下方叠加 ASR 识别到的用户说话内容。
    """
    with state_lock:
        enabled = heard_overlay_visible
        text = heard_overlay_text
    if not enabled or not text:
        return frame

    height, width = frame.shape[:2]
    margin = max(18, int(width * 0.025))

    if Image is None or ImageDraw is None:
        font_scale = max(0.65, min(width, height) / 850.0)
        thickness = max(2, int(font_scale * 2))
        text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        x = margin
        y = height - margin
        cv2.putText(frame, text[:80], (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
        return frame

    font_size = max(22, int(min(width, height) * 0.042))
    font = _get_status_font(font_size)
    if font is None:
        return frame

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(frame_rgb).convert("RGBA")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    pad_x = max(22, int(font_size * 0.75))
    pad_y = max(14, int(font_size * 0.42))
    box_w = width - margin * 2
    max_text_w = box_w - pad_x * 2
    lines = _wrap_text_for_width(draw, text, font, max_text_w)
    if not lines:
        return frame

    line_heights = []
    line_widths = []
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        line_widths.append(bbox[2] - bbox[0])
        line_heights.append(bbox[3] - bbox[1])
    line_gap = max(6, int(font_size * 0.22))
    text_h = sum(line_heights) + line_gap * (len(lines) - 1)
    box_h = text_h + pad_y * 2
    box_x = margin
    box_y = height - margin - box_h
    radius = max(10, int(box_h * 0.18))

    draw.rounded_rectangle(
        (box_x, box_y, box_x + box_w, box_y + box_h),
        radius=radius,
        fill=(0, 0, 0, 165),
        outline=(255, 255, 255, 95),
        width=2,
    )

    y = box_y + pad_y
    for idx, line in enumerate(lines):
        bbox = draw.textbbox((0, 0), line, font=font)
        draw.text((box_x + pad_x, y - bbox[1]), line, font=font, fill=(255, 255, 255, 255))
        y += line_heights[idx] + line_gap

    combined = Image.alpha_composite(image, overlay).convert("RGB")
    return cv2.cvtColor(np.array(combined), cv2.COLOR_RGB2BGR)


def draw_recording_overlay(frame):
    """
    在当前表情帧右上角叠加红点和 REC。
    """
    with state_lock:
        enabled = recording_overlay
    if not enabled:
        return frame

    height, width = frame.shape[:2]
    margin = max(16, int(width * 0.02))
    radius = max(8, int(min(width, height) * 0.018))
    font_scale = max(0.7, min(width, height) / 720.0)
    thickness = max(2, int(font_scale * 2))
    text = "REC"
    text_size, baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    text_w, text_h = text_size

    dot_x = width - margin - text_w - radius * 3
    dot_y = margin + max(radius, text_h // 2)
    text_x = dot_x + radius * 2
    text_y = dot_y + text_h // 2

    cv2.circle(frame, (dot_x, dot_y), radius, (0, 0, 255), -1)
    cv2.putText(
        frame,
        text,
        (text_x, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (0, 0, 255),
        thickness,
        cv2.LINE_AA,
    )
    return frame


def _resolve_expression_path(name: str) -> str | None:
    if name in expressions:
        return expressions[name]
    if name in UPLOADED_EXPRESSIONS:
        return UPLOADED_EXPRESSIONS[name]
    return None


IMAGE_DISPLAY_SECONDS = float(os.getenv("EXPRESSION_IMAGE_SECONDS", "5.0"))


def play_image(image_path: str):
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Failed to load image: {image_path}")
        return

    cv2.namedWindow("Expression", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Expression", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    width = int(os.getenv("EXPRESSION_VIDEO_WIDTH", "1024"))
    height = int(os.getenv("EXPRESSION_VIDEO_HEIGHT", "600"))
    frame = cv2.resize(frame, (width, height))
    frame = draw_status_overlay(frame)
    frame = draw_heard_overlay(frame)
    frame = draw_recording_overlay(frame)

    deadline = time.monotonic() + IMAGE_DISPLAY_SECONDS
    print(f"Displaying image: {image_path} ({IMAGE_DISPLAY_SECONDS}s)")

    while time.monotonic() < deadline and not stop_event.is_set():
        cv2.imshow("Expression", frame)
        if cv2.waitKey(40) & 0xFF == ord("q"):
            break


def play_video(video_path):
    """
    根据传入的视频路径使用 cv2 播放视频，可循环播放。
    当 stop_event 被置位，结束当前视频的播放，返回函数以便切换到新视频。
    """
    if video_path is None:
        print("Terminating expression display")
        frame_delay = float(os.getenv("EXPRESSION_FRAME_DELAY", "0.04"))
        time.sleep(frame_delay)
        try:
            cv2.destroyWindow("Expression")
        except Exception as e:
            print(f"Error destroying window: {e}")
    else:
        print(f"Playing video: {video_path}")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Failed to open video: {video_path}")
            return

        # 创建全屏窗口
        cv2.namedWindow("Expression", cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty(
            "Expression", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN
        )

        while (not stop_event.is_set()) and cap.isOpened():
            ret, frame = cap.read()
            frame_delay = float(os.getenv("EXPRESSION_FRAME_DELAY", "0.04"))
            time.sleep(frame_delay)
            if not ret:
                # 视频播放完毕后，从头开始重放
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            width = int(os.getenv("EXPRESSION_VIDEO_WIDTH", "1024"))
            height = int(os.getenv("EXPRESSION_VIDEO_HEIGHT", "600"))
            frame = cv2.resize(frame, (width, height))
            frame = draw_status_overlay(frame)
            frame = draw_heard_overlay(frame)
            frame = draw_recording_overlay(frame)
            cv2.imshow("Expression", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()


def play_video_continuously():
    """
    主线程循环，持续播放视频。
    每次调用 play_video 完成后，检查是否收到切换请求（stop_event 被置位），
    如果是则根据全局变量 current_video 播放新的视频。
    """
    #global current_video, stop_event
    global current_expression, stop_event, expressions

    while True:
        stop_event.clear()
        path = _resolve_expression_path(current_expression)
        if path is None:
            print(f"Expression '{current_expression}' not found, resetting to default")
            current_expression = DEFAULT_EXPRESSION
            continue
        if is_image_path(path):
            play_image(path)
        else:
            play_video(path)
        if stop_event.is_set():
            print("Switching video...")
            stop_event.clear()


if __name__ == "__main__":
    scan_uploaded_dir(VIDEO_BASE_PATH)
    start_watcher(VIDEO_BASE_PATH)

    # 开启 FastAPI 服务线程（守护线程）
    api_thread = threading.Thread(target=run_api_server, daemon=True)
    api_thread.start()
    # 主线程运行 OpenCV 视频播放（默认及切换表情）
    play_video_continuously()

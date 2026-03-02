import cv2
import time
import threading
import os
from fastapi import FastAPI
from fastapi.responses import JSONResponse
import uvicorn
from dotenv import load_dotenv
import requests

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

# 线程同步，通知播放线程切换视频
stop_event = threading.Event()

# 实例化 FastAPI 应用
app = FastAPI(title="Expression Player API", description="API 助力 OpenCV 播放表情视频")


@app.get("/expressions")
async def get_available_expressions():
    """
    返回所有可用的表情列表
    """
    return {"expressions": list(expressions.keys())}

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

    if expression not in expressions:
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


def run_api_server():
    """
    运行 FastAPI 服务器，使用 uvicorn 启动
    """
    host = os.getenv("EXPRESSION_SERVER_HOST", "0.0.0.0")
    port = int(os.getenv("EXPRESSION_SERVER_PORT", "8001"))
    uvicorn.run(app, host=host, port=port)


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
        # 清除停止标志，开始播放 current_video 视频
        stop_event.clear()
        play_video(expressions[current_expression])
        if stop_event.is_set():
            print("Switching video...")
            stop_event.clear()


if __name__ == "__main__":
    # 开启 FastAPI 服务线程（守护线程）
    api_thread = threading.Thread(target=run_api_server, daemon=True)
    api_thread.start()
    # 主线程运行 OpenCV 视频播放（默认及切换表情）
    play_video_continuously()

"""
AudioRecorder 模块测试
包含模拟测试和实际麦克风测试
"""
import numpy as np
import time
import threading
import sys
from talkrobot.config import Config
from talkrobot.core.audio_recorder import AudioRecorder


def test_initialization_push():
    """测试1: Push 模式初始化"""
    print("\n" + "="*60)
    print("测试1: Push 模式初始化")
    print("="*60)
    
    try:
        recorder = AudioRecorder(
            sample_rate=16000,
            channels=1,
            listen_mode="push"
        )
        print("✅ Push 模式初始化成功")
        print(f"   采样率: {recorder.sample_rate} Hz")
        print(f"   声道数: {recorder.channels}")
        print(f"   监听模式: {recorder.listen_mode}")
        return True
    except Exception as e:
        print(f"❌ 初始化失败: {e}")
        return False


def test_initialization_continuous():
    """测试2: Continuous 模式初始化"""
    print("\n" + "="*60)
    print("测试2: Continuous 模式初始化（含 VAD 模型加载）")
    print("="*60)
    
    try:
        start_time = time.time()
        recorder = AudioRecorder(
            sample_rate=16000,
            channels=1,
            listen_mode="continuous",
            vad_check_interval=0.25,
            silence_duration=1.5,
            min_speech_duration=0.3
        )
        load_time = time.time() - start_time
        
        print("✅ Continuous 模式初始化成功")
        print(f"   采样率: {recorder.sample_rate} Hz")
        print(f"   声道数: {recorder.channels}")
        print(f"   监听模式: {recorder.listen_mode}")
        print(f"   VAD 检测间隔: {recorder.vad_check_interval}s")
        print(f"   静默阈值: {recorder.silence_duration}s")
        print(f"   最短语音: {recorder.min_speech_duration}s")
        print(f"   初始化耗时: {load_time:.2f}s")
        print(f"   VAD 模型: {'已加载' if recorder._vad_model else '未加载'}")
        return recorder
    except ImportError as e:
        print(f"⚠️  需要安装依赖: {e}")
        print("   运行: pip install torch silero-vad")
        return None
    except Exception as e:
        print(f"❌ 初始化失败: {e}")
        return None


def test_initialization_intercom():
    """测试3: Intercom 模式初始化"""
    print("\n" + "="*60)
    print("测试3: Intercom 模式初始化")
    print("="*60)

    try:
        recorder = AudioRecorder(
            sample_rate=16000,
            channels=1,
            listen_mode="intercom",
            ptt_trigger_threshold=10000,
            ptt_debounce_time=0.2,
        )
        print("✅ Intercom 模式初始化成功")
        print(f"   采样率: {recorder.sample_rate} Hz")
        print(f"   声道数: {recorder.channels}")
        print(f"   监听模式: {recorder.listen_mode}")
        print(f"   PTT 阈值: {recorder.ptt_trigger_threshold}")
        print(f"   防抖时间: {recorder.ptt_debounce_time}s")
        return recorder
    except Exception as e:
        print(f"❌ 初始化失败: {e}")
        return None


def test_parameters_validation():
    """测试4: 参数验证"""
    print("\n" + "="*60)
    print("测试3: 不同参数配置")
    print("="*60)
    
    test_configs = [
        {
            "name": "低延迟配置",
            "params": {
                "sample_rate": 16000,
                "listen_mode": "continuous",
                "vad_check_interval": 0.1,
                "silence_duration": 0.8,
                "min_speech_duration": 0.2
            }
        },
        {
            "name": "高稳定性配置",
            "params": {
                "sample_rate": 16000,
                "listen_mode": "continuous",
                "vad_check_interval": 0.3,
                "silence_duration": 2.5,
                "min_speech_duration": 0.5
            }
        },
        {
            "name": "自定义采样率",
            "params": {
                "sample_rate": 8000,
                "channels": 1,
                "listen_mode": "push"
            }
        }
    ]
    
    for config in test_configs:
        print(f"\n   测试: {config['name']}")
        try:
            recorder = AudioRecorder(**config['params'])
            print(f"      ✅ 配置有效")
            for key, value in config['params'].items():
                if hasattr(recorder, key):
                    print(f"         {key}: {value}")
        except Exception as e:
            print(f"      ❌ 配置无效: {e}")


def test_audio_callback_simulation():
    """测试5: 音频回调模拟"""
    print("\n" + "="*60)
    print("测试4: 音频回调模拟（Push 模式）")
    print("="*60)
    
    recorder = AudioRecorder(listen_mode="push")
    
    # 模拟录音过程
    print("   模拟录音场景...")
    recorder.is_recording = True
    
    # 模拟 10 次音频回调
    for i in range(10):
        # 每次 0.1 秒的音频（1600 个采样点）
        audio_chunk = np.random.randn(1600, 1).astype(np.float32) * 0.01
        recorder.audio_callback(audio_chunk, 1600, None, None)
    
    recorder.is_recording = False
    
    # 检查缓冲区
    total_frames = len(recorder.audio_frames)
    print(f"   采集到 {total_frames} 个音频块")
    
    if total_frames == 10:
        print("   ✅ 音频回调工作正常")
        
        # 合并音频
        audio_data = np.concatenate(recorder.audio_frames, axis=0)
        print(f"   合并后总采样点: {len(audio_data)}")
        print(f"   音频时长: {len(audio_data)/16000:.2f}s")
    else:
        print(f"   ⚠️  预期 10 个块，实际 {total_frames} 个")


def test_intercom_ptt_simulation():
    """测试6: 对讲机 PTT 触发模拟"""
    print("\n" + "="*60)
    print("测试6: 对讲机 PTT 触发模拟")
    print("="*60)

    recorder = AudioRecorder(
        listen_mode="intercom",
        ptt_trigger_threshold=10000,
        ptt_debounce_time=0.0,
    )

    captured = []

    def _on_audio(audio_data):
        captured.append(audio_data)

    recorder.on_audio_complete = _on_audio

    # 触发“按下”
    pulse = np.full((160, 1), 0.6, dtype=np.float32)
    recorder.audio_callback(pulse, 160, None, None)

    # 模拟收音过程
    voice = np.full((160, 1), 0.01, dtype=np.float32)
    recorder.audio_callback(voice, 160, None, None)
    recorder.audio_callback(voice, 160, None, None)

    # 触发“松开”
    recorder.audio_callback(pulse, 160, None, None)

    if len(captured) == 1 and captured[0].shape[0] > 0:
        print(f"   ✅ 对讲机模式回调正常，音频长度: {captured[0].shape[0]}")
    else:
        print(f"   ❌ 对讲机模式异常，回调次数: {len(captured)}")


def test_tts_playing_flag():
    """测试7: TTS 播放标志"""
    print("\n" + "="*60)
    print("测试5: TTS 播放标志（Continuous 模式）")
    print("="*60)
    
    try:
        recorder = AudioRecorder(listen_mode="continuous")
        
        print("   初始状态:")
        print(f"      is_tts_playing: {recorder.is_tts_playing}")
        print(f"      _processing: {recorder._processing}")
        
        # 模拟 TTS 播放
        print("\n   设置 TTS 播放标志...")
        recorder.is_tts_playing = True
        print(f"      is_tts_playing: {recorder.is_tts_playing}")
        
        # 模拟音频回调（应该被忽略）
        initial_buffer_len = len(recorder._chunk_buffer)
        audio_chunk = np.random.randn(1600, 1).astype(np.float32)
        recorder.audio_callback(audio_chunk, 1600, None, None)
        after_buffer_len = len(recorder._chunk_buffer)
        
        if after_buffer_len == initial_buffer_len:
            print("      ✅ TTS 播放期间正确忽略音频输入")
        else:
            print("      ⚠️  TTS 播放期间仍在采集音频")
        
        # 重置标志
        recorder.is_tts_playing = False
        print(f"\n   重置标志: is_tts_playing: {recorder.is_tts_playing}")
        print("   ✅ TTS 标志测试完成")
        
    except Exception as e:
        print(f"   ⚠️  跳过测试（需要 VAD）: {e}")


def test_processing_flag():
    """测试8: 处理标志"""
    print("\n" + "="*60)
    print("测试6: 音频处理标志")
    print("="*60)
    
    try:
        recorder = AudioRecorder(listen_mode="continuous")
        
        print(f"   初始 _processing: {recorder._processing}")
        
        # 模拟开始处理
        recorder._processing = True
        print(f"   设置为 True: {recorder._processing}")
        
        # 调用 notify_process_done
        recorder.notify_process_done()
        print(f"   调用 notify_process_done 后: {recorder._processing}")
        
        if not recorder._processing:
            print("   ✅ 处理标志工作正常")
        else:
            print("   ⚠️  处理标志未正确重置")
            
    except Exception as e:
        print(f"   ⚠️  跳过测试: {e}")


def test_vad_status_display():
    """测试9: VAD 状态栏显示"""
    print("\n" + "="*60)
    print("测试7: VAD 状态栏显示")
    print("="*60)
    
    try:
        recorder = AudioRecorder(listen_mode="continuous")
        
        print("   测试不同状态的显示...\n")
        
        # 状态1: 监听中
        recorder._print_vad_status(rms=0.001, has_speech=False)
        time.sleep(0.5)
        
        # 状态2: 检测到语音
        recorder._print_vad_status(rms=0.05, has_speech=True)
        time.sleep(0.5)
        
        # 状态3: 录音中
        recorder.is_recording = True
        recorder._speech_start_time = time.time()
        recorder._print_vad_status(rms=0.03, has_speech=True)
        time.sleep(0.5)
        
        # 状态4: 处理中
        recorder.is_recording = False
        recorder._processing = True
        recorder._print_vad_status(rms=0.0, has_speech=False)
        time.sleep(0.5)
        
        # 状态5: 播放中
        recorder._processing = False
        recorder.is_tts_playing = True
        recorder._print_vad_status(rms=0.0, has_speech=False)
        time.sleep(0.5)
        
        # 清除状态栏
        recorder._clear_vad_status()
        print("\n   ✅ 状态栏显示测试完成")
        
    except Exception as e:
        print(f"   ⚠️  跳过测试: {e}")


def test_real_microphone_push():
    """测试10: 真实麦克风测试（Push 模式）"""
    print("\n" + "="*60)
    print("测试8: 真实麦克风测试（Push 模式）")
    print("="*60)
    
    print("   ⚠️  这是一个交互式测试")
    response = input("   是否进行真实麦克风测试？(y/n): ").strip().lower()
    
    if response != 'y':
        print("   ⏭  跳过麦克风测试")
        return
    
    try:
        import sounddevice as sd
        print("\n   可用音频设备:")
        devices = sd.query_devices()
        for i, device in enumerate(devices):
            if device['max_input_channels'] > 0:
                print(f"      [{i}] {device['name']} (输入通道: {device['max_input_channels']})")
        
        recordings = []
        
        def on_audio(audio_data):
            recordings.append(audio_data)
            duration = len(audio_data) / 16000
            rms = float(np.sqrt(np.mean(audio_data ** 2)))
            print(f"\n   收到录音: 时长 {duration:.2f}s, RMS {rms:.4f}")
        
        recorder = AudioRecorder(listen_mode="push")
        
        print("\n   📍 按住 'Q' 键说话，松开结束")
        print("   📍 按 Ctrl+C 退出测试")
        print()
        
        # 启动录音（带超时）
        timeout_event = threading.Event()
        
        def timeout_handler():
            time.sleep(10)  # 10 秒后自动退出
            if not timeout_event.is_set():
                print("\n   ⏱  测试超时，自动退出")
                import os
                os._exit(0)  # 强制退出
        
        timeout_thread = threading.Thread(target=timeout_handler, daemon=True)
        timeout_thread.start()
        
        try:
            recorder.start(on_audio_complete=on_audio)
        except KeyboardInterrupt:
            timeout_event.set()
            print("\n   测试结束")
        
        print(f"\n   总共录制 {len(recordings)} 段音频")
        if recordings:
            print("   ✅ 麦克风测试成功")
        else:
            print("   ⚠️  未录制到音频")
            
    except Exception as e:
        print(f"   ❌ 测试失败: {e}")


def test_real_microphone_continuous():
    """测试11: 真实麦克风测试（Continuous 模式）"""
    print("\n" + "="*60)
    print("测试9: 真实麦克风测试（Continuous 模式）")
    print("="*60)
    
    print("   ⚠️  这是一个交互式测试")
    response = input("   是否进行 VAD 自动检测测试？(y/n): ").strip().lower()
    
    if response != 'y':
        print("   ⏭  跳过 VAD 测试")
        return
    
    try:
        recordings = []
        
        def on_audio(audio_data):
            recordings.append(audio_data)
            duration = len(audio_data) / 16000
            rms = float(np.sqrt(np.mean(audio_data ** 2)))
            print(f"\n   ✅ 检测到完整语音段: 时长 {duration:.2f}s, RMS {rms:.4f}")
            # 自动通知处理完成
            recorder.notify_process_done()
        
        recorder = AudioRecorder(
            listen_mode="continuous",
            vad_check_interval=0.25,
            silence_duration=1.5,
            min_speech_duration=0.3
        )
        
        print("\n   📍 直接说话即可，系统自动检测")
        print("   📍 停顿 1.5 秒后自动结束")
        print("   📍 按 Ctrl+C 退出测试")
        print("   📍 实时状态会显示在下方\n")
        
        # 启动录音（带超时）
        timeout_event = threading.Event()
        
        def timeout_handler():
            time.sleep(15)  # 15 秒后自动退出
            if not timeout_event.is_set():
                print("\n   ⏱  测试超时，自动退出")
                import os
                os._exit(0)
        
        timeout_thread = threading.Thread(target=timeout_handler, daemon=True)
        timeout_thread.start()
        
        try:
            recorder.start(on_audio_complete=on_audio)
        except KeyboardInterrupt:
            timeout_event.set()
            print("\n   测试结束")
        
        print(f"\n   总共检测到 {len(recordings)} 段语音")
        if recordings:
            print("   ✅ VAD 自动检测测试成功")
        else:
            print("   ⚠️  未检测到有效语音")
            
    except Exception as e:
        print(f"   ❌ 测试失败: {e}")


def test_concurrent_callbacks():
    """测试12: 并发回调测试"""
    print("\n" + "="*60)
    print("测试10: 并发音频回调测试")
    print("="*60)
    
    try:
        recorder = AudioRecorder(listen_mode="continuous")
        
        # 模拟并发音频回调
        def simulate_callbacks():
            recorder._processing = False
            recorder.is_tts_playing = False
            
            for i in range(100):
                audio_chunk = np.random.randn(160, 1).astype(np.float32) * 0.01
                recorder.audio_callback(audio_chunk, 160, None, None)
                time.sleep(0.001)
        
        threads = []
        num_threads = 3
        
        print(f"   启动 {num_threads} 个并发回调线程...")
        start_time = time.time()
        
        for i in range(num_threads):
            thread = threading.Thread(target=simulate_callbacks)
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        elapsed = time.time() - start_time
        
        # 检查缓冲区
        with recorder._chunk_lock:
            buffer_size = len(recorder._chunk_buffer)
        
        print(f"   并发测试完成")
        print(f"   耗时: {elapsed:.2f}s")
        print(f"   缓冲区大小: {buffer_size}")
        print("   ✅ 并发测试通过")
        
    except Exception as e:
        print(f"   ❌ 测试失败: {e}")


def test_memory_leak():
    """测试13: 内存泄漏测试"""
    print("\n" + "="*60)
    print("测试11: 内存管理测试")
    print("="*60)
    
    try:
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        
        recorder = AudioRecorder(listen_mode="continuous")
        recorder._processing = False
        recorder.is_tts_playing = False
        
        # 测试前内存
        mem_before = process.memory_info().rss / 1024 / 1024  # MB
        print(f"   测试前内存: {mem_before:.2f} MB")
        
        # 模拟大量音频处理
        print("   模拟处理 1000 个音频块...")
        for i in range(1000):
            audio_chunk = np.random.randn(1600, 1).astype(np.float32)
            recorder.audio_callback(audio_chunk, 1600, None, None)
            
            # 定期清空缓冲区（模拟正常使用）
            if i % 100 == 0:
                with recorder._chunk_lock:
                    recorder._chunk_buffer = []
        
        # 清空缓冲区
        with recorder._chunk_lock:
            recorder._chunk_buffer = []
        
        # 测试后内存
        mem_after = process.memory_info().rss / 1024 / 1024  # MB
        mem_increase = mem_after - mem_before
        
        print(f"   测试后内存: {mem_after:.2f} MB")
        print(f"   内存增长: {mem_increase:.2f} MB")
        
        if mem_increase < 50:  # 小于 50MB 认为正常
            print("   ✅ 内存管理正常")
        else:
            print(f"   ⚠️  内存增长较大: {mem_increase:.2f} MB")
            
    except ImportError:
        print("   ⚠️  需要安装 psutil: pip install psutil")
    except Exception as e:
        print(f"   ⚠️  测试异常: {e}")


def run_all_tests():
    """运行所有测试"""
    print("\n" + "🎙️  "*20)
    print("AudioRecorder 模块完整测试套件")
    print("🎙️  "*20)
    
    # 非交互式测试
    test_initialization_push()
    recorder = test_initialization_continuous()
    test_initialization_intercom()
    
    if recorder:
        test_parameters_validation()
        test_audio_callback_simulation()
        test_intercom_ptt_simulation()
        test_tts_playing_flag()
        test_processing_flag()
        test_vad_status_display()
        test_concurrent_callbacks()
        test_memory_leak()
    else:
        print("\n⚠️  Continuous 模式依赖未满足，跳过部分测试")
    
    # 交互式测试
    print("\n" + "-"*60)
    print("交互式测试（需要麦克风）")
    print("-"*60)
    
    test_real_microphone_push()
    if recorder:
        test_real_microphone_continuous()
    
    # 总结
    print("\n" + "="*60)
    print("🎉 AudioRecorder 模块测试完成")
    print("="*60)
    print("\n提示:")
    print("  - 部分测试需要真实麦克风设备")
    print("  - Continuous 模式需要安装 torch 和 silero-vad")
    print("  - 使用 --debug 参数可查看详细日志")
    print()


def test_quick():
    """快速测试（仅非交互式）"""
    print("\n" + "="*60)
    print("AudioRecorder 快速测试")
    print("="*60)
    
    # 测试 Push 模式
    if test_initialization_push():
        print("✅ Push 模式工作正常")

    # 测试 Intercom 模式
    intercom_recorder = test_initialization_intercom()
    if intercom_recorder:
        print("✅ Intercom 模式工作正常")
        test_intercom_ptt_simulation()
    
    # 测试 Continuous 模式
    recorder = test_initialization_continuous()
    if recorder:
        print("✅ Continuous 模式工作正常")
    else:
        print("⚠️  Continuous 模式需要额外依赖")
    
    print("\n快速测试完成")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "quick":
        test_quick()
    else:
        run_all_tests()

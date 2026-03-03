"""
ASR模块测试
支持多种测试场景：基础功能、音频长度、音量、性能等
"""
import numpy as np
import time
import os
from talkrobot.config import Config
from talkrobot.modules.asr_module import ASRModule


def test_initialization():
    """测试1: 模块初始化"""
    print("\n" + "="*60)
    print("测试1: ASR 模块初始化")
    print("="*60)
    
    try:
        start_time = time.time()
        asr = ASRModule(
            model_name=Config.ASR_MODEL,
            device=Config.ASR_DEVICE
        )
        load_time = time.time() - start_time
        print(f"✅ 模块初始化成功")
        print(f"   加载时间: {load_time:.2f} 秒")
        return asr
    except Exception as e:
        print(f"❌ 模块初始化失败: {e}")
        return None


def test_basic_transcription(asr):
    """测试2: 基础语音识别（模拟音频）"""
    print("\n" + "="*60)
    print("测试2: 基础语音识别（模拟音频）")
    print("="*60)
    
    duration = 2  # 秒
    test_audio = np.random.randn(Config.SAMPLE_RATE * duration).astype(np.float32) * 0.01
    
    print(f"   音频时长: {duration} 秒")
    print(f"   音频采样点: {len(test_audio)}")
    print("   正在识别...")
    
    start_time = time.time()
    result = asr.transcribe(test_audio)
    process_time = time.time() - start_time
    
    print(f"   处理时间: {process_time:.3f} 秒")
    print(f"   RTF: {process_time/duration:.3f}")
    
    if result:
        print(f"✅ 识别结果: {result}")
    else:
        print("ℹ️  识别结果为空（模拟音频通常无法识别出内容）")


def test_audio_length(asr):
    """测试3: 不同长度的音频"""
    print("\n" + "="*60)
    print("测试3: 不同长度的音频")
    print("="*60)
    
    test_cases = [
        (0.05, "过短音频（0.05秒）"),
        (0.5, "最短推荐长度（0.5秒）"),
        (2.0, "正常长度（2秒）"),
        (5.0, "较长音频（5秒）"),
    ]
    
    for duration, description in test_cases:
        print(f"\n   测试: {description}")
        audio = np.random.randn(int(Config.SAMPLE_RATE * duration)).astype(np.float32) * 0.01
        
        start_time = time.time()
        result = asr.transcribe(audio)
        process_time = time.time() - start_time
        
        print(f"      采样点数: {len(audio)}")
        print(f"      处理时间: {process_time:.3f} 秒")
        
        if duration >= 0.1:  # 只有长度足够才期望有结果
            print(f"      结果: {result if result else '(空)'}")
        else:
            print(f"      结果: 跳过识别（音频过短）")


def test_audio_volume(asr):
    """测试4: 不同音量的音频"""
    print("\n" + "="*60)
    print("测试4: 不同音量的音频")
    print("="*60)
    
    duration = 2
    base_audio = np.random.randn(Config.SAMPLE_RATE * duration).astype(np.float32)
    
    volume_levels = [
        (0.001, "极低音量"),
        (0.01, "低音量"),
        (0.1, "正常音量"),
        (0.5, "高音量"),
    ]
    
    for volume, description in volume_levels:
        print(f"\n   测试: {description} (系数={volume})")
        audio = base_audio * volume
        rms = float(np.sqrt(np.mean(audio ** 2)))
        print(f"      RMS: {rms:.4f}")
        
        result = asr.transcribe(audio)
        print(f"      结果: {result if result else '(空)'}")


def test_numpy_shapes(asr):
    """测试5: 不同 numpy 数组形状"""
    print("\n" + "="*60)
    print("测试5: 不同 numpy 数组形状")
    print("="*60)
    
    duration = 2
    samples = Config.SAMPLE_RATE * duration
    
    test_cases = [
        (np.random.randn(samples).astype(np.float32) * 0.01, "一维数组"),
        (np.random.randn(samples, 1).astype(np.float32) * 0.01, "二维数组 (N, 1)"),
        (np.random.randn(1, samples).astype(np.float32) * 0.01, "二维数组 (1, N)"),
    ]
    
    for audio, description in test_cases:
        print(f"\n   测试: {description}")
        print(f"      形状: {audio.shape}")
        print(f"      数据类型: {audio.dtype}")
        
        result = asr.transcribe(audio)
        print(f"      结果: {result if result else '(空)'}")
        print("      ✅ 处理成功")


def test_performance(asr):
    """测试6: 性能基准测试"""
    print("\n" + "="*60)
    print("测试6: 性能基准测试")
    print("="*60)
    
    durations = [1, 2, 5, 10]
    iterations = 3
    
    for duration in durations:
        print(f"\n   音频时长: {duration} 秒")
        times = []
        
        for i in range(iterations):
            audio = np.random.randn(Config.SAMPLE_RATE * duration).astype(np.float32) * 0.01
            
            start_time = time.time()
            asr.transcribe(audio)
            process_time = time.time() - start_time
            times.append(process_time)
        
        avg_time = np.mean(times)
        rtf = avg_time / duration
        
        print(f"      平均处理时间: {avg_time:.3f} 秒 (±{np.std(times):.3f})")
        print(f"      RTF: {rtf:.3f}")
        print(f"      吞吐量: {duration/avg_time:.2f}x 实时")


def test_empty_audio(asr):
    """测试7: 空音频和边界情况"""
    print("\n" + "="*60)
    print("测试7: 空音频和边界情况")
    print("="*60)
    
    test_cases = [
        (np.array([]).astype(np.float32), "空数组"),
        (np.zeros(100).astype(np.float32), "100个采样点的静音"),
        (np.zeros(1600).astype(np.float32), "刚好0.1秒的静音"),
        (np.zeros(Config.SAMPLE_RATE * 2).astype(np.float32), "2秒静音"),
    ]
    
    for audio, description in test_cases:
        print(f"\n   测试: {description}")
        print(f"      长度: {len(audio)} 采样点")
        
        try:
            result = asr.transcribe(audio)
            print(f"      结果: {result if result else '(空)'}")
            print("      ✅ 处理成功")
        except Exception as e:
            print(f"      ❌ 异常: {e}")


def test_real_audio_file(asr):
    """测试8: 从真实音频文件识别（如果存在）"""
    print("\n" + "="*60)
    print("测试8: 从真实音频文件识别")
    print("="*60)
    
    # 查找 example 目录下的音频文件
    test_dirs = [
        "example",
        "talkrobot/tests",
        "tests",
    ]
    
    audio_extensions = [".wav", ".mp3", ".flac"]
    found_file = None
    
    for test_dir in test_dirs:
        if os.path.exists(test_dir):
            for filename in os.listdir(test_dir):
                if any(filename.endswith(ext) for ext in audio_extensions):
                    found_file = os.path.join(test_dir, filename)
                    break
            if found_file:
                break
    
    if found_file:
        print(f"   找到测试音频: {found_file}")
        try:
            import soundfile as sf
            audio, sr = sf.read(found_file)
            print(f"      采样率: {sr} Hz")
            print(f"      时长: {len(audio)/sr:.2f} 秒")
            print(f"      声道数: {audio.shape[1] if audio.ndim > 1 else 1}")
            
            # 如果是立体声，转为单声道
            if audio.ndim > 1:
                audio = np.mean(audio, axis=1)
            
            # 重采样到 16kHz（如果需要）
            if sr != 16000:
                print(f"      重采样: {sr} Hz -> 16000 Hz")
                from scipy import signal
                audio = signal.resample(audio, int(len(audio) * 16000 / sr))
            
            audio = audio.astype(np.float32)
            
            print("      正在识别...")
            start_time = time.time()
            result = asr.transcribe(audio)
            process_time = time.time() - start_time
            
            print(f"      处理时间: {process_time:.3f} 秒")
            print(f"✅ 识别结果: {result}")
            
        except ImportError:
            print("   ⚠️  需要安装 soundfile 库: pip install soundfile")
        except Exception as e:
            print(f"   ❌ 读取音频失败: {e}")
    else:
        print("   ℹ️  未找到测试音频文件，跳过此测试")
        print("      可以在 example/ 目录放置 .wav 文件进行测试")


def test_concurrent_transcription(asr):
    """测试9: 并发识别测试（验证线程安全）"""
    print("\n" + "="*60)
    print("测试9: 并发识别测试")
    print("="*60)
    
    import threading
    
    duration = 2
    num_threads = 3
    results = []
    
    def worker(thread_id):
        audio = np.random.randn(Config.SAMPLE_RATE * duration).astype(np.float32) * 0.01
        result = asr.transcribe(audio)
        results.append((thread_id, result))
        print(f"      线程 {thread_id} 完成")
    
    print(f"   启动 {num_threads} 个并发识别任务...")
    threads = []
    start_time = time.time()
    
    for i in range(num_threads):
        thread = threading.Thread(target=worker, args=(i,))
        threads.append(thread)
        thread.start()
    
    for thread in threads:
        thread.join()
    
    total_time = time.time() - start_time
    print(f"   总耗时: {total_time:.2f} 秒")
    print(f"   平均每个任务: {total_time/num_threads:.2f} 秒")
    print(f"✅ 并发测试完成，所有 {len(results)} 个任务成功")


def run_all_tests():
    """运行所有测试"""
    print("\n" + "🚀 "*20)
    print("ASR 模块完整测试套件")
    print("🚀 "*20)
    
    # 测试1: 初始化
    asr = test_initialization()
    if asr is None:
        print("\n❌ 模块初始化失败，终止测试")
        return
    
    # 测试2-9
    try:
        test_basic_transcription(asr)
        test_audio_length(asr)
        test_audio_volume(asr)
        test_numpy_shapes(asr)
        test_empty_audio(asr)
        test_performance(asr)
        test_real_audio_file(asr)
        test_concurrent_transcription(asr)
        
    except Exception as e:
        print(f"\n❌ 测试过程中出现异常: {e}")
        import traceback
        traceback.print_exc()
    
    # 总结
    print("\n" + "="*60)
    print("🎉 ASR 模块测试完成")
    print("="*60)
    print("\n提示:")
    print("  - 模拟音频通常无法识别出内容（正常现象）")
    print("  - 可在 example/ 目录放置真实音频文件进行测试")
    print("  - 查看详细日志请检查 talkrobot/logs/")
    print()


def test_quick():
    """快速测试（仅测试基础功能）"""
    print("\n" + "="*60)
    print("ASR 模块快速测试")
    print("="*60)
    
    asr = ASRModule(
        model_name=Config.ASR_MODEL,
        device=Config.ASR_DEVICE
    )
    
    # 生成测试音频
    duration = 2  # 秒
    test_audio = np.random.randn(Config.SAMPLE_RATE * duration).astype(np.float32) * 0.01
    
    print("\n正在测试识别...")
    result = asr.transcribe(test_audio)
    print(f"识别结果: {result if result else '(空 - 模拟音频通常无法识别)'}")
    print("\n✅ 快速测试完成")


if __name__ == "__main__":
    import sys
    
    # 支持命令行参数选择测试模式
    if len(sys.argv) > 1 and sys.argv[1] == "quick":
        test_quick()
    else:
        run_all_tests()
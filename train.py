import threading
import time

import cv2
import numpy as np
from android_tool import AndroidTool
from argparses import args
from dqnAgent import DQNAgent
from getReword import GetRewordUtil
from globalInfo import GlobalInfo

from wzry_env import Environment
from onnxRunner import OnnxRunner

# 全局状态
globalInfo = GlobalInfo()

class_names = ['started']
start_check = OnnxRunner('models/start.onnx', classes=class_names)

rewordUtil = GetRewordUtil()
tool = AndroidTool()
tool.show_scrcpy()
# tool.show_action_log()
env = Environment(tool, rewordUtil)

agent = DQNAgent()

# 自动开始新对局的点击坐标序列 (比例坐标)
AUTO_START_CLICKS = [
    (0.515, 0.946),  # 第1步
    (0.400, 0.909),  # 第2步
    (0.395, 0.747),  # 第3步
    (0.797, 0.052),  # 第4步
    (0.653, 0.456),  # 第5步
    (0.055, 0.746),  # 第6步
    (0.924, 0.928),  # 第7步
    (0.043, 0.945),  # 第8步
    (0.917, 0.934),  # 第9步 - 开始对战
]

def auto_start_new_game():
    """对局结束后自动开始新对局"""
    print("-------------------------------自动开始新对局-----------------------------------")
    import subprocess
    screen_width = 1920
    screen_height = 1080
    
    time.sleep(5)  # 开始前等待5秒
    
    for i, (x_ratio, y_ratio) in enumerate(AUTO_START_CLICKS):
        x = int(x_ratio * screen_width)
        y = int(y_ratio * screen_height)
        print(f"点击第 {i+1} 步: ({x}, {y})")
        subprocess.run([f'{tool.scrcpy_dir}/adb', '-s', tool.device_serial, 'shell', 
                       'input', 'tap', str(x), str(y)])
        time.sleep(4)  # 每步间隔4秒
    
    print("-------------------------------等待对局开始-----------------------------------")

def data_collector():
    wait_start_time = time.time()  # 记录开始等待时间
    TIMEOUT_SECONDS = 20  # 超时时间
    
    while True:
        # 获取当前的图像 (使用 ADB 截图，避免 scrcpy 黑屏问题)
        state = tool.take_screenshot()
        # 保证图像能正常获取
        if state is None:
            time.sleep(0.01)
            continue
        # cv2.imwrite('output_image.jpg', state)
        # 初始化对局状态 对局未开始
        globalInfo.set_game_end()
        # 判断对局是否开始 (debug模式跳过检测)
        if args.debug:
            checkGameStart = 'started'
        else:
            checkGameStart = start_check.get_max_label(state)
            
        # 调试：保存当前截图并打印检测结果
        if checkGameStart != 'started':
            # 每100帧保存一次截图用于调试
            import os
            debug_dir = 'debug_screenshots'
            if not os.path.exists(debug_dir):
                os.makedirs(debug_dir)
            # 只保存一张用于分析
            cv2.imwrite(f'{debug_dir}/current_screen.jpg', state)
            print(f"对局未开始 (检测结果: {checkGameStart}, 截图已保存到 {debug_dir}/current_screen.jpg)")

        if checkGameStart == 'started':
            print("-------------------------------对局开始-----------------------------------")
            globalInfo.set_game_start()
            wait_start_time = time.time()  # 重置等待时间

            # 对局开始了，进行训练
            while globalInfo.is_start_game():
                # 获取预测动作
                action = agent.select_action(state)

                next_state, reward, done, info = env.step(action)
                print(info, reward)

                # 对局结束
                if done == 1:
                    print("-------------------------------对局结束-----------------------------------")
                    globalInfo.set_game_end()
                    # 自动开始新对局
                    auto_start_new_game()
                    wait_start_time = time.time()  # 重置等待时间
                    break

                # 追加经验
                globalInfo.store_transition_dqn(state, action, reward, next_state, done)

                state = next_state

        else:
            # 检查是否超时
            elapsed = time.time() - wait_start_time
            if elapsed > TIMEOUT_SECONDS:
                print(f"-------------------------------超时 {TIMEOUT_SECONDS} 秒未进入对局，退出训练-----------------------------------")
                # 保存模型后退出
                agent.save_model('src/wzry_ai.pt')
                print("模型已保存到 src/wzry_ai.pt")
                import sys
                sys.exit(0)
            
            print(f"对局未开始 (已等待 {elapsed:.1f} 秒)")
            time.sleep(0.1)


def train_agent():
    count = 1
    while True:
        # 只在对局进行中才训练
        if not globalInfo.is_start_game():
            time.sleep(1)
            continue
        if not globalInfo.is_memory_bigger_batch_size_dqn():
            time.sleep(1)
            continue
        print("training")
        agent.replay()
        if count % args.num_episodes == 0:
            agent.save_model('src/wzry_ai.pt')
        count = count + 1
        if count >= 100000:
            count = 1

if __name__ == '__main__':
    import signal
    import sys
    
    # 全局退出标志
    exit_flag = False
    
    def signal_handler(sig, frame):
        global exit_flag
        print("\n-------------------------------收到退出信号，正在保存模型...-----------------------------------")
        exit_flag = True
        agent.save_model('src/wzry_ai.pt')
        print("模型已保存到 src/wzry_ai.pt")
        print("正在退出...")
        sys.exit(0)
    
    # 注册 Ctrl+C 信号处理
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # 设置训练线程为守护线程（主线程退出时自动退出）
    training_thread = threading.Thread(target=train_agent, daemon=True)
    training_thread.start()
    
    try:
        data_collector()
    except KeyboardInterrupt:
        signal_handler(None, None)

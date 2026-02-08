import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import numpy as np
import torch
from rapidocr_onnxruntime import RapidOCR

from argparses import device
from globalInfo import GlobalInfo
from onnxRunner import OnnxRunner


class GetRewordUtil:
    def __init__(self):
        self.device = device

        # 全局状态
        self.globalInfo = GlobalInfo()
        class_names = ['death']
        self.death_check = OnnxRunner('models/death.onnx', classes=class_names)
        # 缓存 OCR 实例，避免每次调用都创建
        self.ocr = RapidOCR()
        
        # HP 追踪变量（用于密集奖励）
        self.last_my_hp = 100  # 上一帧自己血量百分比

    def detect_my_hp(self, img):
        """
        检测自己英雄的血量百分比 (通过屏幕左下角血条颜色)
        返回 0-100 的血量百分比
        """
        if img is None or img.size == 0:
            return self.last_my_hp
        
        try:
            image_height, image_width = img.shape[:2]
            
            # 自己血条位置 (根据用户提供的坐标)
            # 左上角: (0.455, 0.372), 右下角: (0.553, 0.389)
            left = int(image_width * 0.455)
            top = int(image_height * 0.372)
            right = int(image_width * 0.553)
            bottom = int(image_height * 0.389)
            
            # 裁剪血条区域
            hp_bar = img[top:bottom, left:right]
            
            if hp_bar is None or hp_bar.size == 0:
                return self.last_my_hp
            
            # 转换到 HSV 检测绿色（满血）区域
            hsv = cv2.cvtColor(hp_bar, cv2.COLOR_BGR2HSV)
            
            # 绿色范围 (血条颜色)
            green_lower = np.array([35, 50, 50])
            green_upper = np.array([85, 255, 255])
            
            # 创建掩码
            mask = cv2.inRange(hsv, green_lower, green_upper)
            
            # 计算绿色像素占比
            green_pixels = cv2.countNonZero(mask)
            total_pixels = hp_bar.shape[0] * hp_bar.shape[1]
            
            if total_pixels == 0:
                return self.last_my_hp
            
            hp_percentage = int((green_pixels / total_pixels) * 100)
            return min(max(hp_percentage, 0), 100)
            
        except Exception as e:
            return self.last_my_hp

    def predict(self, img):
        if img is None or img.size == 0:
            return False, 0
        is_attack, rewordCount = self.calculate_attack_reword(img)
        return is_attack, rewordCount

    def calculate_attack_reword(self, img):
        # 检查图像是否为空
        if img is None or img.size == 0:
            return False, 0

        # 获取图像的尺寸
        image_height, image_width = img.shape[:2]

        # 截取矩形的固定宽度和高度
        width = image_width * 0.116
        height = image_height * 0.024

        total_area = int(width * height)

        # 计算中心顶部矩形的起始点
        # left = int(image_width * 0.568)
        left = int(image_width * 0.57)
        top = int(image_height * 0.019)  # 从顶部开始
        right = int(left + width)
        bottom = int(top + height)

        # 根据计算出的坐标裁剪图像
        cropped_img = img[top:bottom, left:right]

        # 检查裁剪后的图像是否为空
        if cropped_img is None or cropped_img.size == 0:
            return False, 0

        # 将图片从BGR转换到HSV色彩空间
        hsv_image = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2HSV)

        # 定义BGR颜色 #AF363E
        bgr_color = np.uint8([[[62, 54, 175]]])  # 注意这里是BGR格式
        hsv_color = cv2.cvtColor(bgr_color, cv2.COLOR_BGR2HSV)
        hue = hsv_color[0][0][0]

        # 设置颜色范围的容错率
        tolerance = 10  # 容差值可以根据需要调整

        # 定义HSV中想要提取的颜色范围
        lower_bound = np.array([hue - tolerance, 50, 50])
        upper_bound = np.array([hue + tolerance, 255, 255])

        # 使用cv2.inRange()函数找到图像中颜色在指定范围内的区域
        mask = cv2.inRange(hsv_image, lower_bound, upper_bound)

        # 将掩码应用于原图像，只保留指定颜色的区域
        color_segment = cv2.bitwise_and(cropped_img, cropped_img, mask=mask)

        # 转成灰度图
        gray = cv2.cvtColor(color_segment, cv2.COLOR_BGR2GRAY)

        # 找到指定颜色的最右边的位置
        rightmost_position = 0
        for col in range(gray.shape[1]):
            if np.any(gray[:, col] != 0):
                rightmost_position = col

        # 计算指定颜色的面积
        area = rightmost_position * height

        isAttack = False
        res = 0
        if area > 0:
            isAttack = True
            p = int((area * 10) / total_area)
            if p > 9:
                res = 0
            else:
                res = 11 - int((area * 10) / total_area)

        return isAttack, res

    def calculate_reword(self, status_name, attack_reword, action):
        rewordResult = 0

        if status_name is None:
            rewordResult = -1
        elif status_name == "attack":
            move_action, angle, info_action, attack_action, action_type, arg1, arg2, arg3 = action

            pass_attack_action = [1, 2, 3, 8, 9, 10]
            if move_action != 0 or attack_action in pass_attack_action:

                rewordResult = attack_reword
            else:
                rewordResult = -1
        elif status_name == "backHome":
            move_action, angle, info_action, attack_action, action_type, arg1, arg2, arg3 = action

            if move_action == 0 and info_action == 0 and attack_action == 0:
                rewordResult = 1
            else:
                rewordResult = -1
        elif status_name == "death":
            move_action, angle, info_action, attack_action, action_type, arg1, arg2, arg3 = action

            if move_action == 0 and info_action == 0 and attack_action == 0:
                rewordResult = -10
            else:
                rewordResult = -30  # 增加死亡惩罚

        elif status_name == "successes":
            rewordResult = 50  # 降低胜利奖励，避免 loss 爆炸
        elif status_name == "failed":
            rewordResult = -50  # 降低失败惩罚
        elif status_name == "death":
            rewordResult = -1

        return rewordResult

    def check_finish(self, image):
        result, _ = self.ocr(image)
        done = 0
        class_name = None
        if result:
            for line in result:
                text = line[1]  # RapidOCR 返回格式: [box, text, score]
                if text == "胜利" or text == "VICTORY":
                    done = 1
                    class_name = 'successes'
                    break
                elif text == "失败" or text == "DEFEAT":
                    done = 1
                    class_name = 'failed'
                    break
        return done, class_name

    def check_death(self, image):
        checkGameDeath = self.death_check.get_max_label(image)

        if checkGameDeath == 'death':
            return checkGameDeath
        return None

    def get_reword(self, image_path, isFrame, action):
        if isFrame:
            image = image_path
        else:
            image = cv2.imread(image_path)

        done = 0
        class_name = None
        death_class_name = None
        md_class_name = None
        # 使用 ThreadPoolExecutor 进行并行处理
        with ThreadPoolExecutor() as executor:
            # 记录开始时间
            start_time_class_name = time.time()
            start_time_md_class_name = time.time()

            # 提交任务,预测状态
            future_class_name = executor.submit(self.check_finish, image)
            future_check_death = executor.submit(self.check_death, image)

            future_md_class_name = executor.submit(self.predict, image)

            # 等待所有任务完成
            for future in as_completed([future_class_name, future_check_death, future_md_class_name]):
                end_time = time.time()
                if future == future_class_name:
                    done, class_name = future.result()
                    # print(f"tp运行时间: {end_time - start_time_class_name:.3f} 秒")
                elif future == future_check_death:
                    death_class_name = future.result()
                elif future == future_md_class_name:
                    is_attack, attack_rewordCount = future.result()
                    if is_attack:
                        md_class_name = "attack"
                    # print(f"md运行时间: {end_time - start_time_md_class_name:.3f} 秒")

            # 如果没结束，判断局内状态
            if done == 0:
                if death_class_name is not None:
                    class_name = death_class_name
                elif md_class_name is not None:
                    class_name = md_class_name

        # 计算回报
        rewordCount = self.calculate_reword(class_name, attack_rewordCount, action)
        
        # 计算血量变化奖励（密集奖励）
        current_hp = self.detect_my_hp(image)
        hp_change = current_hp - self.last_my_hp
        
        if hp_change < 0:
            # 受到伤害，给予惩罚（每损失1%血量扣0.5分）
            hp_reward = hp_change * 0.5
            rewordCount += hp_reward
        elif hp_change > 0:
            # 血量恢复，给予奖励（每恢复1%血量加0.3分）
            hp_reward = hp_change * 0.3
            rewordCount += hp_reward
        
        # 更新血量记录
        self.last_my_hp = current_hp
        
        # 奖励归一化 + 裁剪 (将奖励限制在 [-1, +1] 范围)
        # 除以50归一化，然后裁剪
        normalized_reward = rewordCount / 50.0
        final_reward = max(-1.0, min(1.0, normalized_reward))

        return final_reward, done, class_name


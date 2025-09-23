import datetime
import os
import time

import cv2
import numpy as np
import pyautogui
import yaml

from fisher_obj import HotKeyFrameWork, Log
from paddleocr import PaddleOCR

from pynput.mouse import Controller, Button


class YamlConfig:
    """
    操作读写数据
    """
    file_name: str
    content: dict

    def __init__(self, file_name: str):
        """
        初始化
        """
        self.file_name = file_name

        with open(file_name, "r", encoding="utf-8") as file:
            self.content = dict(yaml.safe_load(file))

    def save(self) -> bool:
        """
        保存操作
        """
        with open(self.file_name, "w", encoding="utf-8") as file:
            yaml.dump(self.content, file, allow_unicode=True, default_flow_style=False)
            return True


# 配置读取器
YAML_READER = YamlConfig("config.yml")
# 鼠标控制器
MOUSE = Controller()
# 钓到鱼识别器
GOT = cv2.imread("you_got.png")
# 初始化 OCR 模型
OCR_RECOGNIZER = PaddleOCR(use_angle_cls=True, rec=True, show_log=False, cls=True)

# 定义图片截取范围
X1 = YAML_READER.content['x1']
X2 = YAML_READER.content['x2']
Y1 = YAML_READER.content['y1']
Y2 = YAML_READER.content['y2']

# 图片日志写入路径
PATH: str = datetime.datetime.now().strftime('%Y.%m.%d_fishing')
os.makedirs(f"log/{PATH}", exist_ok=True)

# 失败异常值处理
F_X1 = YAML_READER.content['failure_x1']
F_X2 = YAML_READER.content['failure_x2']
F_Y1 = YAML_READER.content['failure_y1']
F_Y2 = YAML_READER.content['failure_y2']

# 钓鱼时间
FISHING_TIMES = YAML_READER.content['FISHING_TIMES']

# 日志器
logger = Log(__name__)

# 抛竿用的时间
THROWING_ROD_TIME = YAML_READER.content['THROWING_ROD_TIME']

# 首次拉杆/松杆用时
PULL_TIME = YAML_READER.content['PULL_TIME']
RELEASE_TIME = YAML_READER.content['RELEASE_TIME']

# 循环拉杆/松杆控制指针用时
PULL_TIME_LOOP = YAML_READER.content['PULL_TIME_LOOP']
RELEASE_TIME_LOOP = YAML_READER.content['RELEASE_TIME_LOOP']


def ParseImageText(img) -> str:
    """
    使用 PaddleOCR 识别图像中的文字
    :param img: 图片路径
    :return: 识别到的文本列表
    """
    # 读取图像并识别
    result = OCR_RECOGNIZER.ocr(img)

    # 提取识别的文本
    text = ""
    try:
        for line in result:
            for word in line:
                text += word[1][0] + "\n"  # 提取文字部分
        return text
    except TypeError:
        return text


def MatchImg(img_target, threshold=0.006) -> bool:
    # 获取当前截图并转为 numpy 数组
    screenshot = pyautogui.screenshot()
    screenshot_np = np.array(screenshot)  # 将 PIL 图像转为 numpy 数组
    scn_image = cv2.cvtColor(screenshot_np, cv2.COLOR_RGB2BGR)  # 转为 BGR 格式以兼容 OpenCV

    # 确保图像正确加载
    if scn_image is None or img_target is None:
        raise FileNotFoundError(f"[ERROR]\033[1;31m空图像错误\033[0m")

    # 获取目标图像尺寸
    this_high, this_wid, _ = img_target.shape
    # 执行模板匹配
    this_result = cv2.matchTemplate(scn_image, img_target, cv2.TM_SQDIFF_NORMED)
    # 获取匹配结果最小值
    min_val, _, min_loc, _ = cv2.minMaxLoc(this_result)
    # 检查是否超过阈值
    return min_val < threshold


def GetScreenImg():
    # 获取当前截图并转为 numpy 数组
    screenshot = pyautogui.screenshot()
    screenshot_np = np.array(screenshot)  # 将 PIL 图像转为 numpy 数组
    return cv2.cvtColor(screenshot_np, cv2.COLOR_RGB2BGR)  # 转为 BGR 格式以兼容 OpenCV


def GetZoneImage(x1, x2, y1, y2):
    # 截取指定区域 (注意 numpy 的切片是 [y1:y2, x1:x2])
    return GetScreenImg()[y1:y2, x1:x2]


def GetNumOfBails():
    """ 获取诱饵的数量 """
    img = GetZoneImage(X1, X2, Y1, Y2)

    # 放大 4 倍
    img = cv2.resize(img, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
    # 灰度化
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 二值化增强对比
    _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    result = ParseImageText(img)

    # if result == "":
    #     cv2.imwrite(f"log/{time.time()}.png", img)

    num: int
    try:
        num = int(result)
    except ValueError:
        num = -1
    # logger.info(f"获取到的诱饵数量为&a{num}")
    return num


def GetFishingMsg():
    # 检查顶部消息栏
    img = GetZoneImage(
        F_X1,
        F_X2,
        F_Y1,
        F_Y2,
    )
    return ParseImageText(img)


def GetTimeForImg() -> str:
    """
    返回当前时间，格式为 时:分:秒
    """
    return time.strftime("%H-%M-%S", time.localtime())


class Fisherman(HotKeyFrameWork):
    """
    渔夫主类
    """
    # 诱饵数量
    term: int = 0

    def __init__(self):
        super().__init__()

    def pause(self):
        self.term = 0
        super().pause()

    def do(self):
        """
        主进程
        :return:
        """
        # 获得诱饵数量
        num: int = GetNumOfBails()
        logger.info(f"获取到的诱饵数量为 &a{num}")
        while self.running:
            # 检查识别到的诱饵数量
            self.term += 1
            if num in [0, -1]:
                cv2.imwrite(f"log/{PATH}/{GetTimeForImg()}_IllegalNumOfRod.png", GetScreenImg())
                logger.error(f"{self.term}: 诱饵的数量是非法的 -> \"{num}\"")
                self.pause()

            # 目标数量
            targetBails = num - 1

            #  抛竿
            logger.info(f"{self.term}: 执行抛竿操作")
            self.press()
            self.sleep(THROWING_ROD_TIME)
            self.release()

            # 检查当前状态能否钓鱼
            self.sleep(1)
            msg: str = GetFishingMsg()
            if "无法钓鱼" in msg:
                cv2.imwrite(f"log/{PATH}/{GetTimeForImg()}_CannotFishing.png", GetScreenImg())
                logger.error(f"{msg}")
                self.pause()

            logger.info(f"{self.term}: 等待鱼上钩")
            # 等待上钩
            start = time.time()
            while self.running:
                if GetNumOfBails() == targetBails:
                    break
                # 超时
                if time.time() - start > 60:
                    break
                time.sleep(0.3)
            else:
                continue

            logger.info(f"{self.term}: 鱼上钩了")
            # 上钩阶段：先拉紧6秒，再松手，再反复拉松直到抓上来为止
            self.press()
            self.sleep(PULL_TIME)
            self.release()
            self.sleep(RELEASE_TIME)

            start = time.time()
            while self.running:
                # 抓成功了
                if MatchImg(GOT):
                    num = targetBails
                    # 日志记录
                    cv2.imwrite(f"log/{PATH}/{GetTimeForImg()}_Got.png", GetScreenImg())
                    logger.info(f"{self.term}: 正在跳过展示鱼阶段")
                    # 把展示成果的界面点掉，然后继续钓鱼
                    self.sleep(2)
                    self.click()
                    self.sleep(2)
                    break

                # 检查鱼是否逃跑
                img = GetZoneImage(
                    F_X1,
                    F_X2,
                    F_Y1,
                    F_Y2,
                )
                result = ParseImageText(img)

                second_condition = time.time() - start > 30
                # 鱼跑了 or 超时
                if "鱼逃跑了" in result or second_condition:
                    num = targetBails
                    msg = "FishEscape" if not second_condition else "TimeOut"
                    cv2.imwrite(f"log/{PATH}/{GetTimeForImg()}_{msg}.png", GetScreenImg())
                    logger.info(f"{self.term}: 钓鱼失败...CausedBy: &c" + msg)
                    self.sleep(3)
                    break

                self.press()
                self.sleep(PULL_TIME_LOOP)
                self.release()
                self.sleep(RELEASE_TIME_LOOP)
            else:
                continue

            #  没有诱饵了，自动暂停脚本
            if FISHING_TIMES <= -1 and num == 0 \
                    or FISHING_TIMES <= self.term \
                    or num == 0:
                self.pause()

    @staticmethod
    def press():
        """ 摁下左键 """
        MOUSE.press(Button.left)

    @staticmethod
    def release():
        """ 释放左键 """
        MOUSE.release(Button.left)

    def click(self):
        # 左键点击一次
        self.press()
        time.sleep(0.1)
        self.release()


if __name__ == '__main__':
    Fisherman()

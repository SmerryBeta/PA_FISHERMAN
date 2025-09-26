import datetime
import os
import shutil
import threading
import time
from typing import Callable

import cv2
import numpy as np
import pyautogui
import yaml
from paddleocr import PaddleOCR
from pynput.mouse import Controller, Button

from fisher_obj import HotKeyFrameWork, Log, color


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
        self.load()

    def load(self):
        with open(self.file_name, "r", encoding="utf-8") as file:
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
OCR_RECOGNIZER = PaddleOCR(
    use_angle_cls=True,
    rec=True,
    show_log=False,
    cls=True,
    # det_model_dir="models/det/ch/ch_PP-OCRv4_det_infer",
    # rec_model_dir="models/rec/ch/ch_PP-OCRv4_rec_infer",
    # cls_model_dir="models/cls/ch_ppocr_mobile_v2.0_cls_infer"
)
# 日志器
logger = Log(__name__)

# 图片日志写入路径
PATH: str = datetime.datetime.now().strftime('%Y.%m.%d_fishing')
os.makedirs(f"log/{PATH}", exist_ok=True)

# 定义图片截取范围
X1: int
X2: int
Y1: int
Y2: int

# 失败异常值处理
F_X1: int
F_X2: int
F_Y1: int
F_Y2: int

# 钓鱼次数
FISHING_TIMES: int

# 抛竿用的时间
THROWING_ROD_TIME: float

# 展示钓上来的鱼的时间 (单位: 秒)
SHOW_FISH_TIME: float

# 首次拉杆/松杆用时
PULL_TIME: float
RELEASE_TIME: float

# 循环拉杆/松杆控制指针用时
PULL_TIME_LOOP: float
RELEASE_TIME_LOOP: float

# 脚本开关按钮
RUNNING_PAUSE_BTN: str = "f8"

# 关闭开启提示音
SOUND: bool = True

# 关闭开启提示音
LOG_AUTO_CLEANS_UP: bool = True


def Load(reload: bool = True):
    """
    这样就可以支持热重载了
    """
    # 刷新 Yaml 在内存中的内容
    if reload:
        YAML_READER.load()

    # 加载全局变量
    for k, v in YAML_READER.content.items():
        if k == "THROWING_ROD_TIME":
            v = min(v, 2.5)
        elif k == "SHOW_FISH_TIME":
            v = min(max(v, 0.6), 60)

        globals()[k] = v

    logger.info("全局变量加载完毕！")


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
    LEFT_BTN: Button = Button.left

    def __init__(self, start_btn: str, close_btn: str = "None", beep: bool = True):
        super().__init__(hold=False, start_btn=start_btn, close_btn=close_btn, beep=beep)
        threading.Thread(
            target=self.cmdHandler
        ).start()
        self.hold()

    def print_msg(self):
        print(color.yellow + "*=" * 20 + color.clear + "\n\n",
              f"{color.red}     按下 {color.blue}{self.start_btn.upper()} {color.red}启动或暂停脚本{color.clear}\n"
              f"{color.red}      指令 {color.blue}/reload {color.red}重载配置文件{color.clear}\n "
              f"{color.red}     指令 {color.blue}/exit {color.red}关闭脚本{color.clear}\n\n",
              color.yellow + "*=" * 20 + color.clear)

    def cmdHandler(self):
        """
        支持一些普通指令
        """
        while not self.exit_sign:
            cmd = input()
            logger.info(f"用户提交了指令 -> \"{cmd}\"")

            if cmd.startswith("/"):
                cmd = cmd[1:]

            # 关闭脚本指令
            if cmd == "exit":
                self.exit()

            # 重载指令
            elif cmd == "reload":
                Load()

            # 对于无法识别的指令
            else:
                logger.info("未知的指令... 以下是可用指令列表")
                logger.info("- reload: 从配置文件中重载全局变量")
                logger.info("- exit: 关闭脚本")
        else:
            logger.info("正在关闭指令处理器... ")

    def pause(self):
        """ 暂停脚本 """
        self.term = 0
        super().pause()

    def do(self):
        """
        主进程
        :return:
        """
        # 获得诱饵数量
        num: int = -1
        checkBails: bool = True
        while self.running:
            # 检查识别到的诱饵数量
            self.term += 1
            if checkBails:
                num = GetNumOfBails()
                logger.info(f"获取到的诱饵数量为 &a{num}")
                checkBails = False

            if num in [0, -1]:
                cv2.imwrite(f"log/{PATH}/{GetTimeForImg()}_IllegalNumOfBails.png", GetScreenImg())
                logger.warning(f"{self.term}: 诱饵的数量是非法的 -> \"{num}\"，未匹配到 or 配置文件 x1~y2 设置有误。")
                self.pause()
                return

            # 计算目标数量
            targetBails = num - 1

            #  抛竿
            logger.info(f"{self.term}: 执行抛竿操作")
            self.click(THROWING_ROD_TIME)

            # 检查当前状态能否钓鱼
            logger.info(f"{self.term}: 检查当前状态能否钓鱼")
            self.sleep(1)
            msg: str = GetFishingMsg()

            if "无法钓鱼" in msg:
                cv2.imwrite(f"log/{PATH}/{GetTimeForImg()}_CannotFishing.png", GetScreenImg())
                logger.warning(f"{msg}")
                self.pause()
                return

            # 等待上钩
            logger.info(f"{self.term}: 等待鱼上钩")
            start = time.time()
            normal: bool = True
            while self.running:
                # 鱼上钩了
                if GetNumOfBails() == targetBails:
                    break

                # 检查鱼是否逃跑
                result = GetFishingMsg()
                # 鱼跑了 or 超时
                if "鱼逃跑了" in result:
                    num = targetBails
                    msg = "FishEscape"
                    cv2.imwrite(f"log/{PATH}/{GetTimeForImg()}_{msg}.png", GetScreenImg())
                    logger.info(f"{self.term}: 钓鱼失败...CausedBy: &c" + msg)
                    logger.info(f"{self.term}: 由于钓鱼失败, 下次钓鱼前需要检查诱饵数量...")
                    checkBails = True
                    normal = False
                    break

                # 超时
                if time.time() - start > 60:
                    cv2.imwrite(f"log/{PATH}/{GetTimeForImg()}_TimeOut.png", GetScreenImg())
                    logger.info(f"{self.term}: 钓鱼失败...CausedBy: &cTimeOut")

                    # 做诱饵数量检查
                    logger.info(f"{self.term}: 由于钓鱼超时, 下次钓鱼前需要检查诱饵数量...")
                    checkBails = True
                    normal = False
                    break
                time.sleep(0.15)
            else:
                return

            if not normal:
                continue

            # 上钩阶段：先拉紧6秒，再松手，再反复拉松直到抓上来为止
            logger.info(f"{self.term}: 鱼上钩了")

            self.click(PULL_TIME)
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
                    self.sleep(SHOW_FISH_TIME)
                    self.click()
                    self.sleep(0.3)
                    break

                # 检查鱼是否逃跑
                result = GetFishingMsg()
                # 检查是否超时
                second_condition: bool = time.time() - start > 30
                # 鱼跑了 or 超时
                if "鱼逃跑了" in result or second_condition:
                    num = targetBails
                    msg = "FishEscape" if not second_condition else "TimeOut"
                    cv2.imwrite(f"log/{PATH}/{GetTimeForImg()}_{msg}.png", GetScreenImg())
                    logger.info(f"{self.term}: 钓鱼失败...CausedBy: &c" + msg)
                    checkBails = True
                    self.sleep(1)
                    break

                self.click(PULL_TIME_LOOP)
                self.sleep(RELEASE_TIME_LOOP)
            else:
                return

            #  没有诱饵了，自动暂停脚本
            if FISHING_TIMES <= -1 and num == 0 \
                    or self.term >= FISHING_TIMES > 0 \
                    or num == 0:
                self.pause()
                return

    def press(self):
        """ 摁下左键 """
        MOUSE.press(self.LEFT_BTN)

    def release(self):
        """ 释放左键 """
        MOUSE.release(self.LEFT_BTN)

    def click(self, duration: float = 0.1):
        # 左键点击一次
        self.press()
        time.sleep(duration)
        self.release()

    def runTask(self, task: Callable, elseDo: Callable = None):
        """
        执行一些任务，当 running 变为 False 的时候也可以及时停下来。
        :param task: 任务， 返回 bool。
        :param elseDo: 如果不 running 了执行啥 ， 返回 bool。
        :return: task or elseDo 返回的 bool 值
        """
        while self.running:
            if task():
                return
        else:
            if elseDo is not None:
                elseDo()


def CleanUpLogImg():
    if not LOG_AUTO_CLEANS_UP:
        return

    file_path: str = (datetime.datetime.now() + datetime.timedelta(days=-1)). \
        strftime('log/%Y.%m.%d_fishing')

    if os.path.isdir(file_path):
        shutil.rmtree(file_path)  # 删除整个目录
        logger.info(f"已自动清理目录: {file_path}")


# 重载全局变量
Load(reload=False)

if __name__ == '__main__':
    CleanUpLogImg()
    Fisherman(start_btn=RUNNING_PAUSE_BTN, beep=SOUND)

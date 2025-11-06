import datetime
import math
import os
import shutil
import threading
import time
import random as rd
from typing import Callable

import cv2
import keyboard
import numpy as np
import pyautogui
import yaml
import pandas as pd
from paddleocr import PaddleOCR
from pynput.mouse import Controller, Button

from fisher_obj import HotKeyFrameWork, Log, color, WinPic


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

# 加时按钮自动点掉
ADD_TIME: bool = True
Yes = cv2.imread("img/Yes.png")
No = cv2.imread("img/No.png")

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

# 窗口截取器
scn_shot: WinPic

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

# 卖出鱼价格信息捕获
S_X1: int
S_X2: int
S_Y1: int
S_Y2: int

# 防脚本检测随机震荡指数
USING_OSCILLATION_INDEX: bool = False

# 防脚本检测随机震荡指数
OSCILLATION_INDEX: float = 0

# 随机发送表情包的概率
SEND_EMO: float = 0

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

# 最高可售出的价值
SELL_LIMIT: int = 899

# 当前已出售价值
HAS_SELLED_PRICE: int = 0

# 单次最大卖出价格
SINGLE_SELL_LIMIT: int = 20

# 识别到对应的图片自动执行的策略
IMG_STRATEGY = []

# 存放小鱼干数量的图片
DRY_FISH_IMG = {}

# 品质
QUALITY: list = []

# 鱼的种类
CATEGORY: list = []

# 记录鱼的对象
DATA: pd.DataFrame

# 保存的未知
DATA_PATH = "data.xlsx"


def RegStrategy(img_path: str, threshold: float = 0.006):
    def decorator(func):
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"{img_path} 这个路径指向的不是一个图片！")

        def rf(xy):
            func(img, xy)

        IMG_STRATEGY.append((img, rf, threshold))
        return rf

    return decorator


def AddFish(name: str, quality: str, count: int):
    """增加鱼数量，如果已存在则叠加"""
    global DATA
    existing = DATA[(DATA["种类"] == name) & (DATA["品质"] == quality)]
    if not existing.empty:
        DATA.loc[existing.index, "数量"] += count
    else:
        new_row = {"种类": name, "品质": quality, "数量": count}
        DATA = pd.concat([DATA, pd.DataFrame([new_row])], ignore_index=True)
    # 自动保存
    if DATA_PATH.endswith('xlsx'):
        try:
            DATA.to_excel(DATA_PATH, index=False)
        except PermissionError as e:
            logger.error("无法计入数量，请关闭xlsx文件！", e)
    else:
        DATA.to_csv(DATA_PATH, index=False)
    logger.info(f"捕获了{quality}的{name}共{count}个")
    return DATA


def SellFish():
    global HAS_SELLED_PRICE

    keys = list(DRY_FISH_IMG.keys())
    min_price = min(keys)
    running = True
    img = DRY_FISH_IMG.get(min_price)

    while running:
        if min_price > SINGLE_SELL_LIMIT or \
                min_price > SELL_LIMIT - HAS_SELLED_PRICE:
            logger.info("当前已卖达上限")
            break

        xy = MatchImg(
            img_target=img,
            scn_img=GetZoneImage(
                S_X1,
                S_X2,
                S_Y1,
                S_Y2,
            ))

        if xy:
            # 当前这个价位能卖：卖出并计入
            HAS_SELLED_PRICE += min_price
            pyautogui.click(xy[0], xy[1])
            logger.info(f"卖出了 {min_price} 鱼干")
            time.sleep(1)
        else:
            # 当前这个价位不可卖：移除当前价格，价格上涨一位
            keys.remove(min_price)
            min_price = min(keys)


def Load(reload: bool = True):
    """
    这样就可以支持热重载了
    """
    # 刷新 Yaml 在内存中的内容
    if reload:
        YAML_READER.load()

    # 加载卖鱼干的图片
    path = "img/dry"
    for p in os.listdir(path):
        base_name = os.path.basename(p)

        for e in [".png", ".jpg"]:
            if not base_name.endswith(e):
                continue

        num = base_name. \
            replace(".png", ""). \
            replace(".jpg", "")

        if num.isdigit():
            DRY_FISH_IMG[int(num)] = cv2.imread(f"{path}/{base_name}")

    # 加载全局变量
    for k, v in YAML_READER.content.items():
        if k == "THROWING_ROD_TIME":
            v = min(v, 2.5)
        elif k == "SHOW_FISH_TIME":
            v = min(max(v, 0.6), 60)
        elif k == 'USE_WIN_CAPTURE':
            k = 'scn_shot'
            v = WinPic(v)

        globals()[k] = v

    # 保存钓鱼记录的部分代码
    file_path = globals()["DATA_PATH"]
    # 判断文件是否存在
    if os.path.exists(file_path):
        # 文件存在，读取
        globals()['DATA'] = pd.read_excel(file_path) \
            if file_path.endswith('xlsx') \
            else pd.read_csv(file_path)
    else:
        # 文件不存在，创建一个空的 DataFrame 并保存
        globals()['DATA'] = pd.DataFrame(columns=["种类", "品质", "数量"])
        if file_path.endswith('xlsx'):
            globals()['DATA'].to_excel(file_path, index=False)
        else:
            globals()['DATA'].to_csv(file_path, index=False)
        logger.info(f"📄已创建新的 Excel 文件： {file_path}")

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


def MatchImg(img_target, scn_img=None, threshold=0.006):
    if scn_img is None:
        scn_img = PyAutoGUIShot()

    # 确保图像正确加载
    if scn_img is None or img_target is None:
        raise FileNotFoundError(f"[ERROR]\033[1;31m空图像错误\033[0m")

    # 获取目标图像尺寸
    this_high, this_wid, _ = img_target.shape

    # 执行模板匹配
    this_result = cv2.matchTemplate(scn_img, img_target, cv2.TM_SQDIFF_NORMED)

    # 获取匹配结果最小值
    min_val, _, min_loc, _ = cv2.minMaxLoc(this_result)

    # 检查是否超过阈值
    if min_val > threshold:
        return None

    # 计算中心位置坐标
    upper_left = min_loc
    lower_right = (upper_left[0] + this_wid, upper_left[1] + this_high)
    center_x = int((upper_left[0] + lower_right[0]) / 2)
    center_y = int((upper_left[1] + lower_right[1]) / 2)
    return center_x, center_y


def PyAutoGUIShot():
    screenshot = pyautogui.screenshot()
    screenshot_np = np.array(screenshot)  # 将 PIL 图像转为 numpy 数组
    return cv2.cvtColor(screenshot_np, cv2.COLOR_RGB2BGR)  # 转为 BGR 格式以兼容 OpenCV


def GetScreenImg():
    if globals().get('USE_WIN_CAPTURE', None) is not None:
        try:
            return scn_shot.getWinPhoto()
        except Exception as e:
            scn_shot.reload()
            logger.error("使用窗口截取工具截取图片时出错，将直接使用截屏功能", e)

    return PyAutoGUIShot()


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


def MakeOscillation(number: float) -> float:
    """
    使用震荡指数
    """
    return rd.random() * OSCILLATION_INDEX * number + number


@RegStrategy(img_path="img/AddTime.png")
def onAddTime(_, __):
    button = Yes if ADD_TIME else No
    xy = MatchImg(button)
    if xy:
        btn = 'esc' if not ADD_TIME else 'space'
        duration = 2
        keyboard.press(btn)
        time.sleep(MakeOscillation(duration))
        keyboard.release(btn)
        logger.info(f"检测到&c加时&r，已自动点击 {'&a是' if ADD_TIME else '&c否'}")
    else:
        logger.error("检测到加时，但是未能匹配到对应的按钮！")


@RegStrategy(img_path="img/Afk.png")
def onAfk(_, __):
    # 按 t 防掉线
    keyboard.press('t')
    time.sleep(0.05)
    keyboard.release('t')
    logger.info(f"检测到&c挂机检测...")


@RegStrategy(img_path="img/Space.png")
def onReadyForGame(_, __):
    # 按 space 准备游戏
    keyboard.press('space')
    time.sleep(0.05)
    keyboard.release('space')
    logger.info(f"检测到&c准备游戏...")


@RegStrategy(img_path="img/Start.png")
def onStartGame(_, xy):
    # 点击启动游戏
    pyautogui.click(x=xy[0], y=xy[1])
    logger.info(f"检测到&c自定义房间页面...自动启动中")


@RegStrategy(img_path="img/GiveUp.png")
def onGiveUp(_, xy):
    # 按 t 防掉线
    pyautogui.click(x=xy[0], y=xy[1])
    logger.info(f"检测到&c弃票...")


@RegStrategy(img_path="img/Close.png")
def onClose(_, __):
    # 关闭UI
    pyautogui.press('esc')
    logger.info(f"检测到&c上一局详情UI...")


@RegStrategy(
    img_path="img/Me.png",
    threshold=0.03,
)
def onPrepare(_, xy):
    # 按 t 防掉线
    sz = pyautogui.size()
    tx = sz[0] * 0.5
    ty = sz[1] * 0.8

    if tx > xy[0]:
        # 在右边
        keyboard.press('a')
        time.sleep(0.5)
        keyboard.release('a')
    else:
        keyboard.press('d')
        time.sleep(0.5)
        keyboard.release('d')

    if ty > xy[1]:
        # 在上边
        keyboard.press('w')
        time.sleep(0.5)
        keyboard.release('w')
    else:
        keyboard.press('s')
        time.sleep(0.5)
        keyboard.release('s')

    logger.info(f"检测到&c准备状态...")


@RegStrategy(img_path="img/PressF.png")
def onPressF(_, __):
    keyboard.press('f')
    time.sleep(2)
    keyboard.release('f')
    logger.info(f"检测到&c长按F准备...")


class Fisherman(HotKeyFrameWork):
    """
    渔夫主类
    """
    # 诱饵数量
    term: int = 0
    LEFT_BTN: Button = Button.left
    RIGHT_BTN: Button = Button.right

    def __init__(self, start_btn: str, close_btn: str = "None", beep: bool = True):
        super().__init__(hold=False, start_btn=start_btn, close_btn=close_btn, beep=beep)
        threading.Thread(
            target=self._cmdHandler
        ).start()
        self.hold()

    def sleep(self, t: float):
        """
        添加了震荡指数，防官方脚本检测。
        """
        if USING_OSCILLATION_INDEX:
            # org = t
            t = MakeOscillation(t)
            # logger.info(f"DEBUG ->org: '{org}', '{t}'")
        super().sleep(t)

    def print_msg(self):
        print(color.yellow + "*=" * 20 + color.clear + "\n\n",
              f"{color.red}     按下 {color.blue}{self.start_btn.upper()} {color.red}启动或暂停脚本{color.clear}\n"
              f"{color.red}      指令 {color.blue}/reload {color.red}重载配置文件{color.clear}\n "
              f"{color.red}     指令 {color.blue}/exit {color.red}关闭脚本{color.clear}\n\n",
              color.yellow + "*=" * 20 + color.clear)

    def _cmdHandler(self):
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

            elif cmd == "info":
                logger.info(""
                            "\n  INFO  &b| &r感谢使用！这是一个猛兽派对的钓鱼脚本"
                            "\n  INFO  &b| &r由 &dSmerryBeta &r开发并由 &cPaddleOcr &r等第三方库强力驱动。"
                            "\n  INFO  &b| &r最早的版本始于 &a9/20 &r晚。"
                            "\n  INFO  &b| &r仓库地址: &chttps://github.com/SmerryBeta/PA_FISHERMAN&r"
                            "\n  INFO  &b| &r欢迎各位前来对此项目做出贡献！")

            # 加时指令
            elif cmd.lower() == "addTime".lower():
                YAML_READER.content['ADD_TIME'] = not YAML_READER.content['ADD_TIME']
                YAML_READER.save()
                globals()['ADD_TIME'] = YAML_READER.content['ADD_TIME']
                logger.info("当前为" + ("&c加时" if ADD_TIME else "&a不加时"))

            # 对于无法识别的指令
            else:
                logger.info("未知的指令... 以下是可用指令列表")
                logger.info("- reload: 从配置文件中重载全局变量")
                logger.info("- addtime: 自动点击加时按钮的是/否")
                logger.info("- info: 展示信息")
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
        self.doCheck()
        self.doFishing()

    def doAfk(self):
        self.asyncBeep(1000, 1000)
        logger.wrt_info('启动挂机模式')
        # while self.running:
        #     # 按 t 防掉线
        #     keyboard.press('t')
        #     time.sleep(0.05)
        #     keyboard.release('t')
        #     time.sleep(2)

    def doCheck(self):
        def check():
            logger.info("已启动加时检测线程")
            while self.running:
                for c in IMG_STRATEGY:
                    tg_pos = MatchImg(
                        img_target=c[0],
                        threshold=c[2]
                    )
                    if tg_pos:
                        c[1](tg_pos)

                # 三秒一次检查
                self.sleep(3)

        threading.Thread(target=check).start()

    def doFishing(self):
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
                self.doAfk()
                return

            # 计算目标数量
            targetBails = num - 1

            #  抛竿
            logger.info(f"{self.term}: 执行抛竿操作")
            self.click(duration=THROWING_ROD_TIME)

            # 检查当前状态能否钓鱼
            logger.info(f"{self.term}: 检查当前状态能否钓鱼")
            self.sleep(1)
            msg: str = GetFishingMsg()

            if "无法钓鱼" in msg:
                cv2.imwrite(f"log/{PATH}/{GetTimeForImg()}_CannotFishing.png", GetScreenImg())
                logger.warning(f"{msg}")
                self.doAfk()
                return

            # 等待上钩
            logger.info(f"{self.term}: 等待鱼上钩")
            start = time.time()
            self.sleep(1)
            self.click(right=True)
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
                    AddFish("空军", "未知", 1)
                    logger.info(f"{self.term}: 钓鱼失败...CausedBy: &c" + msg)
                    logger.info(f"{self.term}: 由于钓鱼失败, 下次钓鱼前需要检查诱饵数量...")
                    checkBails = True
                    normal = False
                    break

                # 超时
                if time.time() - start > 70:
                    cv2.imwrite(f"log/{PATH}/{GetTimeForImg()}_TimeOut.png", GetScreenImg())
                    logger.info(f"{self.term}: 钓鱼失败...CausedBy: &cTimeOut")

                    # 做诱饵数量检查
                    logger.info(f"{self.term}: 由于钓鱼超时, 下次钓鱼前需要检查诱饵数量...")
                    checkBails = True
                    normal = False
                    break
                self.sleep(0.15)
            else:
                return

            if not normal:
                continue

            # 上钩阶段：先拉紧6秒，再松手，再反复拉松直到抓上来为止
            logger.info(f"{self.term}: 鱼上钩了")

            self.click(duration=PULL_TIME)
            self.sleep(RELEASE_TIME)

            start = time.time()
            x, y = pyautogui.size()
            x1 = int(0.35 * x)
            x2 = int(x - x1)
            y1 = 0
            y2 = int(0.185 * y)
            while self.running:
                # 抓成功了
                csq = self._judgeGot(x1, x2, y1, y2)
                if csq[2]:
                    num = targetBails
                    # 日志记录
                    cv2.imwrite(
                        f"log/{PATH}/{GetTimeForImg()}_"
                        f"Got_{csq[1]}.png",
                        GetScreenImg()
                    )
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
                    if not second_condition:
                        AddFish("空军", "未知", 1)
                    logger.info(f"{self.term}: 钓鱼失败...CausedBy: &c" + msg)
                    checkBails = True
                    self.sleep(1)
                    break

                self.click(duration=PULL_TIME_LOOP)
                self.sleep(RELEASE_TIME_LOOP)
            else:
                return

            # 发送表情包
            if 1 > SEND_EMO >= 0 and rd.random() < SEND_EMO:
                logger.info("发送了表情")
                self._sendEmo()
                self.sleep(0.3)

            #  没有诱饵了，自动暂停脚本
            if FISHING_TIMES <= -1 and num == 0 \
                    or self.term >= FISHING_TIMES > 0 \
                    or num == 0:
                logger.info("暂停脚本...")
                self.doAfk()
                return

    def press(self, right: bool = False):
        """ 摁下左键 """
        MOUSE.press(self.LEFT_BTN if not right else self.RIGHT_BTN)

    def release(self, right: bool = False):
        """ 释放左键 """
        MOUSE.release(self.LEFT_BTN if not right else self.RIGHT_BTN)

    def click(self, right: bool = False, duration: float = 0.1):
        # 左键点击一次
        self.press(right)
        time.sleep(duration)
        self.release(right)

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

    def _sendEmo(self, angle: int = 180):
        angel = math.radians(angle)
        self.sleep(0.1)
        keyboard.press('t')
        self.sleep(0.2)
        w, h = pyautogui.size()
        cw, ch = w / 2, h / 2
        radius = ch / 2
        radius = MakeOscillation(radius)
        pyautogui.moveTo(
            x=cw + radius * math.cos(angel),
            y=ch + radius * math.sin(angel),
            duration=0.1
        )
        self.sleep(0.1)
        keyboard.release('t')

    @staticmethod
    def _getElementFromSentence(sentence: str, content: list):
        for e in content:
            if e in sentence:
                return e
        return "未知"

    def _judgeGot(
            self,
            x1,
            x2,
            y1,
            y2
    ) -> tuple:
        img = GetZoneImage(x1, x2, y1, y2)
        result = ParseImageText(img)
        if self._getElementFromSentence(
                result,
                ['你钓到了', '首次捕获']
        ) == "未知":
            return None, None, False

        c = self._getElementFromSentence(result, CATEGORY)
        q = self._getElementFromSentence(result, QUALITY)
        AddFish(c, q, 1)
        return c, q, True


def CleansUpLogImg():
    if not LOG_AUTO_CLEANS_UP:
        return

    file_path: str = datetime.datetime.now().strftime('%Y.%m.%d')
    for path in os.listdir("log/"):
        if file_path not in path:
            if os.path.isdir(f"log/{path}"):
                shutil.rmtree("log/" + path)  # 删除整个目录
            elif os.path.exists(f"log/{path}"):
                os.remove(f"log/{path}")

            logger.info(f"已自动清理 {'log/' + path}")


# 重载全局变量
Load(reload=False)

if __name__ == '__main__':
    CleansUpLogImg()
    Fisherman(start_btn=RUNNING_PAUSE_BTN, beep=SOUND)

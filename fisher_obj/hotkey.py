import threading
import time
from abc import abstractmethod, ABC

import keyboard
import winsound

from .log import Log, color

logger = Log(__name__)


class HotKeyFrameWork(ABC):
    """
    这个框架需要实现抽象方法do() ，用来执行按下热键后执行的操作。
    """
    # const 变量
    thread = None
    exit_sign, running = False, False

    def __init__(
            self,
            hold: bool = True,
            beep: bool = True,
            start_btn="f8",
            close_btn="f12"
    ):
        self.beep = beep
        self.start_btn = start_btn.lower()
        self.close_btn = close_btn.lower()
        # 发送提示
        self.print_msg()

        # 添加全局监听
        keyboard.hook(self.on_key_event)
        # 锁定程序
        if hold:
            self.hold()

    @abstractmethod
    def do(self):
        """
        需要子类实现
        :return: None
        """
        pass

    def asyncBeep(self, fre: int, dur: int):
        if self.beep:
            threading.Thread(
                target=lambda: winsound.Beep(fre, dur)
            ).start()

    def print_msg(self):
        logger.info("脚本启动")

        print(color.yellow + "*=" * 20 + color.clear + "\n\n",
              f"{color.red}     按下 {color.blue}{self.start_btn.upper()} {color.red}启动或暂停脚本 "
              f"{color.blue}{self.close_btn.upper()} {color.red}关闭脚本。{color.clear}\n\n",
              color.yellow + "*=" * 20 + color.clear)

    # 定义按下按键时的回调函数
    def on_key_event(self, press_key):
        if press_key.event_type == "up":
            return

        # 检查是否是按键按下事件
        if press_key.name == self.start_btn:
            self.running = not self.running

            if self.running:
                self.thread = threading.Thread(target=self.do)
                self.thread.start()
                logger.info(f"&a程序已启动")
                self.asyncBeep(500, 1000)
            else:
                self.thread.join()
                logger.info(f"&c程序已暂停")
                self.asyncBeep(500, 1000)

        # 退出脚本
        elif press_key.name == self.close_btn:
            self.exit()

    def exit(self):
        logger.info("&b正在关闭程序！")
        # 关闭脚本
        self.running = False
        self.exit_sign = True

    def hold(self):
        try:
            while not self.exit_sign:
                time.sleep(0.1)
            self.asyncBeep(666, 1000)
        except KeyboardInterrupt as e:
            logger.info(f"&b程序已被关闭: {e}")
            self.asyncBeep(666, 1000)

    def sleep(self, t: float):
        """
        休眠方法，避免time.sleep(100)的长休眠导致无法立即关闭 or 暂停脚本
        """
        while self.running and t > 0.1:
            time.sleep(0.1)
            t -= 0.1
        else:
            time.sleep(t)

    def pause(self):
        if self.running:
            self.running = not self.running
            logger.info(f"&c程序已暂停")
            self.asyncBeep(500, 1000)

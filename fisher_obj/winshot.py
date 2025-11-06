import ctypes
import os
from ctypes import wintypes

import cv2
import numpy as np
import win32gui
import win32ui

from .log import color

# 加载 DWM 库
dwmapi = ctypes.WinDLL("dwmapi")

# DwmGetWindowAttribute 函数定义
DWM_CLOAKED = 14
DWMWA_EXTENDED_FRAME_BOUNDS = 9

# Load user32 DLL
user32 = ctypes.WinDLL("user32", use_last_error=True)

# Define PrintWindow
user32.PrintWindow.argtypes = [wintypes.HWND, wintypes.HDC, wintypes.UINT]
user32.PrintWindow.restype = wintypes.BOOL


class WinPic:
    """
    通过 title 获取 hwnd 再获取窗口图片截屏
    """

    def __init__(self, title: str = None):
        self._load(title)
        self.image = self.getWinPhoto()

    def _load(self, title):
        if title is None or title.lower() == '__current__':
            '''
            使用当前聚焦窗口
            '''
            self.hwnd = self.current_hWnd()
            self.title = self.get_window_title(self.hwnd)
        else:
            '''
            使用title找到窗口
            '''
            self.title = title
            self.hwnd = self.find_window_by_title()

    def reload(self):
        self._load(title=self.title)

    def find_window_by_title(self):
        """
        根据窗口标题查找窗口句柄
        """
        this_hwnd = win32gui.FindWindow(None, self.title)
        if this_hwnd == 0:
            raise Exception(f"未找到标题为'{self.title}'的窗口")

        return this_hwnd

    @classmethod
    def current_hWnd(cls):
        temp_user32 = ctypes.windll.user32
        return temp_user32.GetForegroundWindow()

    @classmethod
    def get_window_title(cls, hwnd):
        """
        通过窗口句柄获取窗口标题
        """
        # 定义 Windows API 函数
        temp_user32 = ctypes.windll.user32
        GetWindowText = temp_user32.GetWindowTextW
        GetWindowTextLength = temp_user32.GetWindowTextLengthW
        length = GetWindowTextLength(hwnd)  # 获取标题长度
        if length > 0:
            buffer = ctypes.create_unicode_buffer(length + 1)  # 创建缓冲区
            GetWindowText(hwnd, buffer, length + 1)  # 获取窗口标题
            return buffer.value
        return None

    def getWinPhoto(self):
        left, top, right, bottom = win32gui.GetWindowRect(self.hwnd)
        width = right - left
        height = bottom - top

        hwndDC = win32gui.GetWindowDC(self.hwnd)
        mfcDC = win32ui.CreateDCFromHandle(hwndDC)
        saveDC = mfcDC.CreateCompatibleDC()

        saveBitMap = win32ui.CreateBitmap()
        saveBitMap.CreateCompatibleBitmap(mfcDC, width, height)
        saveDC.SelectObject(saveBitMap)

        # Try to capture using PrintWindow
        result = user32.PrintWindow(self.hwnd, saveDC.GetSafeHdc(), 2)
        if not result:
            raise Exception("PrintWindow failed. The window may not support this operation.")

        signedIntsArray = saveBitMap.GetBitmapBits(True)

        # Release resources
        win32gui.DeleteObject(saveBitMap.GetHandle())
        saveDC.DeleteDC()
        mfcDC.DeleteDC()
        win32gui.ReleaseDC(self.hwnd, hwndDC)

        # Convert to OpenCV format
        im_opencv = np.frombuffer(signedIntsArray, dtype='uint8')
        im_opencv.shape = (height, width, 4)  # BGRA format
        return im_opencv

    def save(self, path: str):
        # 创建目录（如果不存在）
        directory = os.path.dirname(path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
        # 图片为空
        if self.image is None or self.image.size == 0:
            raise ValueError("图像无效或为空")
        # 转换图像格式（如果需要）
        if len(self.image.shape) == 3 and self.image.shape[2] == 4:  # 判断是否有透明通道
            self.image = cv2.cvtColor(self.image, cv2.COLOR_BGRA2BGR)
        # 保存图片
        print(f'{color.info}{color.green}图片保存成功{color.clear}'
              if cv2.imwrite(path, self.image) else f'{color.red}图片保存失败{color.clear}')

    def show(self):
        cv2.imshow('Captured Window', self.image)
        cv2.waitKey(0)


if __name__ == '__main__':
    """
    用于测试代码
    """
    pic = WinPic()
    # Convert BGRA to BGR
    pic.show()
    # pic.save(path='S:/pyProjects/AnimalParty_Script/log/2024.12.04图片记录/test.jpg')

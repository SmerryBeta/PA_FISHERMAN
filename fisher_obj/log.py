import datetime
import os
import re


class color:
    green = '\033[1;32m'
    blue = '\033[1;36m'
    red = '\033[1;31m'
    yellow = '\033[1;33m'
    purple = '\033[1;95m'
    clear = '\033[0m'
    white = '\033[97m'
    info = f'[{purple}INFO{clear}] '
    warning = f'[{yellow}Warning{clear}] '
    error = f'[{red}ERROR{clear}] '

    dic = {
        "a": green,
        "b": blue,
        "c": red,
        "d": purple,
        "e": yellow,
        "r": clear,
        "f": white,
        "i": info,
        "w": warning,
        "o": error
    }

    @classmethod
    def trans(cls, msg: str) -> str:
        match = re.search('&([a-foriw])', msg)
        # 替换每一个 &<颜色符>
        while match:
            c = cls.dic.get(match.group(1), '')
            msg = msg[:match.start()] + c + msg[match.end():]
            match = re.search('&([a-foriw])', msg)

        return msg


class Log:
    """
    日志器
    """

    info_fmt: str
    time_fmt: str
    error_fmt: str
    warning_fmt: str
    by_time: bool
    path: str
    file_fmt: str

    def __init__(
            self,
            name: str = None,
            by_time=True,
            path: str = 'log/',
            file_fmt: str = '%Y.%m.%d日志记录',
            time_fmt: str = '%Y.%m.%d %H:%M:%S',
            error_fmt: str = f"{color.error}{color.red}%n {color.blue}|{color.clear} %t{color.yellow} >>{color.clear} ",
            info_fmt: str = f"{color.info}{color.white}%n {color.blue}|{color.clear} %t{color.yellow} >>{color.clear} ",
            warning_fmt: str = f"{color.warning}{color.white}%n%t{color.yellow} >>{color.clear} "
    ):
        self.path = path
        self.name = name
        self.file_fmt = file_fmt
        self.by_time = by_time
        self.info_fmt = info_fmt
        self.time_fmt = time_fmt
        self.error_fmt = error_fmt
        self.warning_fmt = warning_fmt

    def wrt_info(self, text: str):
        """
        一起操作
        """
        self.info(text)
        self.write(text)

    def write(self, text: str):
        """
         写入日志器
        """
        file_name = datetime.datetime.now().strftime(self.file_fmt)
        full_path = os.path.join(self.path, f'{file_name}.txt')

        os.makedirs(self.path, exist_ok=True)

        this_time = datetime.datetime.now().strftime(self.time_fmt)
        with open(full_path, 'a', encoding='utf-8') as file:
            file.write(f'[{this_time}] {text}\n')

    def getTime(self) -> str:
        """
        获取时间
        """
        return datetime.datetime.now().strftime(self.time_fmt)

    def info(self, msg: str):
        """
        打印信息
        """
        msg = color.trans(msg)
        prefix = self.info_fmt.replace("%t", self.getTime())
        prefix = prefix.replace("%n", "" if not self.name else self.name)
        print(f"{prefix}{msg}{color.clear}")

    def warning(self, msg: str):
        """
        打印warning
        """
        msg = color.trans(msg)
        prefix = self.warning_fmt.replace("%t", self.getTime())
        prefix = prefix.replace("%n", "" if not self.name else self.name)
        print(f"{prefix}{msg}{color.clear}")

    def error(self, msg: str, e=None):
        """
        打印错误
        """
        msg = color.trans(msg)
        prefix = self.error_fmt.replace("%t", self.getTime())
        prefix = prefix.replace("%n", "" if not self.name else self.name)
        end_msg = f"-> \"{color.red}{e}{color.clear}\"" if e else ""
        print(f"{prefix}{msg}{color.clear}{end_msg}")
        self.write(f"{msg}-> \"{e}\"")


logger = Log(__name__)

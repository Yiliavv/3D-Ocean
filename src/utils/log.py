# log class for logging
from enum import Enum, auto
import csv

class Level(Enum):
    DEBUG = auto()
    INFO = auto()
    WARNING = auto()
    ERROR = auto()
    CRITICAL = auto()

    def __lt__(self, other):
        if isinstance(other, Level):
            return self.value < other.value
        return NotImplemented

    def __le__(self, other):
        if isinstance(other, Level):
            return self.value <= other.value
        return NotImplemented

    def __gt__(self, other):
        if isinstance(other, Level):
            return self.value > other.value
        return NotImplemented

    def __ge__(self, other):
        if isinstance(other, Level):
            return self.value >= other.value
        return NotImplemented

    def __eq__(self, other):
        if isinstance(other, Level):
            return self.value == other.value
        return NotImplemented


# 定义日志级别
LOG_LEVEL = Level.DEBUG


class Log:

    @staticmethod
    def d(*msg):
        if LOG_LEVEL <= Level["DEBUG"]:
            print(f"\033[90mDEBUG: \033[0m    ", *msg)

    @staticmethod
    def i(*msg):
        if LOG_LEVEL <= Level["INFO"]:
            print(f"\033[92mINFO: \033[0m    ", *msg)

    @staticmethod
    def w(*msg):
        if LOG_LEVEL <= Level["WARNING"]:
            print(f"\033[93mWARNING: \033[0m    ", *msg)

    @staticmethod
    def e(*msg):
        if LOG_LEVEL <= Level["ERROR"]:
            print(f"\033[91mERROR: \033[0m    ", *msg)

    @staticmethod
    def c(*msg):
        if LOG_LEVEL <= Level["CRITICAL"]:
            print(f"\033[95mCRITICAL: \033[0m    ", *msg)


class CSVLogger:
    def __init__(self, filename):
        self.filename = filename
        self.writer = csv.writer(open(filename, 'w'))

    def log(self, tensor):
       for row in tensor:
           self.writer.writerow(row.tolist())


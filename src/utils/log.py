# log class for logging
from enum import Enum, auto


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
            print(*msg)

    @staticmethod
    def i(*msg):
        if LOG_LEVEL <= Level["INFO"]:
            print(*msg)

    @staticmethod
    def w(*msg):
        if LOG_LEVEL <= Level["WARNING"]:
            print(*msg)

    @staticmethod
    def e(*msg):
        if LOG_LEVEL <= Level["ERROR"]:
            print(*msg)

    @staticmethod
    def c(*msg):
        if LOG_LEVEL <= Level["CRITICAL"]:
            print(*msg)

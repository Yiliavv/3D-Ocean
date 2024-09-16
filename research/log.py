# log class for logging

from research.config.params import LOG_LEVEL, Level

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
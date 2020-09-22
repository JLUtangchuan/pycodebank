# -*- coding=utf-8 -*-

# 用于微信与控制台的交互
# 1. 将消息发送给微信
# 2. 获取窗口截图（for windows） 、图片
# 3. 微信向控制台发送指令
from wxpy import *
import sys


class PrintRedirect:
    def __init__(self):
        self.terminal = sys.stdout
        self.content = ""
 
    def write(self, message):
        self.terminal.write(message)
        self.content += message
 
    def flush(self):
        self.content = ""
 


class WXRobot():
    
    def __init__(self,redirect = True, cache_path = True):
        """初始化微信机器人

        Args:
            redirect (bool, optional): [stdout重定向，收集print函数的内容]. Defaults to True.
            cache_path (bool, optional): [登录缓存，短期不会再次扫码]. Defaults to True.
        """
        self.bot = Bot(cache_path=cache_path) # 扫码
        self.cmd = ""
        # self.bot.auto_mark_as_read = False
        self.pr = None
        if redirect == True:
            self.origin_stdout = sys.stdout
            self.pr = PrintRedirect()
            sys.stdout = self.pr

    def resetStdout(self):
        """还原Stdout
        """
        sys.stdout = self.origin_stdout

    def sendMessageToWX(self, msg=None, max_line=20):
        """发送消息给WX，注意：当发送的内容是控制台输出时，需要先重定向一下

        Args:
            msg ([type], optional): [发送消息，当为None时，输出此前print的结果]]. Defaults to None.
            max_line (int, optional): [输出最多行数]. Defaults to 20.
        """
        if msg is None:
            contents = self.pr.content.split('\n')
            num_lines = len(contents)
            contents = [l for i,l in enumerate(contents) if i > num_lines-max_line]
            res = "\n".join(contents)
            self.bot.file_helper.send(res)
        else:  
            self.bot.file_helper.send(msg)
    
    def sendImage(self, file):
        """发送图片
        """
        self.bot.file_helper.send_image(file)

if __name__ == "__main__":
    bot = WXRobot()
    for i in range(30):
        print('Hello World',i)
    bot.sendMessageToWX()
    bot.resetStdout()
    
    bot.sendImage('../img.jpg')



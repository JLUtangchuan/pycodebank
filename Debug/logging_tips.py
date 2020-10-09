# -*- coding=utf-8 -*-
# 更多内容看博客中的笔记
# https://blog.tangchuan.ink/2020/10/08/ji-zhu/bian-cheng/python-debug/

import logging  # 引入logging模块
# 默认输出级别为 warning
log_fmt = "%(asctime)s|%(levelname)s|%(filename)s:%(lineno)s|%(message)s"
logging.basicConfig(filename='debug.log', filemode='w', format=log_fmt, level=logging.DEBUG)
# 将信息打印到控制台上
logging.debug(u"aaa")
logging.info(u"bbb")
logging.warning(u"ccc")
logging.error(u"ddd")
logging.critical(u"fff")


if __name__ == "__main__":
    try:
        a = 20
        b = 0
        c = a/b
    except Exception as e:
        print('Error:',e)
        print(e.args)
        print(e.with_traceback())
        logging.critical(e)
    




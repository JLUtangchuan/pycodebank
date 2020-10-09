# -*- coding=utf-8 -*-
import smtplib
import json
import poplib
from email import parser
from email.mime.text import MIMEText
from email.header import Header


def sendEmail(msg_from, msg_to, auth_id, title, content):
    """发送邮件：目前只支持qq邮箱自动发送邮件

    Args:
        msg_from ([type]): [description]
        msg_to ([type]): [description]
        auth_id ([type]): [description]
        title ([type]): [description]
        content ([type]): [description]
    """
    msg = MIMEText(content)
    msg['Subject'] = title
    msg['From'] = msg_from
    msg['To'] = msg_to
    try:
        s = smtplib.SMTP_SSL("smtp.qq.com",465)
        s.login(msg_from, auth_id)
        s.sendmail(msg_from, msg_to, msg.as_string())
        print("发送成功")
    except s.SMTPException:
        print("发送失败")
    finally:
        s.quit()


def main():
    # 读取信息                       
    filename = './info.json'
    with open(filename, 'r') as f:
        dic = json.load(f)
    msg_from = dic['msg_from'] #发送方邮箱
    passwd = dic['passwd']  #填入发送方邮箱的授权码
    msg_to = dic['msg_to']  #收件人邮箱

    subject = "Hello World"                                     #主题     
    content = "发送邮件"     # 内容


    sendEmail(msg_from, msg_to, passwd, subject, content)

if __name__ == "__main__":
    main()
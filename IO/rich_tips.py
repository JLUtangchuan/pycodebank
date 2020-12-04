#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   rich_tips.py
@Time    :   2020/12/04 09:41:18
@Author  :   Tang Chuan 
@Contact :   tangchuan20@mails.jlu.edu.cn
@Desc    :   rich库介绍
'''

# python3 -m rich查看功能

# 想学习的内容
# 1. Markdown
# 2. 各种Style与颜色（加粗、斜体、下划线、高亮、闪烁、。。。）
# 3. Table制作
# 4. Markdown
# 5. 
from rich.markdown import Markdown
from rich.console import Console
from rich.table import Column, Table
from rich import print
md_str = """
# Hello World
## Outline
- fun1
- fun2

"""

md = Markdown(md_str)
console = Console()
console.print(md)
# 

table = Table(show_header=True, header_style="bold magenta")
table.add_column("Date", style="dim", width=12)
table.add_column("Title 😜")
table.add_column("Production Budget", justify="center")
table.add_column("Box Office", justify="center")
table.add_row(
    "Dev 20, 2019", "Star Wars: The Rise of Skywalker", "$275,000,000", "$375,126,118"
)
table.add_row(
    "May 25, 2018",
    "[red]Solo[/red]: A Star Wars Story",
    "$275,000,000",
    "$393,151,347",
)
table.add_row(
    "Dec 15, 2017",
    "Star Wars Ep. VIII: The Last Jedi",
    "$262,000,000",
    "[bold]$1,332,539,889[/bold]",
)

console.print(table)
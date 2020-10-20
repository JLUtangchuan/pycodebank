import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
import pandas as pd
from dash.dependencies import Input, Output

app = dash.Dash(name='Dash Demo')


def getDash(layout_list, name):
    # app = dash.Dash(name=name)
    app.layout = html.Div(children=layout_list)


def getImg(img):
    fig = img
    fig_id = 'Img'
    return fig, fig_id

# 定义表格组件
def create_table(df, max_rows=12):
    """基于dataframe，设置表格格式"""
    
    table = html.Table(
        # Header
        [
            html.Tr(
                [
                    html.Th(col) for col in df.columns
                ]
            )
        ] +
        # Body
        [
            html.Tr(
                [
                    html.Td(
                        df.iloc[i][col]
                    ) for col in df.columns
                ]
            ) for i in range(min(len(df), max_rows))
        ]   
    )
    return table

def optionComp():
    div = html.Div([
        # 下拉菜单（适合单选项非常多的）
        html.Label('下拉菜单'),
        dcc.Dropdown(
            options = [{'label': '北京', 'value': '北京'},
                    {'label': '天津', 'value': '天津'},
                    {'label': '上海', 'value': '上海'}],

            value = '北京'), # 默认值
        
        html.Label('多选下拉菜单'),
        dcc.Dropdown(
            options = [{'label': '北京', 'value': '北京'},
                    {'label': '天津', 'value': '天津'},
                    {'label': '上海', 'value': '上海'}],
            value = ['北京', '上海'],
            multi = True),
        html.Br(),

        # 按钮式
        html.Label('单选钮'),
        dcc.RadioItems(
            options = [{'label': '北京', 'value': '北京'},
                    {'label': '天津', 'value': '天津'},
                    {'label': '上海', 'value': '上海'}],
            value = '北京'),
        
        html.Label('多选框'),
        dcc.Checklist(
            options = [{'label': '北京', 'value': '北京'},
                    {'label': '天津', 'value': '天津'},
                    {'label': '上海', 'value': '上海'}],
            value=['北京', '上海']),
        
        # 文本输入式
        html.Label('Text Input'),
        dcc.Input(value = '天津', type = 'text'),
        html.Br(),
        
        html.Label('文本输入'),
        dcc.Slider(
            min = 0, max = 9, value = 5,
            marks = {i: '标签 {}'.format(i) if i == 1 else str(i) for i in range(1, 9)})
    ],style={'columnCount': 1})

    return div

# 输入、输出组件ID，以及其传递位置

def getTextInput():
    
    div = html.Div([
        html.Label('请输入你的账号：'),
        dcc.Input(id = 'my-id', value = '0', type = 'text'),
        html.Div(id = 'my-div')
    ])
    
    

    return div

def getLayoutList():
    """获取显示内容
    主要工作就是写这个
    """
    # 导入数据
    data = pd.read_csv('iris.data', names=['sepal length', 'sepal width', 'petal length', 'petal width', 'class'])
    
    md = ''' 
        ### Dash使用
        #### Outline
        - 可视化饼图
        - 可视化柱状图
        - 可视化散点图
        - 可视化图片
        #### 主要组件介绍
        1. html基本单元
            - `Div` 、 `H1`
        2. 表格
            - `html.Table`



        请关注我的[博客](blog.tangchuan.ink)
        '''
    li = [
        html.H1(children=f"Visual something"),
        optionComp(),
        # 绘制图片
        dcc.Markdown(children = md),

        # dcc.Graph(
        #     id=fig_id,
        #     figure=fig
        #)
        html.Div(children = create_table(data.head())),
        getTextInput()
    ]
    return li



# app.callback语法糖 重新更新layout
@app.callback(
    Output(component_id = 'my-div', component_property = 'children'),
    [Input(component_id = 'my-id', component_property = 'value')]
)
def update(val):
    """更新操作
    """
    return f"你的输入为：{val}"

if __name__ == "__main__":
    
    li = getLayoutList()
    getDash(li, 'markdown')
    app.run_server(host='0.0.0.0', port='8034', debug=True)
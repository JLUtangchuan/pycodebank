# -*- coding=utf-8 -*-
# 一个交互式demo


import dash
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html
import plotly.express as px
# import plotly.plotly as py
# import plotly.graph_objs as go
import pandas as pd
from dash.dependencies import Input, Output

app = dash.Dash('Dash nav demo', external_stylesheets=[dbc.themes.BOOTSTRAP])
# layout
controls = dbc.FormGroup(
    [
        html.P('Dropdown', style={
            'textAlign': 'center'
        }),
        dcc.Dropdown(
            id='dropdown',
            options=[{
                'label': 'Value One',
                'value': 'value1'
            }, {
                'label': 'Value Two',
                'value': 'value2'
            },
                {
                    'label': 'Value Three',
                    'value': 'value3'
                }
            ],
            value=['value1'],  # default value
            multi=True
        ),
        html.Br(),
        html.P('Range Slider', style={
            'textAlign': 'center'
        }),
        dcc.RangeSlider(
            id='range_slider',
            min=0,
            max=20,
            step=0.5,
            value=[5, 15]
        ),
        html.P('Check Box', style={
            'textAlign': 'center'
        }),
        dbc.Card([dbc.Checklist(
            id='check_list',
            options=[{
                'label': 'Value One',
                'value': 'value1'
            },
                {
                    'label': 'Value Two',
                    'value': 'value2'
                },
                {
                    'label': 'Value Three',
                    'value': 'value3'
                }
            ],
            value=['value1', 'value2'],
            inline=True
        )]),
        html.Br(),
        html.P('Radio Items', style={
            'textAlign': 'center'
        }),
        dbc.Card([dbc.RadioItems(
            id='radio_items',
            options=[{
                'label': 'Value One',
                'value': 'value1'
            },
                {
                    'label': 'Value Two',
                    'value': 'value2'
                },
                {
                    'label': 'Value Three',
                    'value': 'value3'
                }
            ],
            value='value1',
            style={
                'margin': 'auto'
            }
        )]),
        html.Br(),
        dbc.Button(
            id='submit_button',
            n_clicks=0,
            children='Submit',
            color='primary',
            block=True
        ),
    ]
)

sidebar = html.Div(
    [
        html.H2('Parameters'),
        html.Hr(),
        controls
    ],
    style={'width': '20%', 'display':'inline-block'}
)

# 图像
## 绘制plot图
## 显示指定图像
md = '''
## 图片显示区域（Powered by Markdown）
![](https://gitee.com/JLUtangchuan/imgbed/raw/master/img/samurai.jpg)

'''
content = html.Div([
        html.Img(src='https://gitee.com/JLUtangchuan/imgbed/raw/master/img/samurai.jpg', style={'width':'50%'}),
        # dcc.Markdown(md)
    ],
    style={'width': '80%', 'display':'inline-block'}
)

# @app.callback(Output(component_id='', component_property=),

#     [

#         Input(component_id=, component_property=),

#         Input(component_id=, component_property=)

#     ]
# )
# def update():
#     pass

app.layout = html.Div([sidebar, content], style={'width': '100%'})

if __name__ == "__main__":
    app.run_server(host='0.0.0.0', port='8034', debug=True)

import dash
from dash import dcc, html
import pandas as pd
import plotly.express as px

# 读取数据文件
data = pd.read_csv(r"C:\Users\许馨元\Desktop\Web5\Web4\data\Final_score.csv")

# 筛选出2023年的数据
# 这里只保留示例中的数据，如果需要特定年份，可以进一步筛选
# data = data[data['Year'] == 2023]  # 假设有'Year'列

# 根据要求生成图表
#fig_e = px.bar(data, x="Company Name", y="Score E", title="2023 Score E")
#fig_s = px.bar(data, x="Company Name", y="Score S", title="2023 Score S")
#fig_g = px.bar(data, x="Company Name", y="Score G", title="2023 Score G")

# 初始化Dash应用
app = dash.Dash(__name__)

# Layouts for each score
T_layout = html.Div([
    dcc.Graph(
        figure=px.bar(data, x='Company Name', y='Total ESG Score', color_discrete_sequence=['#4CAF50'])
        .update_xaxes(tickangle=45)
        .update_layout(template="simple_white") 
    )
])
e_layout = html.Div([
    dcc.Graph(
        figure=px.bar(data, x='Company Name', y='Score E', color_discrete_sequence=['#4CAF50'])
        .update_xaxes(tickangle=45)
        .update_layout(template="simple_white") 
    )
])

s_layout = html.Div([
    dcc.Graph(
        figure=px.bar(data, x='Company Name', y='Score S',  color_discrete_sequence=['#2196F3'])
        .update_xaxes(tickangle=45)
        .update_layout(template="simple_white") 
    )
])

g_layout = html.Div([
    dcc.Graph(
        figure=px.bar(data, x='Company Name', y='Score G',  color_discrete_sequence=['#FFC107'])
        .update_xaxes(tickangle=45)
        .update_layout(template="simple_white") 
    )
])

# Define the app layout and callback for routing
app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
])

@app.callback(
    dash.dependencies.Output('page-content', 'children'),
    [dash.dependencies.Input('url', 'pathname')]
)
def display_page(pathname):
    if pathname == '/T_chart':
        return T_layout
    elif pathname == '/e_chart':
        return e_layout
    elif pathname == '/s_chart':
        return s_layout
    elif pathname == '/g_chart':
        return g_layout
    else:
        return html.Div("404 - Page not found")

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)

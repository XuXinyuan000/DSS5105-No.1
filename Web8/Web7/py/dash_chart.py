import dash
from dash import dcc, html
import pandas as pd
import plotly.express as px

# 读取数据文件
data = pd.read_excel(r"D:\DSS\DSS5105\Web7\data\final_score.xlsx")

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
import pandas as pd
import plotly.express as px
from dash import html, dcc

# Define your score ranges and corresponding colors
score_ranges = [(0, 1.429), (1.429, 2.857), (2.857, 4.286), (4.286, 5.714), (5.714, 7.143), (7.143, 8.571), (8.571, 10)]
colors = ['#F4B1B1', '#E8C47A', '#D7D65E', '#D0E178', '#C4D88F', '#A8D585', '#8ACD6F']

# Function to map scores to colors based on ranges
def get_color(score):
    for (low, high), color in zip(score_ranges, colors):
        if low <= score < high:
            return color
    return '#000000'  # Default color if no range is matched (optional)

# Assuming 'Total ESG Score' column is in your data
data['Color'] = data['Total ESG Score'].apply(get_color)

# Create the figure
fig = px.bar(data, x='Company Name', y='Total ESG Score', color='Color',color_discrete_map='identity')

# Update layout and x-axis angle
fig.update_xaxes(tickangle=45)
fig.update_layout(template="simple_white")

# Dash layout
T_layout = html.Div([
    dcc.Graph(figure=fig)
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

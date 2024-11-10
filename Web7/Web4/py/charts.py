from flask import Flask, jsonify, Response
import pandas as pd
import plotly.express as px
import plotly.io as pio
from flask_cors import CORS
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load data
data = pd.read_csv(r"D:\DSS\DSS5105\Web4\data\Final_score.csv")

# Helper function to create Plotly chart as JSON string
def create_chart(score_column, title):
    fig = px.bar(data, x='Company Name', y=score_column, title=title)
    fig.update_layout(height=400)
    return pio.to_json(fig)  # 转换为 JSON 字符串

# Route for Score E chart
@app.route('/e_chart')
def e_chart():
    return Response(create_chart('Score E', 'Score E by Company'), mimetype='application/json')

# Route for Score S chart
@app.route('/s_chart')
def s_chart():
    return Response(create_chart('Score S', 'Score S by Company'), mimetype='application/json')

# Route for Score G chart
@app.route('/g_chart')
def g_chart():
    return Response(create_chart('Score G', 'Score G by Company'), mimetype='application/json')

if __name__ == '__main__':
    app.run(debug=True)

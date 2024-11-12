from flask import Flask, render_template_string
import pandas as pd
from flask_cors import CORS
from flask import jsonify


app = Flask(__name__)
CORS(app)

# 假设我们有这个数据文件
data = pd.read_excel(r"C:\Users\许馨元\Desktop\Web8\Web7\data\Final_score_year.xlsx")

# 生成ESG摘要函数
def generate_esg_summary_line(row):
    if 0 <= row['Score E'] < 4:
        e_description = "has low environmental performance, indicating limited efforts in environmental sustainability."
    elif 4 <= row['Score E'] < 7:
        e_description = "shows moderate environmental performance, reflecting some commitment to environmental issues."
    else:
        e_description = "demonstrates high environmental performance, showcasing strong sustainability initiatives."

    if 0 <= row['Score S'] < 4:
        s_description = "has low social responsibility performance, suggesting minimal focus on social welfare and workforce well-being."
    elif 4 <= row['Score S'] < 7:
        s_description = "shows moderate social responsibility performance, indicating some focus on social issues and employee welfare."
    else:
        s_description = "achieves high social responsibility performance, underlining strong commitment to community welfare and workforce development."
    
    if 0 <= row['Score G'] <= 2:
        g_description = "exhibits poor governance practices, possibly indicating areas for improvement in transparency and accountability."
    elif 2 < row['Score G'] <= 4:
        g_description = "has moderate governance performance, showing reasonable levels of transparency and risk management."
    else:
        g_description = "displays strong governance practices, reflecting a robust framework for accountability and ethical decision-making."

    if 0 <= row['Total ESG Score'] < 3:
        total_description = "has an overall low ESG performance, suggesting significant room for improvement in sustainability and governance practices."
    elif 3 <= row['Total ESG Score'] < 6:
        total_description = "achieves moderate overall ESG performance, indicating a balanced approach but with potential for further development in some areas."
    elif 6 <= row['Total ESG Score'] < 9:
        total_description = "shows good overall ESG performance, demonstrating solid efforts across environmental, social, and governance aspects."
    else:
        total_description = "achieves excellent overall ESG performance, standing out as a leader in sustainability and ethical practices."

    summary = (
        f"{row['Company Name']} ({row['Year']}): {e_description} "
        f"It {s_description} Additionally, it {g_description} Overall, the company {total_description}"
    )
    return summary


@app.route('/generate_esg_summary', methods=['GET'])
def generate_esg_summary():
    data['ESG Summary Line'] = data.apply(generate_esg_summary_line, axis=1)
    result = data[['Company Name', 'Year', 'ESG Summary Line']].to_dict(orient='records')
    return jsonify(result)

@app.route('/esg_summary_table')
def esg_summary_table():
    data['ESG Summary Line'] = data.apply(generate_esg_summary_line, axis=1)
    html_table = data[['Company Name', 'Year', 'ESG Summary Line']].to_html(index=False)
    
    html_template = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>ESG Summary Table</title>
        <style>
            table {{
                width: 100%;
                border-collapse: collapse;
            }}
            table, th, td {{
                border: 1px solid black;
                padding: 8px;
                text-align: left;
            }}
            th {{
                background-color: #f2f2f2;
            }}
        </style>
    </head>
    <body>
        <h1>ESG Summary Report</h1>
        {html_table}
    </body>
    </html>
    """
    return render_template_string(html_template)

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)

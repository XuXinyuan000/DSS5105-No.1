from flask import Flask, render_template_string, jsonify, send_file
import pandas as pd
from flask_cors import CORS
import matplotlib.pyplot as plt
from io import BytesIO
import subprocess
from extraction import df_pviot, score_final

app = Flask(__name__)
CORS(app)


def generate_description(row):
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

@app.route('/esg_chart')
def esg_chart():
    row = score_final.iloc[0]  

    plt.figure(figsize=(6, 4))
    esg_scores = [row['Score E'], row['Score S'], row['Score G']]
    categories = ['Score E', 'Score S', 'Score G']
    plt.bar(categories, esg_scores, color=['green', 'blue', 'purple'])
    plt.title(f"ESG Scores for {row['Company Name']} ({row['Year']})")
    plt.ylim(0, 10)
    
    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close()

    return send_file(img, mimetype='image/png')

@app.route('/esg_summary_table')
def esg_summary_table():
    score_final['ESG Summary Line'] = score_final.apply(generate_description, axis=1)
    html_table = score_final[['Company Name', 'Year', 'ESG Summary Line']].to_html(index=False)
    
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
        <h2>ESG Score Chart</h2>
        <img src="/esg_chart" alt="ESG Chart">
    </body>
    </html>
    """
    return render_template_string(html_template)

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)

subprocess.run(["npm", "run", "dev"])


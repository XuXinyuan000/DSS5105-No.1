from flask import Flask, render_template
from py.esg_description import get_esg_description  # 导入函数

app = Flask(__name__)

@app.route('/')
def index():
    # 示例分数
    e_score = 4.5  # Environmental score
    s_score = 6.0  # Social score
    g_score = 5.5  # Governance score

    # 调用函数生成描述
    description = get_esg_description(e_score, s_score, g_score)
    print(description)

    # 将描述传递给 HTML 模板
    return render_template('index.html', description=description)

if __name__ == "__main__":
    app.run(debug=True)






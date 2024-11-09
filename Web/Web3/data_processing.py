import pandas as pd
import matplotlib.pyplot as plt

# 更新为具体的 CSV 文件路径
data = pd.read_csv(r"D:\DSS\DSS5105\Web3\Data_Folder\Final_score.csv")

def generate_charts():
    categories = ['Score E', 'Score S', 'Score G']
    for category in categories:
        plt.figure(figsize=(10, 6))
        plt.plot(data['Company Name'], data[category], label=category)
        plt.title(f"{category} Score Over Time")
        plt.xlabel("Company Name")
        plt.ylabel("Score")
        plt.legend()
        plt.grid(True)
        plt.savefig(f'charts/score_{category[0].lower()}_chart.png')
        plt.close()

generate_charts()

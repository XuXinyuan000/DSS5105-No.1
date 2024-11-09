document.addEventListener("DOMContentLoaded", function () {
    Papa.parse("../data/Final_score.csv", {
        download: true,
        header: true,
        complete: function (results) {
            const data = results.data;

            // 从每一列提取数据
            const labels = data.map((_, index) => `Row ${index + 1}`);
            const columnA = data.map(row => parseFloat(row["Total ESG Score"]));
            const columnB = data.map(row => parseFloat(row["Score E"]));
            const columnC = data.map(row => parseFloat(row["Score S"]));
            const columnD = data.map(row => parseFloat(row["Score G"]));

            // 创建图表
            createChart("chart-a", labels, columnA, "Total ESG Score");
            createChart("chart-b", labels, columnB, "Environmental");
            createChart("chart-c", labels, columnC, "Social");
            createChart("chart-d", labels, columnD, "Governance");
        }
    });
});

// 使用 Chart.js 绘制图表的函数
function createChart(chartId, labels, data, label) {
    const ctx = document.getElementById(chartId).getContext("2d");
    new Chart(ctx, {
        type: "bar", // 选择图表类型
        data: {
            labels: labels,
            datasets: [{
                label: label,
                data: data,
                borderColor: "rgba(75, 192, 192, 1)",
                backgroundColor: "rgba(75, 192, 192, 0.2)",
                borderWidth: 2
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    title: {
                        display: true,
                        text: "Row"
                    }
                },
                y: {
                    title: {
                        display: true,
                        text: "Value"
                    }
                }
            }
        }
    });
}

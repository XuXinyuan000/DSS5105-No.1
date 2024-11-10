function fetchAndRenderChart(url, containerId) {
    fetch("http://127.0.0.1:8000" + url)  // 将 API 地址指向 Flask 应用
        .then(response => response.json())
        .then(data => {
            Plotly.newPlot(containerId, data.data, data.layout);
        })
        .catch(error => console.error('Error fetching chart data:', error));
}

// 调用示例
fetchAndRenderChart("/e_chart", "e-chart-container");
fetchAndRenderChart("/s_chart", "s-chart-container");
fetchAndRenderChart("/g_chart", "g-chart-container");

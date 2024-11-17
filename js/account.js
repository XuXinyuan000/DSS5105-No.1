// 确保 DOM 加载后执行代码
document.addEventListener('DOMContentLoaded', function () {
    // 获取图表容器
    var chart = echarts.init(document.getElementById('account'));

    // 配置环形图
    var option = {
        tooltip: {
            trigger: 'item',
            formatter: '{b}: {c} ({d}%)' // 显示名称、数值和百分比
        },
        series: [
            {
                name: 'ESG Breakdown',
                type: 'pie',
                radius: ['50%', '70%'], // 设置内外半径形成环形
                avoidLabelOverlap: false,
                label: {
                    show: false, // 隐藏默认的标签
                    position: 'center'
                },
                emphasis: {
                    label: {
                        show: false,
                        fontSize: '20',
                        fontWeight: 'bold'
                    }
                },
                labelLine: {
                    show: false // 不显示引导线
                },
                data: [
                    { value: 40, name: 'Environmental (E)', itemStyle: { color: '#6a5acd' } }, // 紫色
                    { value: 30, name: 'Social (S)', itemStyle: { color: '#87ceeb' } },       // 浅蓝色
                    { value: 30, name: 'Governance (G)', itemStyle: { color: '#32cd32' } }  // 浅绿色
                ]
            }
        ],
        graphic: {
            type: 'text',
            left: 'center',
            top: 'center',
            style: {
                text: 'ESG\n40%-30%-30%',
                textAlign: 'center',
                fontSize: 20,
                fontWeight: 'bold',
                fill: '#000' // 字体颜色
            }
        }
    };

    // 设置配置并渲染图表
    chart.setOption(option);
});

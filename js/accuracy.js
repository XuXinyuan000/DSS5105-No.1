// 确保 DOM 加载后执行代码
document.addEventListener('DOMContentLoaded', function () {
    // 获取图表容器
    var chart = echarts.init(document.getElementById('accuracyChart'));

    // 配置环形图
    var option = {
        tooltip: {
            trigger: 'item',
            formatter: '{b}: {c} ({d}%)' // 显示名称、数值和百分比
        },
        series: [
            {
                name: 'Accuracy',
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
                    { value: 84.72, name: 'Valid', itemStyle: { color: '#9370DB' } }, // 紫色部分 (HEX 为 #9370DB)
                    { value: 15.28, name: 'Invalid', itemStyle: { color: '#d3d3d3' } } // 灰色部分
                ]
            }
        ],
        graphic: {
            type: 'text',
            left: 'center',
            top: 'center',
            style: {
                text: 'Accuracy\n84.72%',
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

$(function () {
    ceshis();

    function ceshis() {
        var myChart = echarts.init(document.getElementById('chart2'));

        var option = {
            tooltip: {
                trigger: 'axis'
            },
            toolbox: {
                show: true,
                feature: {
                    mark: { show: true },
                    dataView: { show: true, readOnly: false },
                    magicType: { show: true, type: ['line', 'bar'] },
                    restore: { show: true },
                    saveAsImage: { show: true }
                }
            },
            grid: {
                top: '10%',
                left: '3%',
                right: '4%',
                bottom: '3%',
                containLabel: true
            },
            legend: {
                data: ['E Score'],
                textStyle: {
                    color: "#000", // 图例字体颜色改为黑色
                    fontSize: 12
                }
            },
            xAxis: [
                {
                    type: 'category',
                    data: [
                        'AIA Group', 'Allianz', 'Barclays', 'BlackRock', 
                        'China Life Insurance', 'Citibank', 'DBS', 'HSBC', 
                        'JPMorgan Chase', 'MUFG', 'Metlife', 'Nippon Life'
                    ],
                    axisLabel: {
                        show: true,
                        textStyle: {
                            color: "#000", // 横坐标字体颜色改为黑色
                            fontSize: 12
                        },
                        rotate: 90 // 文字倾斜，避免重叠
                    },
                    axisLine: {
                        lineStyle: {
                            color: '#000' // 横坐标轴线颜色改为黑色
                        }
                    }
                }
            ],
            yAxis: [
                {
                    type: 'value',
                    name: 'E Score',
                    axisLabel: {
                        formatter: '{value} pts',
                        textStyle: {
                            color: "#000", // 纵坐标字体颜色改为黑色
                            fontSize: 12
                        }
                    },
                    axisLine: {
                        lineStyle: {
                            color: '#000' // 纵坐标轴线颜色改为黑色
                        }
                    }
                }
            ],
            series: [
                {
                    name: 'E Score',
                    type: 'bar',
                    data: [10, 10, 10, 2, 5, 9, 10, 7, 9, 9, 10, 9],
                    itemStyle: {
                        normal: {
                            barBorderRadius: 5,
                            color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
                                { offset: 0, color: "#FFA500" },
                                { offset: 1, color: "#800080" }
                            ])
                        }
                    }
                }
            ]
        };

        myChart.setOption(option);
    }
});

$(function () {
    ceshis();

    function ceshis() {
        var myChart = echarts.init(document.getElementById('chart4'));

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
                data: ['G Score'],
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
                    name: 'G Score',
                    min: 0, // 设置纵坐标的最小值
                    max: 10,
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
                    name: 'G Score',
                    type: 'bar',
                    data: [4, 4, 3, 3, 3, 4, 4, 3, 2, 2, 3, 2],
                    itemStyle: {
                        normal: {
                            barBorderRadius: 5,
                            color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
                                { offset: 0, color: "#FFC0CB" },
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

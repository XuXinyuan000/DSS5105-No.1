$(function () {
    ceshis();

    function ceshis() {
        var myChart = echarts.init(document.getElementById('heatmap'));

        var option = {
            tooltip: {
                trigger: 'item',
                formatter: function (params) {
                    return `Correlation: ${params.value[2]}`;
                }
            },
            toolbox: {
                show: true,
                feature: {
                    saveAsImage: { show: true }
                }
            },
            grid: {
                left: '10%',
                right: '10%',
                bottom: '15%',
                containLabel: true
            },
            xAxis: {
                type: 'category',
                data: ['ROE', 'Net profit margin', 'Asset-liability ratio', 'PE'],
                axisLabel: {
                    textStyle: {
                        color: "#000",
                        fontSize: 12
                    }
                },
                axisLine: {
                    lineStyle: {
                        color: '#000'
                    }
                }
            },
            yAxis: {
                type: 'category',
                data: ['Score E', 'Score S', 'Score G', 'Total ESG Score'],
                axisLabel: {
                    textStyle: {
                        color: "#000",
                        fontSize: 12
                    }
                },
                axisLine: {
                    lineStyle: {
                        color: '#000'
                    }
                }
            },
            visualMap: {
                min: -1,
                max: 1,
                calculable: true,
                orient: 'horizontal',
                left: 'center',
                bottom: '5%',
                inRange: {
                    color: ['#ff99cc', '#f2f2b3', '#91d7c6', '#667af0'] // 从柔和粉到深紫
                }
            },
            series: [{
                name: 'Correlation Heatmap',
                type: 'heatmap',
                data: [
                    [0, 0, 0.22], [0, 1, 0.28], [0, 2, -0.07], [0, 3, -0.16],
                    [1, 0, -0.34], [1, 1, -0.62], [1, 2, 0.18], [1, 3, 0.59],
                    [2, 0, -0.03], [2, 1, -0.17], [2, 2, 0.22], [2, 3, 0.39],
                    [3, 0, -0.32], [3, 1, -0.25], [3, 2, 0.54], [3, 3, -0.19]
                ],
                label: {
                    show: true,
                    formatter: function (params) {
                        return params.value[2].toFixed(2); // 显示相关性数值
                    },
                    textStyle: {
                        color: "#000"
                    }
                },
                emphasis: {
                    itemStyle: {
                        shadowBlur: 10,
                        shadowColor: 'rgba(0, 0, 0, 0.5)'
                    }
                }
            }]
        };

        myChart.setOption(option);

        // 调试信息
        console.log('Updated heatmap with simplified correlation labels.');
    }
});

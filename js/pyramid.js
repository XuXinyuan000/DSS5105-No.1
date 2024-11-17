$(function () {
    ceshis();

    function ceshis() {
        var myChart = echarts.init(document.getElementById('chart5'));

        var option = {
            tooltip: {
                trigger: 'axis',
                axisPointer: {
                    type: 'shadow'
                }
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
                right: '10%',
                bottom: '10%',
                containLabel: true
            },
            xAxis: {
                type: 'value',
                name: '',
                axisLabel: {
                    textStyle: {
                        color: "#000", // 坐标轴颜色改为黑色
                        fontSize: 12
                    }
                },
                axisLine: {
                    lineStyle: {
                        color: '#000'
                    }
                },
                splitLine: {
                    lineStyle: {
                        type: 'dashed',
                        color: '#ccc' // 网格线颜色
                    }
                }
            },
            yAxis: {
                type: 'category',
                data: ['AAA', 'AA', 'A', 'BBB', 'BB', 'B', 'CCC'], // 金字塔等级
                axisLabel: {
                    textStyle: {
                        color: "#000", // Y轴文字颜色改为黑色
                        fontSize: 12
                    }
                },
                axisLine: {
                    lineStyle: {
                        color: '#000'
                    }
                }
            },
            series: [
                {
                    name: 'Score',
                    type: 'bar',
                    data: [10, 8.571, 7.143, 5.714, 4.286, 2.857, 1.429], // 对应每个等级的分数
                    itemStyle: {
                        normal: {
                            barBorderRadius: [0, 10, 10, 0], // 右侧圆角
                            color: function (params) {
                                // 动态设置颜色，根据数据顺序设置渐变色
                                const colorList = [
                                    new echarts.graphic.LinearGradient(0, 0, 1, 0, [
                                        { offset: 0, color: "#B8E986" }, // AAA: 浅绿色
                                        { offset: 1, color: "#72C472" }
                                    ]),
                                    new echarts.graphic.LinearGradient(0, 0, 1, 0, [
                                        { offset: 0, color: "#A3D891" }, // AA
                                        { offset: 1, color: "#5EA85A" }
                                    ]),
                                    new echarts.graphic.LinearGradient(0, 0, 1, 0, [
                                        { offset: 0, color: "#FDE68A" }, // A: 黄色
                                        { offset: 1, color: "#D4B259" }
                                    ]),
                                    new echarts.graphic.LinearGradient(0, 0, 1, 0, [
                                        { offset: 0, color: "#FDC184" }, // BBB: 橙色
                                        { offset: 1, color: "#F8A055" }
                                    ]),
                                    new echarts.graphic.LinearGradient(0, 0, 1, 0, [
                                        { offset: 0, color: "#F5B8A1" }, // BB: 浅红
                                        { offset: 1, color: "#D77A5A" }
                                    ]),
                                    new echarts.graphic.LinearGradient(0, 0, 1, 0, [
                                        { offset: 0, color: "#F7A3A3" }, // B: 红色
                                        { offset: 1, color: "#D16969" }
                                    ]),
                                    new echarts.graphic.LinearGradient(0, 0, 1, 0, [
                                        { offset: 0, color: "#E38181" }, // CCC: 深红
                                        { offset: 1, color: "#C94A4A" }
                                    ])
                                ];
                                return colorList[params.dataIndex]; // 根据数据索引返回对应的渐变色
                            }
                        }
                    },
                    barWidth: '40%' // 设置柱状图宽度
                }
            ]
        };

        myChart.setOption(option);
    }
});

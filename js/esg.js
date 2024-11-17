$(function () {
    ceshis();

    function ceshis() {
        var myChart = echarts.init(document.getElementById('esg'));

        var option = {
            tooltip: {
                trigger: 'axis',
                axisPointer: {
                    type: 'shadow' // 鼠标悬停显示阴影效果
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
            legend: {
                data: ['ESG Scores', 'Total ESG Score'],
                top: 10,
                right: 'center',
                textStyle: {
                    color: "#000",
                    fontSize: 12
                },
                top: '5%', // 图例位置调整到顶部
                itemGap: 20, // 图例之间的间距
                orient: 'horizontal', // 水平布局
                align: 'auto', // 自动对齐
                right: 'center' // 水平居中
            },
            grid: {
                left: '3%',
                right: '20%', // 预留空间给表格
                bottom: '3%',
                containLabel: true
            },
            xAxis: {
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
                },
                boundaryGap: true // 使柱状图完全居中
            },
            yAxis: {
                type: 'value',
                name: 'Scores',
                axisLabel: {
                    formatter: '{value}',
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
            series: [
                {
                    name: 'ESG Scores',
                    type: 'bar',
                    data: [10, 2, 6, null],
                    itemStyle: {
                        normal: {
                            barBorderRadius: 5,
                            color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
                                { offset: 0, color: "#FFC0CB" },
                                { offset: 1, color: "#800080" }
                            ])
                        }
                    },
                    barWidth: '30%'
                },
                {
                    name: 'Total ESG Score',
                    type: 'bar',
                    data: [null, null, null, 6.4],
                    itemStyle: {
                        normal: {
                            barBorderRadius: 5,
                            color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
                                { offset: 0, color: "#FFA500" },
                                { offset: 1, color: "#4693EC" }
                            ])
                        }
                    },
                    barWidth: '30%'
                },
                {
                    name: 'Benchmark (Total)',
                    type: 'line',
                    markLine: {
                        data: [
                            {
                                yAxis: 5.4,
                                name: 'Benchmark Total'
                            }
                        ],
                        lineStyle: {
                            type: 'dashed',
                            color: '#FF4500'
                        },
                        label: {
                            formatter: 'Benchmark Total: {c}',
                            position: 'middle',
                            textStyle: {
                                color: '#FF4500',
                                fontSize: 12
                            },
                            distance: 10
                        }
                    }
                }
            ],
            graphic: [
                {
                    type: 'group',
                    right: '2%', // 表格距离右侧的距离
                    top: '10%', // 表格距离顶部的距离
                    z: 100,
                    children: [
                        {
                            type: 'rect',
                            shape: { width: 180, height: 120 },
                            style: {
                                fill: '#F8F8F8',
                                stroke: '#000',
                                lineWidth: 1,
                                shadowBlur: 5,
                                shadowColor: 'rgba(0, 0, 0, 0.5)',
                                shadowOffsetX: 3,
                                shadowOffsetY: 3
                            }
                        },
                        {
                            type: 'text',
                            left: 10,
                            top: 10,
                            style: {
                                text: 'Benchmark Scores\nE: 8.4375\nS: 2.6875\nG: 4.0625\nTotal: 5.4',
                                font: '14px Arial',
                                fill: '#000',
                                textAlign: 'left',
                                textVerticalAlign: 'top',
                                lineHeight: 20
                            }
                        }
                    ]
                }
            ]
        };

        myChart.setOption(option);
    }
});

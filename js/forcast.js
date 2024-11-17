$(function () {
    ceshis();

    function ceshis() {
        var myChart = echarts.init(document.getElementById('chart6'));

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
                    dataView: { show: true, readOnly: false }, // 数据视图
                    magicType: { show: true, type: ['line', 'bar'] }, // 切换为折线图或柱状图
                    restore: { show: true }, // 还原
                    saveAsImage: { show: true } // 保存为图片
                }
            },
            legend: {
                data: ['Total ESG Score'],
                textStyle: {
                    color: "#000", // 图例字体颜色为黑色
                    fontSize: 12
                }
            },
            grid: {
                left: '3%',
                right: '4%',
                bottom: '3%',
                containLabel: true
            },
            xAxis: {
                type: 'category',
                data: ['2015', '2016', '2017', '2018', '2019', '2020', '2021', '2022', '2023', '2024', '2025', '2026'],
                axisLabel: {
                    textStyle: {
                        color: "#000", // 横坐标字体颜色为黑色
                        fontSize: 12
                    }
                },
                axisLine: {
                    lineStyle: {
                        color: '#000' // 横坐标轴线颜色为黑色
                    }
                }
            },
            yAxis: {
                type: 'value',
                name: 'Total ESG Score',
                axisLabel: {
                    formatter: '{value}', // 显示分数
                    textStyle: {
                        color: "#000", // 纵坐标字体颜色为黑色
                        fontSize: 12
                    }
                },
                axisLine: {
                    lineStyle: {
                        color: '#000' // 纵坐标轴线颜色为黑色
                    }
                }
            },
            series: [
                {
                    name: 'Total ESG Score',
                    type: 'bar',
                    data: [4.4, 4.4, 4.0, 4.4, 4.4, 4.4, 4.8, 4.8, 6.4, 6.6, 7.2, 7.8],
                    itemStyle: {
                        normal: {
                            barBorderRadius: 5, // 设置柱状图圆角
                            color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
                                { offset: 0, color: "#7FFF00" }, // 起始颜色
                                { offset: 1, color: "#4682B4" }  // 结束颜色
                            ])
                        }
                    },
                    barWidth: '40%' // 设置柱宽度
                }
            ]
        };

        myChart.setOption(option);

        // 调试：打印加载状态
        console.log('Chart loaded with toolbox functionality.');
    }
});



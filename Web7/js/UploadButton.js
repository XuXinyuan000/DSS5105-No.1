// scripts.js
document.addEventListener('DOMContentLoaded', function () {
    // 获取上传按钮和隐藏的文件输入框
    const uploadBtn = document.querySelector('.upload-button');
    const fileInput = document.getElementById('fileInput');

    // 点击上传按钮时，触发文件选择框的点击事件
    uploadBtn.addEventListener('click', function (event) {
        console.log(111)
        event.preventDefault(); // 防止页面跳转
        fileInput.click(); // 触发文件选择框
    });

    // 监听文件选择框的变化，获取选中的文件
    fileInput.addEventListener('change', function () {
        if (fileInput.files.length > 0) {
            alert('你选择了文件: ' + fileInput.files[0].name);
        }
    });
});


document.getElementById("fileInput").addEventListener("change", function (event) {
    const file = event.target.files[0];
    if (file && file.type === "application/pdf") {
      // 显示 description、score 和 accuracy 的框
      document.getElementById("descriptionBox").style.display = "block";
      document.getElementById("scoreBox").style.display = "block";
      document.getElementById("accuracyBox").style.display = "block";
  
      // 模拟文件处理后的内容，您可以根据实际需要进行替换
      document.getElementById("descriptionContent").textContent = "AIA Group in 2023 had an ESG score of 7, with strong environmental and moderate governance performance...Allianz maintained a strong rating of AA with balanced scores across E, S, and G";
      document.getElementById("scoreContent").textContent = "文件评分: 85";
      document.getElementById("accuracyContent").textContent = "识别准确率: 92%";
  
      // 如果您有实际的文件上传处理，可以在这里添加异步请求和响应逻辑
    } else {
      alert("place upload the pdf");
    }
  });
  
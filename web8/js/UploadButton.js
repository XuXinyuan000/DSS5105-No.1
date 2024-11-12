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
      document.getElementById("descriptionBox").style.display = "block";
      document.getElementById("scoreBox").style.display = "block";
      document.getElementById("accuracyBox").style.display = "block";
      document.getElementById("descriptionIframeContainer").style.display = "block";  // 显示iframe

      // 发起fetch请求生成ESG概要
      fetch('http://127.0.0.1:5000/generate_esg_summary')
          .then(response => response.json())
          .then(data => {
              document.getElementById("descriptionContent").textContent = data.description;
              document.getElementById("scoreContent").textContent = "文件评分: " + data.score;
              document.getElementById("accuracyContent").textContent = "识别准确率: " + data.accuracy;
          })
          .catch(error => console.error('Error fetching ESG summary:', error));
  } else {
      alert("请上传PDF文件");
  }
});

  
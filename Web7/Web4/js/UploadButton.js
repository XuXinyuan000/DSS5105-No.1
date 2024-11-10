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

// 确保 toggleContent.js 包含以下代码
function showMainContent() {
    document.getElementById('mainContent').style.display = 'block';
    document.getElementById('container').style.display = 'none';
    document.getElementById('uploadContent').style.display = 'none';
}

function showAboutUsContent() {
    document.getElementById('mainContent').style.display = 'none';
    document.getElementById('container').style.display = 'flex';
    document.getElementById('uploadContent').style.display = 'none';
}

function showUploadContent() {
    document.getElementById('mainContent').style.display = 'none';
    document.getElementById('container').style.display = 'none';
    document.getElementById('uploadContent').style.display = 'block';
    console.log(222)
}

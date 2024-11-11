fetch('http://127.0.0.1:5000/generate_esg_summary')
    .then(response => response.json())
    .then(data => {
        document.getElementById("descriptionContent").textContent = data.description;
    })
    .catch(error => console.error('Error fetching description:', error));

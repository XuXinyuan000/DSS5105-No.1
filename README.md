## Automated ESG Data Extraction and Performance Analysis
## Project Overview
This project is an automated tool for ESG (Environmental, Social, and Governance) data extraction, analysis, and prediction. By uploading ESG report PDFs, the system automatically extracts key ESG indicators, generates analysis results, and predicts future ESG performance.

## Key Features
- Automated PDF document processing and text extraction
- ESG key indicator identification and value extraction
- Automated ESG scoring calculation
- ESG metrics trend prediction
- Data visualization
- Intuitive web interface

## Technical Architecture
### Frontend
- Web interface for PDF file upload and result display
- Interactive data visualization components
- Prediction model result display
### Backend
- PDF text processing engine
- ESG data extraction algorithm
- Scoring system
- Machine learning prediction models

## Installation Guide
### Backend Setup
- Clone the repository
  ```python
  git clone https://github.com/XuXinyuan000/DSS5105-No.1.git
  cd DSS5105-No.1/backend
  ```
- Install dependencies
  ```python
  pip install -r requirements_backend.txt
  ```
- Start the backend server
  ```python
  python app.py
  ```
### Frontend Setup
- Navigate to frontend directory
  ```python
  cd DSS5105-No.1/backend
  ``` 
- Start development server
  ```python
  npm start
  ``` 
  
## Usage Instructions
- Open your browser and visit http://localhost:3000
- Click the upload button to select an ESG report PDF file
- The system will automatically process the document and generate three main outputs:
  - ESG metrics data table (including indicator names and corresponding values)
  - ESG comprehensive score
  - ESG metrics prediction analysis

## Project Structure
  ```python
 project/
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── Upload/
│   │   │   ├── Analysis/
│   │   │   └── Prediction/
│   │   ├── pages/
│   │   └── utils/
│   ├── public/
│   └── package.json
├── backend/
│   ├── app.py
│   ├── models/
│   │   ├── extraction/
│   │   ├── description/
│   │   └── prediction/
│   └── utils/
└── README.md
  ```

# 🌐 Marathi Hate Speech Subclass Classification Web App


A Flask-based web application that detects hate speech in Marathi text using advanced BiLSTM deep learning models.
It supports both single-text analysis and bulk CSV file classification, with subclass categorization for types of hate speech.
  

---

## ✨ Key Highlights
- ✅ **Detects whether Marathi text contains Hate Speech or Not Hate Speech**

- **✅ Predicts the subclass/type of hate speech (e.g., Insulting, Harassing, Religious Intolerance, etc.)**
- ✅**Supports real-time text input and CSV file upload for batch processing**
- ✅ Automatically cleans and preprocesses Marathi text
- ✅ Generates downloadable classified CSV files with predictions
- ✅ Built using Flask, TensorFlow/Keras, NLTK, and Pandas
- ⚡ **Real-time prediction** for single Marathi text inputs  
- 📁 **Bulk CSV classification** with downloadable output  
- 🧠 **BiLSTM-based deep learning models** for robust results  
- 🖥️ **Simple, elegant web UI** powered by Flask templates  

---


## 🧠 Subclass Mapping

| 🆔 **ID** | 🏷️ **Subclass Label** |
|------------|-----------------------|
| 0 | Not Hate Speech |
| 1 | Insulting |
| 2 | Religious Intolerance |
| 3 | Harassing |
| 4 | Gender Abusive |


## 🧩 Project Structure

```bash
🏗️ Project Structure
Marathi_Hate_Speech_Detector/
│
├── app.py                      # Main Flask application file
├── models/                     # Pre-trained models and tokenizer
│   ├── bilstm_tokenizer_final.pkl
│   ├── best_bilstm_label_model_final.keras
│   ├── best_bilstm_subclass_model_final.keras
│   └── marathi_stopwords.txt
│
├── templates/
│   └── index.html              # Front-end UI template
├── Dataset/ 
│
├── uploads/                    # Stores uploaded and processed CSV files
│
├── static/                     #  For CSS, JS, images
│
└── README.md                   # Project documentation
```

## ⚙️ Installation & Setup

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/<your-username>/marathi-hate-speech-detector.git
cd marathi-hate-speech-detector
```

### 2️⃣ Create a Virtual Environment
```baash
python -m venv venv
source venv/bin/activate       # On Mac/Linux
venv\Scripts\activate          # On Windows
```

### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
If you don’t have a requirements.txt, you can install the libraries manually:
pip install flask tensorflow pandas numpy nltk werkzeug
```
 
### 4️⃣ Setup NLTK Stopwords
```bash
The app automatically downloads stopwords if missing.
If needed, you can download manually:
python -m nltk.downloader stopwords
```

### 5️⃣ Place Model Files
```bash
Ensure the following files are in the models/ directory:
bilstm_tokenizer_final.pkl
best_bilstm_label_model_final.keras
best_bilstm_subclass_model_final.keras
marathi_stopwords.txt
```
### 🚀 Run the Application
```bash
Start the Flask App
python app.py
```

### Open in Browser
```bash
Visit:
👉 http://localhost:5000
```
### 3 💻 Usage

### 3 🗣️ Single Text Classification
1. Enter a Marathi sentence in the text box.
2. Click **Analyze**.
3. The system will display:
   - **Hate Speech label**
   - **Confidence probability**
   - **Subclass prediction**

### 📁 CSV File Classification
1. Upload a `.csv` file containing a **"text"** column.
2. The app will classify each row and add the following columns:
   - `Predicted_Label`
   - `Hate_Speech_Probability`
   - `Predicted_Subclass`
3. A downloadable CSV file is generated automatically.

### 📊 Example Output (CSV)

| text                       | Predicted_Label | Hate_Speech_Probability | Predicted_Subclass |
|----------------------------|----------------|------------------------|------------------|
| ही पोस्ट वाईट आहे          | Hate Speech    | 0.8975                 | Insulting        |
| मला सर्व धर्मांचा आदर आहे | Not Hate Speech| 0.1221                 | Not Hate Speech  |

### 🧩 Technologies Used

| Category         | Tools/Frameworks                     |
|-----------------|------------------------------------|
| Frontend        | HTML, CSS (Flask Jinja Template)   |
| Backend         | Flask (Python)                     |
| Machine Learning| TensorFlow / Keras, BiLSTM         |
| Text Preprocessing | NLTK, Regular Expressions        |
| Data Handling   | Pandas, NumPy                      |

### 3 🧠 Model Overview

The project uses **BiLSTM (Bidirectional LSTM)** models for contextual understanding of Marathi language.  

Two models are used:

1. `best_bilstm_label_model_final.keras` → Binary classification (**Hate / Not Hate**)  
2. `best_bilstm_subclass_model_final.keras` → Multi-class classification (**subtypes**)



---
### Linkedin
```bash
https://www.linkedin.com/in/nirajbhagat7803/
```
### Email
```bash
nirajbhagat7803@gmail.com
```

---
### Screenshots of UI
<img width="1911" height="1024" alt="image" src="https://github.com/user-attachments/assets/96966a4c-98db-45fc-b962-5187131b802c" />

---

<img width="1918" height="1009" alt="image" src="https://github.com/user-attachments/assets/576a64c6-16c6-4cce-b795-efada2afada4" />

--- 

<img width="1916" height="1022" alt="image" src="https://github.com/user-attachments/assets/509b74ca-d2cb-417c-8cb4-ff6f20b83551" />

---

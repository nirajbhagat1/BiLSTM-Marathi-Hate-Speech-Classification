🏷️ Marathi Hate Speech Detection Web App

A Flask-based web application that detects hate speech in Marathi text using advanced BiLSTM deep learning models.
It supports both single-text analysis and bulk CSV file classification, with subclass categorization for types of hate speech.

📌 Features

✅ Detects whether Marathi text contains Hate Speech or Not Hate Speech
✅ Predicts the subclass/type of hate speech (e.g., Insulting, Harassing, Religious Intolerance, etc.)
✅ Supports real-time text input and CSV file upload for batch processing
✅ Automatically cleans and preprocesses Marathi text
✅ Generates downloadable classified CSV files with predictions
✅ Built using Flask, TensorFlow/Keras, NLTK, and Pandas

🧠 Subclass Mapping
ID	Subclass Label
0	Not Hate Speech
1	Insulting
2	Religious Intolerance
3	Harassing
4	Gender Abusive

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
│
├── uploads/                    # Stores uploaded and processed CSV files
│
├── static/                     # (Optional) For CSS, JS, images
│
└── README.md                   # Project documentation

⚙️ Installation Guide
1️⃣ Clone the Repository
git clone https://github.com/<your-username>/marathi-hate-speech-detector.git
cd marathi-hate-speech-detector

2️⃣ Create a Virtual Environment
python -m venv venv
source venv/bin/activate       # On Mac/Linux
venv\Scripts\activate          # On Windows

3️⃣ Install Dependencies
pip install -r requirements.txt
If you don’t have a requirements.txt, you can install the libraries manually:
pip install flask tensorflow pandas numpy nltk werkzeug

4️⃣ Setup NLTK Stopwords
The app automatically downloads stopwords if missing.
If needed, you can download manually:

python -m nltk.downloader stopwords

5️⃣ Place Model Files

Ensure the following files are in the models/ directory:
bilstm_tokenizer_final.pkl
best_bilstm_label_model_final.keras
best_bilstm_subclass_model_final.keras
marathi_stopwords.txt

🚀 Run the Application
Start the Flask App
python app.py

Open in Browser

Visit:
👉 http://localhost:5000

💻 Usage
🗣️ Single Text Classification

Enter a Marathi sentence in the text box
Click Analyze
The system shows:
Hate Speech label
Confidence probability
Subclass prediction

📁 CSV File Classification

Upload a .csv file containing a "text" column
The app will classify each row and add:
Predicted_Label_ID
Predicted_Label
Hate_Speech_Probability
Predicted_Subclass_ID
Predicted_Subclass
A downloadable CSV file is generated automatically.

📊 Example Output (CSV)
text	Predicted_Label	Hate_Speech_Probability	Predicted_Subclass
ही पोस्ट वाईट आहे	Hate Speech	0.8975	Insulting
मला सर्व धर्मांचा आदर आहे	Not Hate Speech	0.1221	Not Hate Speech
🧩 Technologies Used
Category	Tools/Frameworks
Frontend	HTML, CSS (Flask Jinja Template)
Backend	Flask (Python)
Machine Learning	TensorFlow / Keras, BiLSTM
Text Preprocessing	NLTK, Regular Expressions
Data Handling	Pandas, NumPy
🧠 Model Overview

The project uses:

BiLSTM (Bidirectional LSTM) models for contextual understanding of Marathi language

Two models:

best_bilstm_label_model_final.keras → Binary classification (Hate/Not Hate)
best_bilstm_subclass_model_final.keras → Multi-class classification (subtypes)

⚠️ Notes
Ensure that uploaded CSV files contain a column named text.
Maximum allowed file size: 16 MB
Non-Marathi text may produce unpredictable results.
👨‍💻 Authors
Team Members:
Niraj Bhagat
Rohit Katkar
Nidhi Patil
Gitesh Mitkar

🏁 Future Enhancements
Add visualization charts (e.g., subclass distribution pie chart)
Integrate with APIs for real-time social media comment analysis
Expand dataset for more hate speech categories
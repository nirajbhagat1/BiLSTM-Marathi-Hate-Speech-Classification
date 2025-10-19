📘 Marathi Hate Speech Detection Dataset
🧩 Overview

This dataset is designed for Marathi Hate Speech Detection and Classification, focusing on binary classification (Hate Speech or Not) and multi-class subclass categorization (specific hate types).
It contains 13,037 text samples collected and annotated for research and model development in Natural Language Processing (NLP).

📊 Data Summary
Feature	Description
Text	Marathi text comment or statement
Label	Binary label indicating hate speech (1) or not (0)
Subclass	Numeric label representing specific hate type (0–4)
🔍 Data Overview & Preprocessing
✅ Missing Values (Before Handling)
Text        0
Label       0
Subclass    0
dtype: int64


No missing values detected in essential columns.
Dropped 0 rows during preprocessing.
Final shape: (13037, 3)

📈 Data Distribution
Label Distribution
Label	Description	Count	Percentage
1	Hate Speech	7112	54.55%
0	Not Hate Speech	5925	45.45%
Subclass Distribution
Subclass	Description	Count	Percentage
0	Not Hate Speech	5925	45.45%
1	Insulting	1813	13.91%
2	Religious Intolerance	1342	10.29%
3	Harassing	2114	16.22%
4	Gender Abusive	1843	14.14%

⚙️ Usage
This dataset can be used for:
Binary Classification: Hate vs. Not Hate
Multi-class Classification: Specific hate type identification
Text preprocessing experiments: Tokenization, lemmatization, stopword removal, etc.
Model training: Logistic Regression, LSTM, or Ensemble methods

🧠 Example Use Case

Researchers and developers can use this dataset to:
Build Marathi hate speech detection systems.
Study linguistic patterns in hate speech.
Improve social media content moderation models.

🪪 Citation

If you use this dataset in your work, please cite:
“Marathi Hate Speech Detection Dataset (2025)” — Created by Niraj Bhagat and team.
Linkedin : https://www.linkedin.com/in/nirajbhagat7803/
mail: nirajbhagat7803@gmail.com
import os
import re
import pickle
import warnings
import numpy as np
import pandas as pd # For CSV handling
import nltk

# --- FIX: Changed the exception from nltk.downloader.DownloadError to LookupError ---
# This block now correctly handles the case where the 'stopwords' resource is not found.
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    print("NLTK stopwords not found. Downloading...")
    nltk.download('stopwords', quiet=True)

from nltk.corpus import stopwords
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from flask import (
    Flask, request, render_template, flash,
    redirect, url_for, send_file # For CSV handling
)
from werkzeug.utils import secure_filename # For secure file handling

# --- Configuration & Setup ---

# Suppress TensorFlow/Keras warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
warnings.filterwarnings('ignore', category=FutureWarning) # General future warning ignore

# --- Constants ---
MODELS_DIR = 'models' # Define the directory name
TOKENIZER_PATH = os.path.join(MODELS_DIR, 'bilstm_tokenizer_final.pkl')
LABEL_MODEL_PATH = os.path.join(MODELS_DIR, 'best_bilstm_label_model_final.keras')
SUBCLASS_MODEL_PATH = os.path.join(MODELS_DIR, 'best_bilstm_subclass_model_final.keras')
STOPWORDS_PATH = os.path.join(MODELS_DIR, 'marathi_stopwords.txt')

UPLOAD_FOLDER = 'uploads' # Folder to store uploaded/processed CSVs
ALLOWED_EXTENSIONS = {'csv'} # Allowed file extensions

MAX_SEQUENCE_LENGTH = 100 # Must match the value used during training
BATCH_SIZE = 64 # Defined globally, mainly for reference/training consistency

# Define subclass mapping
sublabel_mapping = {
    0: "Not Hate Speech", 1: "Insulting", 2: "Religious Intolerance",
    3: "Harassing", 4: "Gender Abusive",
}
num_subclasses = len(sublabel_mapping)

# --- Create Upload Folder if it doesn't exist ---
if not os.path.exists(UPLOAD_FOLDER):
    try:
        os.makedirs(UPLOAD_FOLDER)
        print(f"Created upload folder: {UPLOAD_FOLDER}")
    except OSError as e:
        print(f"Error creating upload folder {UPLOAD_FOLDER}: {e}")
        exit()

# --- Load Models and Tokenizer (Load once at startup) ---
print("Loading models and tokenizer...")
if not os.path.exists(MODELS_DIR):
    print(f"Error: Models directory '{MODELS_DIR}' not found.")
    exit()
try:
    if not os.path.exists(TOKENIZER_PATH): raise FileNotFoundError(f"Tokenizer not found at {TOKENIZER_PATH}")
    with open(TOKENIZER_PATH, 'rb') as f:
        tokenizer = pickle.load(f)
    print(f"Tokenizer loaded successfully from {TOKENIZER_PATH}")

    if not os.path.exists(LABEL_MODEL_PATH): raise FileNotFoundError(f"Label model not found at {LABEL_MODEL_PATH}")
    model_label = load_model(LABEL_MODEL_PATH)
    print(f"Label model loaded successfully from {LABEL_MODEL_PATH}")

    if not os.path.exists(SUBCLASS_MODEL_PATH): raise FileNotFoundError(f"Subclass model not found at {SUBCLASS_MODEL_PATH}")
    model_subclass = load_model(SUBCLASS_MODEL_PATH)
    print(f"Subclass model loaded successfully from {SUBCLASS_MODEL_PATH}")

except FileNotFoundError as e:
    print(f"Error loading file: {e}.")
    exit()
except Exception as e:
    print(f"An unexpected error occurred during model/tokenizer loading: {e}")
    exit()

# --- Load Stopwords ---
print("Loading stopwords...")
try:
    if not os.path.exists(STOPWORDS_PATH): raise FileNotFoundError(f"Stopwords file not found at {STOPWORDS_PATH}")
    with open(STOPWORDS_PATH, 'r', encoding='utf-8') as f:
        marathi_stopwords = set(line.strip() for line in f if line.strip())
    print(f"Successfully loaded {len(marathi_stopwords)} stopwords from {STOPWORDS_PATH}.")
except FileNotFoundError as e:
    print(f"Warning: {e}. Stopword removal will be skipped.")
    marathi_stopwords = set()
except Exception as e:
     print(f"An error occurred loading stopwords: {e}")
     marathi_stopwords = set()


# --- Preprocessing Functions ---
def clean_marathi_text(text):
    """Cleans Marathi text."""
    if not isinstance(text, str): return ""
    marathi_only = re.sub(r'[^\u0900-\u097F\s]', '', text)
    marathi_only = re.sub(r'\s+', ' ', marathi_only).strip()
    return marathi_only

def remove_stopwords(text):
    """Removes Marathi stopwords."""
    if not marathi_stopwords or not isinstance(text, str): return text
    return ' '.join([word for word in text.split() if word not in marathi_stopwords])

def preprocess_input(text):
    """Applies cleaning, stopword removal, tokenization, and padding for single text."""
    cleaned = clean_marathi_text(text)
    stopped = remove_stopwords(cleaned)
    sequence = tokenizer.texts_to_sequences([stopped])
    padded = pad_sequences(sequence, maxlen=MAX_SEQUENCE_LENGTH, padding='post', truncating='post')
    return padded

def preprocess_series(text_series):
    """Applies cleaning, stopword removal, tokenization, and padding for a pandas Series."""
    cleaned = text_series.fillna('').astype(str).apply(clean_marathi_text)
    stopped = cleaned.apply(remove_stopwords)
    sequences = tokenizer.texts_to_sequences(stopped)
    padded = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH, padding='post', truncating='post')
    return padded

# --- Prediction Function (Single Text) ---
def predict_hate_speech(text):
    """Takes raw text and returns predictions from both models."""
    if not text or not isinstance(text, str) or text.strip() == "":
        return None, None, None, None

    processed_input = preprocess_input(text)
    pred_label_proba = model_label.predict(processed_input, verbose=0).flatten()[0]
    pred_subclass_proba = model_subclass.predict(processed_input, verbose=0)
    pred_label = 1 if pred_label_proba > 0.5 else 0
    pred_subclass = np.argmax(pred_subclass_proba, axis=1)[0]
    label_name = "Hate Speech" if pred_label == 1 else "Not Hate Speech"
    subclass_name = sublabel_mapping.get(pred_subclass, "Unknown")
    return label_name, float(pred_label_proba), subclass_name, int(pred_subclass)

# --- Flask Application ---
app = Flask(__name__)
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'default_secret_key_change_me')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 # Optional: Limit file upload size

def allowed_file(filename):
    """Checks if the uploaded file has an allowed extension."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def index():
    """Handles requests for single text input."""
    prediction_result = None
    input_text = ""
    if request.method == 'POST':
        if 'text_input' in request.form:
            input_text = request.form.get('text_input', '')
            if not input_text.strip():
                 flash("Please enter some text to analyze.", "warning")
            else:
                try:
                    label, proba, subclass, _ = predict_hate_speech(input_text)
                    if label is not None:
                        prediction_result = {
                            'label': label, 'probability': f"{proba:.4f}", 'subclass': subclass
                        }
                    else: flash("Could not process empty input.", "warning")
                except Exception as e:
                     print(f"Error during prediction: {e}")
                     flash(f"An error occurred during prediction: {e}", "danger")

    return render_template('index.html', input_text=input_text, prediction=prediction_result)

# --- ROUTE FOR CSV CLASSIFICATION ---
@app.route('/classify-csv', methods=['POST'])
def classify_csv():
    """Handles CSV file upload, processing, and download."""
    if 'file' not in request.files:
        flash('No file part in the request.', 'danger')
        return redirect(url_for('index'))
    file = request.files['file']
    if file.filename == '':
        flash('No selected file.', 'warning')
        return redirect(url_for('index'))

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        upload_filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        output_filename = 'classified_' + filename
        output_filepath = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)

        try:
            file.save(upload_filepath)

            print(f"Processing uploaded file: {filename}")
            df = pd.read_csv(upload_filepath)

            text_col = None
            potential_cols = ['text']
            for col in df.columns:
                 if col.strip().lower() in potential_cols:
                     text_col = col
                     break

            if text_col is None:
                 flash(f"Error: Could not find a text column (expected one of: {potential_cols}, case-insensitive) in the CSV.", "danger")
                 if os.path.exists(upload_filepath): os.remove(upload_filepath)
                 return redirect(url_for('index'))

            print(f"Using text column: '{text_col}'")
            X_padded = preprocess_series(df[text_col])

            print(f"Predicting labels for {len(df)} rows...")
            pred_labels_proba = model_label.predict(X_padded, verbose=1)
            pred_labels = (pred_labels_proba.flatten() > 0.5).astype(int)

            print(f"Predicting subclasses for {len(df)} rows...")
            pred_subclasses_proba = model_subclass.predict(X_padded, verbose=1)
            pred_subclasses = np.argmax(pred_subclasses_proba, axis=1)

            df['Predicted_Label_ID'] = pred_labels
            df['Predicted_Label'] = df['Predicted_Label_ID'].map({0: 'Not Hate Speech', 1: 'Hate Speech'})
            df['Hate_Speech_Probability'] = pred_labels_proba.flatten()
            df['Predicted_Subclass_ID'] = pred_subclasses
            df['Predicted_Subclass'] = df['Predicted_Subclass_ID'].map(sublabel_mapping)

            df.to_csv(output_filepath, index=False, encoding='utf-8-sig')
            print(f"Saved classified data to {output_filepath}")

            # Provide the file for download
            return send_file(output_filepath, as_attachment=True, download_name=output_filename)

        except pd.errors.ParserError as e:
             flash(f"Error reading the CSV file: {e}. Please ensure it's a valid CSV.", "danger")
        except Exception as e:
             print(f"Error during CSV processing: {e}")
             flash(f"An error occurred during processing: {e}", "danger")
        finally:
             if os.path.exists(upload_filepath):
                 try:
                     os.remove(upload_filepath)
                     print(f"Removed uploaded file: {upload_filepath}")
                 except OSError as e:
                     print(f"Error removing uploaded file {upload_filepath}: {e}")

        return redirect(url_for('index'))

    else: # If file extension is not allowed
        flash('Invalid file type. Only .csv files are allowed.', 'warning')
        return redirect(url_for('index'))

# --- Run the App ---
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)











# import os
# import re
# import pickle
# import warnings
# import numpy as np
# import pandas as pd # For CSV handling
# import nltk
# # Download stopwords data if needed (safer check before import)
# try:
#     nltk.data.find('corpora/stopwords')
# except nltk.downloader.DownloadError:
#     print("NLTK stopwords not found. Downloading...")
#     nltk.download('stopwords', quiet=True)
# from nltk.corpus import stopwords
# import tensorflow as tf
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing.sequence import pad_sequences
# from flask import (
#     Flask, request, render_template, flash,
#     redirect, url_for, send_file # For CSV handling
# )
# from werkzeug.utils import secure_filename # For secure file handling
#
# # --- Configuration & Setup ---
#
# # Suppress TensorFlow/Keras warnings
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# tf.get_logger().setLevel('ERROR')
# warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
# warnings.filterwarnings('ignore', category=FutureWarning) # General future warning ignore
#
# # --- Constants ---
# MODELS_DIR = 'models' # Define the directory name
# TOKENIZER_PATH = os.path.join(MODELS_DIR, 'bilstm_tokenizer_final.pkl')
# LABEL_MODEL_PATH = os.path.join(MODELS_DIR, 'best_bilstm_label_model_final.keras')
# SUBCLASS_MODEL_PATH = os.path.join(MODELS_DIR, 'best_bilstm_subclass_model_final.keras')
# STOPWORDS_PATH = os.path.join(MODELS_DIR, 'marathi_stopwords.txt')
#
# UPLOAD_FOLDER = 'uploads' # Folder to store uploaded/processed CSVs
# ALLOWED_EXTENSIONS = {'csv'} # Allowed file extensions
#
# MAX_SEQUENCE_LENGTH = 100 # Must match the value used during training
# BATCH_SIZE = 64 # Defined globally, mainly for reference/training consistency
#
# # Define subclass mapping
# sublabel_mapping = {
#     0: "Not Hate Speech", 1: "Insulting", 2: "Religious Intolerance",
#     3: "Harassing", 4: "Gender Abusive",
# }
# num_subclasses = len(sublabel_mapping)
#
# # --- Create Upload Folder if it doesn't exist ---
# if not os.path.exists(UPLOAD_FOLDER):
#     try:
#         os.makedirs(UPLOAD_FOLDER)
#         print(f"Created upload folder: {UPLOAD_FOLDER}")
#     except OSError as e:
#         print(f"Error creating upload folder {UPLOAD_FOLDER}: {e}")
#         exit()
#
# # --- Load Models and Tokenizer (Load once at startup) ---
# print("Loading models and tokenizer...")
# if not os.path.exists(MODELS_DIR):
#     print(f"Error: Models directory '{MODELS_DIR}' not found.")
#     exit()
# try:
#     if not os.path.exists(TOKENIZER_PATH): raise FileNotFoundError(f"Tokenizer not found at {TOKENIZER_PATH}")
#     with open(TOKENIZER_PATH, 'rb') as f:
#         tokenizer = pickle.load(f)
#     print(f"Tokenizer loaded successfully from {TOKENIZER_PATH}")
#
#     if not os.path.exists(LABEL_MODEL_PATH): raise FileNotFoundError(f"Label model not found at {LABEL_MODEL_PATH}")
#     model_label = load_model(LABEL_MODEL_PATH)
#     print(f"Label model loaded successfully from {LABEL_MODEL_PATH}")
#
#     if not os.path.exists(SUBCLASS_MODEL_PATH): raise FileNotFoundError(f"Subclass model not found at {SUBCLASS_MODEL_PATH}")
#     model_subclass = load_model(SUBCLASS_MODEL_PATH)
#     print(f"Subclass model loaded successfully from {SUBCLASS_MODEL_PATH}")
#
# except FileNotFoundError as e:
#     print(f"Error loading file: {e}.")
#     exit()
# except Exception as e:
#     print(f"An unexpected error occurred during model/tokenizer loading: {e}")
#     exit()
#
# # --- Load Stopwords ---
# print("Loading stopwords...")
# try:
#     if not os.path.exists(STOPWORDS_PATH): raise FileNotFoundError(f"Stopwords file not found at {STOPWORDS_PATH}")
#     with open(STOPWORDS_PATH, 'r', encoding='utf-8') as f:
#         marathi_stopwords = set(line.strip() for line in f if line.strip())
#     print(f"Successfully loaded {len(marathi_stopwords)} stopwords from {STOPWORDS_PATH}.")
# except FileNotFoundError as e:
#     print(f"Warning: {e}. Stopword removal will be skipped.")
#     marathi_stopwords = set()
# except Exception as e:
#      print(f"An error occurred loading stopwords: {e}")
#      marathi_stopwords = set()
#
#
# # --- Preprocessing Functions ---
# def clean_marathi_text(text):
#     """Cleans Marathi text."""
#     if not isinstance(text, str): return ""
#     marathi_only = re.sub(r'[^\u0900-\u097F\s]', '', text)
#     marathi_only = re.sub(r'\s+', ' ', marathi_only).strip()
#     return marathi_only
#
# def remove_stopwords(text):
#     """Removes Marathi stopwords."""
#     if not marathi_stopwords or not isinstance(text, str): return text
#     return ' '.join([word for word in text.split() if word not in marathi_stopwords])
#
# def preprocess_input(text):
#     """Applies cleaning, stopword removal, tokenization, and padding for single text."""
#     cleaned = clean_marathi_text(text)
#     stopped = remove_stopwords(cleaned)
#     sequence = tokenizer.texts_to_sequences([stopped])
#     padded = pad_sequences(sequence, maxlen=MAX_SEQUENCE_LENGTH, padding='post', truncating='post')
#     return padded
#
# def preprocess_series(text_series):
#     """Applies cleaning, stopword removal, tokenization, and padding for a pandas Series."""
#     cleaned = text_series.fillna('').astype(str).apply(clean_marathi_text)
#     stopped = cleaned.apply(remove_stopwords)
#     sequences = tokenizer.texts_to_sequences(stopped)
#     padded = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH, padding='post', truncating='post')
#     return padded
#
# # --- Prediction Function (Single Text) ---
# def predict_hate_speech(text):
#     """Takes raw text and returns predictions from both models."""
#     if not text or not isinstance(text, str) or text.strip() == "":
#         return None, None, None, None
#
#     processed_input = preprocess_input(text)
#     pred_label_proba = model_label.predict(processed_input, verbose=0).flatten()[0]
#     pred_subclass_proba = model_subclass.predict(processed_input, verbose=0)
#     pred_label = 1 if pred_label_proba > 0.5 else 0
#     pred_subclass = np.argmax(pred_subclass_proba, axis=1)[0]
#     label_name = "Hate Speech" if pred_label == 1 else "Not Hate Speech"
#     subclass_name = sublabel_mapping.get(pred_subclass, "Unknown")
#     return label_name, float(pred_label_proba), subclass_name, int(pred_subclass)
#
# # --- Flask Application ---
# app = Flask(__name__)
# app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'default_secret_key_change_me')
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 # Optional: Limit file upload size
#
# def allowed_file(filename):
#     """Checks if the uploaded file has an allowed extension."""
#     return '.' in filename and \
#            filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
#
# @app.route('/', methods=['GET', 'POST'])
# def index():
#     """Handles requests for single text input."""
#     prediction_result = None
#     input_text = ""
#     if request.method == 'POST':
#         if 'text_input' in request.form:
#             input_text = request.form.get('text_input', '')
#             if not input_text.strip():
#                  flash("Please enter some text to analyze.", "warning")
#             else:
#                 try:
#                     label, proba, subclass, _ = predict_hate_speech(input_text)
#                     if label is not None:
#                         prediction_result = {
#                             'label': label, 'probability': f"{proba:.4f}", 'subclass': subclass
#                         }
#                     else: flash("Could not process empty input.", "warning")
#                 except Exception as e:
#                      print(f"Error during prediction: {e}")
#                      flash(f"An error occurred during prediction: {e}", "danger")
#
#     return render_template('index.html', input_text=input_text, prediction=prediction_result)
#
# # --- ROUTE FOR CSV CLASSIFICATION ---
# @app.route('/classify-csv', methods=['POST'])
# def classify_csv():
#     """Handles CSV file upload, processing, and download."""
#     if 'file' not in request.files:
#         flash('No file part in the request.', 'danger')
#         return redirect(url_for('index'))
#     file = request.files['file']
#     if file.filename == '':
#         flash('No selected file.', 'warning')
#         return redirect(url_for('index'))
#
#     if file and allowed_file(file.filename):
#         filename = secure_filename(file.filename)
#         upload_filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#         output_filename = 'classified_' + filename
#         output_filepath = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)
#
#         try:
#             file.save(upload_filepath)
#
#             print(f"Processing uploaded file: {filename}")
#             df = pd.read_csv(upload_filepath)
#
#             text_col = None
#             potential_cols = ['text']
#             for col in df.columns:
#                  if col.strip().lower() in potential_cols:
#                      text_col = col
#                      break
#
#             if text_col is None:
#                  flash(f"Error: Could not find a text column (expected one of: {potential_cols}, case-insensitive) in the CSV.", "danger")
#                  if os.path.exists(upload_filepath): os.remove(upload_filepath)
#                  return redirect(url_for('index'))
#
#             print(f"Using text column: '{text_col}'")
#             X_padded = preprocess_series(df[text_col])
#
#             print(f"Predicting labels for {len(df)} rows...")
#             # --- FIX: Removed batch_size=BATCH_SIZE from predict() calls ---
#             # Keras will use a default batch size for prediction.
#             pred_labels_proba = model_label.predict(X_padded, verbose=1)
#             pred_labels = (pred_labels_proba.flatten() > 0.5).astype(int)
#
#             print(f"Predicting subclasses for {len(df)} rows...")
#             # --- FIX: Removed batch_size=BATCH_SIZE from predict() calls ---
#             pred_subclasses_proba = model_subclass.predict(X_padded, verbose=1)
#             pred_subclasses = np.argmax(pred_subclasses_proba, axis=1)
#
#             df['Predicted_Label_ID'] = pred_labels
#             df['Predicted_Label'] = df['Predicted_Label_ID'].map({0: 'Not Hate Speech', 1: 'Hate Speech'})
#             df['Hate_Speech_Probability'] = pred_labels_proba.flatten()
#             df['Predicted_Subclass_ID'] = pred_subclasses
#             df['Predicted_Subclass'] = df['Predicted_Subclass_ID'].map(sublabel_mapping)
#
#             df.to_csv(output_filepath, index=False, encoding='utf-8-sig')
#             print(f"Saved classified data to {output_filepath}")
#
#             # Provide the file for download
#             return send_file(output_filepath, as_attachment=True, download_name=output_filename)
#
#         except pd.errors.ParserError as e:
#              flash(f"Error reading the CSV file: {e}. Please ensure it's a valid CSV.", "danger")
#         except Exception as e:
#              print(f"Error during CSV processing: {e}")
#              flash(f"An error occurred during processing: {e}", "danger")
#         finally:
#              if os.path.exists(upload_filepath):
#                  try:
#                      os.remove(upload_filepath)
#                      print(f"Removed uploaded file: {upload_filepath}")
#                  except OSError as e:
#                      print(f"Error removing uploaded file {upload_filepath}: {e}")
#
#         return redirect(url_for('index'))
#
#     else: # If file extension is not allowed
#         flash('Invalid file type. Only .csv files are allowed.', 'warning')
#         return redirect(url_for('index'))
#
# # --- Run the App ---
# if __name__ == '__main__':
#     app.run(debug=True, host='0.0.0.0', port=5000)

# import os
# import re
# import pickle
# import warnings
# import numpy as np
# import pandas as pd # For CSV handling
# import nltk
# # Download stopwords data if needed (safer check before import)
# try:
#     nltk.data.find('corpora/stopwords')
# except nltk.downloader.DownloadError:
#     print("NLTK stopwords not found. Downloading...")
#     nltk.download('stopwords', quiet=True)
# from nltk.corpus import stopwords
# import tensorflow as tf
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing.sequence import pad_sequences
# from flask import (
#     Flask, request, render_template, flash,
#     redirect, url_for, send_file # For CSV handling
# )
# from werkzeug.utils import secure_filename # For secure file handling
#
# # --- Configuration & Setup ---
#
# # Suppress TensorFlow/Keras warnings
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# tf.get_logger().setLevel('ERROR')
# warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
# warnings.filterwarnings('ignore', category=FutureWarning) # General future warning ignore
#
# # --- Constants ---
# # --- Paths point to a 'models' subdirectory ---
# MODELS_DIR = 'models' # Define the directory name
# TOKENIZER_PATH = os.path.join(MODELS_DIR, 'bilstm_tokenizer_final.pkl')
# LABEL_MODEL_PATH = os.path.join(MODELS_DIR, 'best_bilstm_label_model_final.keras')
# SUBCLASS_MODEL_PATH = os.path.join(MODELS_DIR, 'best_bilstm_subclass_model_final.keras')
# STOPWORDS_PATH = os.path.join(MODELS_DIR, 'marathi_stopwords.txt') # Also moved stopwords here
#
# UPLOAD_FOLDER = 'uploads' # Folder to store uploaded/processed CSVs
# ALLOWED_EXTENSIONS = {'csv'} # Allowed file extensions
#
# MAX_SEQUENCE_LENGTH = 100 # Must match the value used during training
#
# # --- Corrected subclass mapping ---
# sublabel_mapping = {
#     0: "Not Hate Speech",
#     1: "Insulting",
#     2: "Religious Intolerance",
#     3: "Harassing",
#     4: "Gender Abusive",
# }
# num_subclasses = len(sublabel_mapping)
#
# # --- Create Upload Folder if it doesn't exist ---
# if not os.path.exists(UPLOAD_FOLDER):
#     try:
#         os.makedirs(UPLOAD_FOLDER)
#         print(f"Created upload folder: {UPLOAD_FOLDER}")
#     except OSError as e:
#         print(f"Error creating upload folder {UPLOAD_FOLDER}: {e}")
#         exit()
#
# # --- Load Models and Tokenizer (Load once at startup) ---
# print("Loading models and tokenizer...")
# # Check if models directory exists
# if not os.path.exists(MODELS_DIR):
#     print(f"Error: Models directory '{MODELS_DIR}' not found.")
#     print("Please create the 'models' directory and place the required files inside.")
#     exit()
#
# try:
#     # Load Tokenizer
#     if not os.path.exists(TOKENIZER_PATH): raise FileNotFoundError(f"Tokenizer not found at {TOKENIZER_PATH}")
#     with open(TOKENIZER_PATH, 'rb') as f:
#         tokenizer = pickle.load(f)
#     print(f"Tokenizer loaded successfully from {TOKENIZER_PATH}")
#
#     # Load Keras Models
#     if not os.path.exists(LABEL_MODEL_PATH): raise FileNotFoundError(f"Label model not found at {LABEL_MODEL_PATH}")
#     model_label = load_model(LABEL_MODEL_PATH)
#     print(f"Label model loaded successfully from {LABEL_MODEL_PATH}")
#
#     if not os.path.exists(SUBCLASS_MODEL_PATH): raise FileNotFoundError(f"Subclass model not found at {SUBCLASS_MODEL_PATH}")
#     model_subclass = load_model(SUBCLASS_MODEL_PATH)
#     print(f"Subclass model loaded successfully from {SUBCLASS_MODEL_PATH}")
#
# except FileNotFoundError as e:
#     print(f"Error loading file: {e}.")
#     print(f"Ensure the required files exist inside the '{MODELS_DIR}' directory.")
#     exit()
# except Exception as e:
#     print(f"An unexpected error occurred during model/tokenizer loading: {e}")
#     exit()
#
# # --- Load Stopwords ---
# print("Loading stopwords...")
# try:
#     if not os.path.exists(STOPWORDS_PATH): raise FileNotFoundError(f"Stopwords file not found at {STOPWORDS_PATH}")
#     with open(STOPWORDS_PATH, 'r', encoding='utf-8') as f:
#         marathi_stopwords = set(line.strip() for line in f if line.strip())
#     print(f"Successfully loaded {len(marathi_stopwords)} stopwords from {STOPWORDS_PATH}.")
# except FileNotFoundError as e:
#     print(f"Warning: {e}. Stopword removal will be skipped.")
#     marathi_stopwords = set()
# except Exception as e:
#      print(f"An error occurred loading stopwords: {e}")
#      marathi_stopwords = set()
#
#
# # --- Preprocessing Functions ---
# def clean_marathi_text(text):
#     """Cleans Marathi text."""
#     if not isinstance(text, str): return ""
#     marathi_only = re.sub(r'[^\u0900-\u097F\s]', '', text)
#     marathi_only = re.sub(r'\s+', ' ', marathi_only).strip()
#     return marathi_only
#
# def remove_stopwords(text):
#     """Removes Marathi stopwords."""
#     if not marathi_stopwords or not isinstance(text, str): return text
#     return ' '.join([word for word in text.split() if word not in marathi_stopwords])
#
# def preprocess_input(text):
#     """Applies cleaning, stopword removal, tokenization, and padding for single text."""
#     cleaned = clean_marathi_text(text)
#     stopped = remove_stopwords(cleaned)
#     sequence = tokenizer.texts_to_sequences([stopped])
#     padded = pad_sequences(sequence, maxlen=MAX_SEQUENCE_LENGTH, padding='post', truncating='post')
#     return padded
#
# def preprocess_series(text_series):
#     """Applies cleaning, stopword removal, tokenization, and padding for a pandas Series."""
#     # Ensure input is treated as string, handle potential NaN/float values gracefully
#     cleaned = text_series.fillna('').astype(str).apply(clean_marathi_text)
#     stopped = cleaned.apply(remove_stopwords)
#     sequences = tokenizer.texts_to_sequences(stopped)
#     padded = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH, padding='post', truncating='post')
#     return padded
#
# # --- Prediction Function (Single Text) ---
# def predict_hate_speech(text):
#     """Takes raw text and returns predictions from both models."""
#     if not text or not isinstance(text, str) or text.strip() == "":
#         return None, None, None, None # Handle empty input
#
#     processed_input = preprocess_input(text)
#     # Use loaded models for prediction
#     pred_label_proba = model_label.predict(processed_input, verbose=0).flatten()[0]
#     pred_subclass_proba = model_subclass.predict(processed_input, verbose=0)
#     pred_label = 1 if pred_label_proba > 0.5 else 0
#     pred_subclass = np.argmax(pred_subclass_proba, axis=1)[0]
#     label_name = "Hate Speech" if pred_label == 1 else "Not Hate Speech"
#     subclass_name = sublabel_mapping.get(pred_subclass, "Unknown")
#     return label_name, float(pred_label_proba), subclass_name, int(pred_subclass)
#
# # --- Flask Application ---
# app = Flask(__name__)
# # Important: Change this secret key for production environments!
# app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'default_secret_key_change_me')
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 # Optional: Limit file upload size (e.g., 16MB)
#
#
# def allowed_file(filename):
#     """Checks if the uploaded file has an allowed extension."""
#     return '.' in filename and \
#            filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
#
# @app.route('/', methods=['GET', 'POST'])
# def index():
#     """Handles requests for single text input."""
#     prediction_result = None
#     input_text = ""
#     if request.method == 'POST':
#         # Ensure the POST request is for the text form, not the file upload form
#         if 'text_input' in request.form:
#             input_text = request.form.get('text_input', '')
#             if not input_text.strip():
#                  flash("Please enter some text to analyze.", "warning")
#             else:
#                 try:
#                     label, proba, subclass, _ = predict_hate_speech(input_text)
#                     if label is not None:
#                         prediction_result = {
#                             'label': label, 'probability': f"{proba:.4f}", 'subclass': subclass
#                         }
#                     else: flash("Could not process empty input.", "warning")
#                 except Exception as e:
#                      print(f"Error during prediction: {e}")
#                      flash(f"An error occurred during prediction: {e}", "danger")
#
#     # Render the main page
#     return render_template('index.html', input_text=input_text, prediction=prediction_result)
#
# # --- ROUTE FOR CSV CLASSIFICATION ---
# @app.route('/classify-csv', methods=['POST'])
# def classify_csv():
#     """Handles CSV file upload, processing, and download."""
#     if 'file' not in request.files:
#         flash('No file part in the request.', 'danger')
#         return redirect(url_for('index'))
#     file = request.files['file']
#     if file.filename == '':
#         flash('No selected file.', 'warning')
#         return redirect(url_for('index'))
#
#     if file and allowed_file(file.filename):
#         filename = secure_filename(file.filename) # Prevent directory traversal attacks
#         upload_filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#         output_filename = 'classified_' + filename
#         output_filepath = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)
#
#         try:
#             file.save(upload_filepath)
#             # Don't flash here, flash after processing or on error
#
#             # --- Read and Process CSV ---
#             print(f"Processing uploaded file: {filename}")
#             df = pd.read_csv(upload_filepath)
#
#             # Find text column (case-insensitive)
#             text_col = None
#             potential_cols = ['text'] # Add other common names if needed
#             for col in df.columns:
#                  if col.strip().lower() in potential_cols:
#                      text_col = col
#                      break # Found the first match
#
#             if text_col is None:
#                  flash(f"Error: Could not find a text column (expected one of: {potential_cols}, case-insensitive) in the CSV.", "danger")
#                  # Clean up uploaded file on error
#                  if os.path.exists(upload_filepath): os.remove(upload_filepath)
#                  return redirect(url_for('index'))
#
#             print(f"Using text column: '{text_col}'")
#             # Preprocess the text data
#             X_padded = preprocess_series(df[text_col])
#
#             # Predict using both BiLSTM models (use batch prediction for efficiency)
#             print(f"Predicting labels for {len(df)} rows...")
#             pred_labels_proba = model_label.predict(X_padded, batch_size=BATCH_SIZE, verbose=1)
#             pred_labels = (pred_labels_proba.flatten() > 0.5).astype(int)
#
#             print(f"Predicting subclasses for {len(df)} rows...")
#             pred_subclasses_proba = model_subclass.predict(X_padded, batch_size=BATCH_SIZE, verbose=1)
#             pred_subclasses = np.argmax(pred_subclasses_proba, axis=1)
#
#             # Add predictions to the DataFrame
#             df['Predicted_Label_ID'] = pred_labels
#             df['Predicted_Label'] = df['Predicted_Label_ID'].map({0: 'Not Hate Speech', 1: 'Hate Speech'})
#             df['Hate_Speech_Probability'] = pred_labels_proba.flatten()
#             df['Predicted_Subclass_ID'] = pred_subclasses
#             df['Predicted_Subclass'] = df['Predicted_Subclass_ID'].map(sublabel_mapping)
#
#             # Save the result to a new CSV file
#             df.to_csv(output_filepath, index=False, encoding='utf-8-sig') # Use utf-8-sig for Excel compatibility
#             print(f"Saved classified data to {output_filepath}")
#
#             # Provide the file for download
#             return send_file(output_filepath, as_attachment=True, download_name=output_filename)
#
#         except pd.errors.ParserError as e:
#              flash(f"Error reading the CSV file: {e}. Please ensure it's a valid CSV.", "danger")
#         except Exception as e:
#              print(f"Error during CSV processing: {e}")
#              flash(f"An error occurred during processing: {e}", "danger")
#         finally:
#              # Clean up the originally uploaded file after processing (success or failure)
#              if os.path.exists(upload_filepath):
#                  try:
#                      os.remove(upload_filepath)
#                      print(f"Removed uploaded file: {upload_filepath}")
#                  except OSError as e:
#                      print(f"Error removing uploaded file {upload_filepath}: {e}")
#
#         # Redirect back to index if an error occurred during processing
#         return redirect(url_for('index'))
#
#     else: # If file extension is not allowed
#         flash('Invalid file type. Only .csv files are allowed.', 'warning')
#         return redirect(url_for('index'))
#
# # --- Run the App ---
# if __name__ == '__main__':
#     # Set host='0.0.0.0' to make it accessible on your local network
#     # Set debug=False for production deployment
#     app.run(debug=True, host='0.0.0.0', port=5000)

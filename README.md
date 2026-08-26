# 🧠 Guess My Mood

An AI-powered facial emotion recognition application that uses a Convolutional Neural Network (CNN) to predict emotions from facial expressions.

The application allows users to take a picture using their webcam and predicts their facial emotion using a trained deep learning model.

---

## 🎯 Project Overview

Guess My Mood is a Deep Learning project built using the FER-2013 facial expression dataset.

The model recognizes 7 different emotions:

- 😡 Angry
- 🤢 Disgust
- 😨 Fear
- 😄 Happy
- 😢 Sad
- 😲 Surprise
- 😐 Neutral

The project includes:

- CNN-based emotion classification
- Data augmentation
- Class weighting for imbalanced data
- Face detection using OpenCV
- Streamlit web interface
- Webcam image input
- Model deployment using Streamlit Community Cloud

---

## 🧠 Machine Learning Model

The model is a Convolutional Neural Network (CNN).

### Input

```text
48 × 48 grayscale facial image


Input Image
    ↓
Data Augmentation
    ↓
Convolutional Layer
    ↓
Batch Normalization
    ↓
Convolutional Layer
    ↓
Max Pooling
    ↓
Dropout
    ↓
Convolutional Layer
    ↓
Batch Normalization
    ↓
Max Pooling
    ↓
Dropout
    ↓
Convolutional Layer
    ↓
Max Pooling
    ↓
Flatten
    ↓
Dense Layer
    ↓
Dropout
    ↓
7-Class Softmax Output

📊 Dataset

The project uses the FER-2013 facial expression dataset.

The dataset contains 48×48 grayscale facial images categorized into seven emotion classes.

Training dataset:

28,709 images

The dataset is intentionally not included in this repository because of its large size.

📈 Model Performance

The current CNN achieved approximately:

Validation Accuracy: 57.07%

The model performs differently across emotion classes because facial expressions can be difficult to distinguish and the FER-2013 dataset is imbalanced.

Class weighting and data augmentation were used to improve performance.

🛠️ Technologies Used
Programming Language
Python
Machine Learning / Deep Learning
TensorFlow
Keras
Scikit-learn
Computer Vision
OpenCV
Pillow
Data Processing
NumPy
Pandas
Visualization
Matplotlib
Seaborn
Web Application
Streamlit
Deployment
Streamlit Community Cloud
📁 Project Structure
GuessMyMood/
│
├── app.py
├── train.py
├── test.py
├── mood_model.keras
├── requirements.txt
├── .gitignore
├── README.md
│
└── dataset/
    ├── train.csv
    └── test.csv

The dataset/ directory is not uploaded to GitHub because the dataset is large.

⚙️ Installation

Clone the repository:

git clone https://github.com/gaurawarkhede/GuessMyMood.git

Move into the project:

cd GuessMyMood

Create a virtual environment:

python -m venv .venv

Activate it on Windows:

.venv\Scripts\activate

Install dependencies:

pip install -r requirements.txt
🚀 Run the Application

Start the Streamlit application:

streamlit run app.py

The application will open in your browser.

You can then:

Allow camera access.
Take a picture.
The application detects your face.
The face is converted to grayscale.
The face is resized to 48×48.
The CNN predicts the emotion.
The application displays the predicted emotion and confidence.
🔍 How It Works
Webcam
   ↓
Capture Image
   ↓
OpenCV Face Detection
   ↓
Crop Face
   ↓
Convert to Grayscale
   ↓
Resize to 48×48
   ↓
Normalize Pixel Values
   ↓
CNN Model
   ↓
Emotion Probabilities
   ↓
Predicted Emotion
🎥 Example

The application produces results such as:

🎯 My Guess

😄 You look happy!

Predicted Emotion:
Happy

Confidence:
72.4%

It also displays the probability for each emotion.

⚠️ Limitations

The model is not perfect.

Facial emotion recognition is challenging because:

Different emotions can have similar facial expressions.
Lighting can affect predictions.
Camera quality can affect predictions.
Facial orientation can affect predictions.
FER-2013 contains class imbalance.
The model has approximately 57% validation accuracy.

Therefore, the predictions should be considered an AI estimation rather than a reliable measurement of someone's actual mood or emotional state.

🔮 Future Improvements

Possible improvements include:

 Improve CNN architecture
 Train for longer with better learning-rate scheduling
 Use transfer learning
 Improve face detection
 Add real-time webcam emotion detection
 Add emotion history
 Add prediction charts
 Improve UI/UX
 Add dark mode
 Deploy the improved model
 Experiment with EfficientNet / MobileNet
 Improve performance on minority classes
👨‍💻 Author

Gaurav Warkhede

Artificial Intelligence & Machine Learning

⭐ If you like this project

Feel free to star the repository and experiment with the model!


### Then push it

After saving `README.md`:

```powershell
git add README.md
git commit -m "Add project documentation"
git push origin main

Then refresh your GitHub repository. The README will automatically appear underneath your project files.

Your repository will now look much more like a proper portfolio project:

🧠 GuessMyMood
├── 📄 README.md
├── 🐍 app.py
├── 🧠 mood_model.keras
├── 🐍 train.py
├── 🐍 test.py
├── 📦 requirements.txt
└── ⚙️ .gitignore
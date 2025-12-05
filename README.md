### Vitamin Deficiency Detection System (AI + CNN + MLP)

This project is an AI-powered non-invasive vitamin deficiency detection system that identifies potential deficiencies using image-based analysis (CNN) and symptom-based text analysis (MLP).  
The model analyzes visible markers on eyes, lips, tongue, and nails, as well as manually entered symptoms, then provides predicted deficiencies and recommended precautions.

---

📌 Features
- 🖼 Image-based prediction using trained CNN model  
- ✍️ Symptom-based prediction using MLP model  
- 🌐 Fully working frontend (HTML, CSS, JS, Bootstrap)  
- 🧠 Flask backend integrates ML models  
- 🗄 Stores user results in MySQL database  
- ⚡ Fast, lightweight, and easy to deploy  
- 📱 User-friendly interface  

---

🧰 Tech Stack

Frontend
- HTML  
- CSS  
- JavaScript  
- Bootstrap  

Backend
- Python  
- Flask  

Machine Learning
- CNN (Image Classification)  
- MLP (Symptom Classification)  
- TensorFlow / Keras  
- OpenCV  
- NumPy, Pandas  

Database
- MySQL  
- phpMyAdmin / MySQL Workbench  

---

📁 Project Structure

```Project/
│
├── frontend/
│ ├── index.html
│ ├── styles/
│ ├── scripts/
│ └── assets/
│
├── backend/
│ ├── app.py
│ ├── cnn_model.h5
│ ├── mlp_model.pkl
│ ├── utils/
│ └── uploads/
│
├── documentation/
│ ├── PROJECT_REPORT.md
│ └── diagrams/
│
├── requirements.txt
├── README.md
└── LICENSE 
---

⚙️ Installation Instructions

1️⃣ Clone the Repository
git clone https://github.com/Amoghavarshab/Vitamin-Deficiency-Detection.git

cd Vitamin-Deficiency-Detection


---

2️⃣ Backend Setup (Flask + ML Models)

Create a virtual environment:


python -m venv venv
venv\Scripts\activate


Install dependencies:


pip install -r requirements.txt


Run Flask backend:


python app.py


Backend runs on:  
👉 http://127.0.0.1:5000/

---

3️⃣ Frontend Setup
Just open:


frontend/index.html

The frontend communicates with the Flask backend through HTTP requests.

---

🧠 Machine Learning Models

CNN Model
- Trained on image dataset (lips, eyes, tongue, nails)
- Detects deficiency-related visual patterns  
- Saved as `cnn_model.h5`

MLP Model
- Trained on symptom–deficiency dataset  
- Predicts deficiency based on user-entered text  
- Saved as `mlp_model.pkl`

---

🔄 Workflow

1. User uploads image **OR** enters symptoms  
2. Frontend sends data to Flask backend  
3. Backend preprocesses data  
4. ML models (CNN/MLP) generate predictions  
5. Result stored in MySQL database  
6. User sees prediction + recommendations  

---

📊 Accuracy Graphs (Add your images)


/documentation/accuracy_graph.png
/documentation/loss_graph.png
/documentation/classification_report.png


---

📝 License
This project is licensed under the **MIT License**.  
See the [LICENSE](LICENSE) file for details.

---

✨ Contributors
- Amoghavarsha B (Developer)
- Team Members
- Project Guide

---

⭐ If you like this project, give it a star on GitHub!

 ### 🚀 How to Run the Project
1️⃣ Clone the Repository
```bash
git clone https://github.com/Amoghavarshab/Vitamin-Deficiency-Detection.git
cd Vitamin-Deficiency-Detection
2️⃣ Backend Setup (Flask + Models)
Create & activate virtual environment (Windows):
bash
Copy code
python -m venv venv
venv\Scripts\activate
Install dependencies:
bash
Copy code
pip install -r requirements.txt
Run the Flask server:
bash
Copy code
python app.py
By default the backend runs at:
http://127.0.0.1:5000/

3️⃣ Frontend Setup
The frontend is a simple web interface made with HTML, CSS, JS, and Bootstrap.

You can open the main HTML file directly in your browser (adjust the path according to your repo):

text
Copy code
frontend/Project-V/index.html
Or if you later rename it:

text
Copy code
frontend/index.html
The frontend will send requests to the Flask backend for:

Image-based vitamin deficiency prediction (CNN)

Symptom-based text prediction (MLP)

📊 Model Details
CNN Model for image classification (eyes, lips, tongue, nails)

MLP Model for symptom text classification

Trained on curated datasets of vitamin deficiencies

You can add your accuracy graph, loss graph, and classification report images under a section like:

documentation/ and embed them here with markdown image tags.

🧰 Tech Stack Summary
Frontend: HTML, CSS, JavaScript, Bootstrap

Backend: Python, Flask

ML: TensorFlow, Keras, OpenCV, scikit-learn

Database: MySQL



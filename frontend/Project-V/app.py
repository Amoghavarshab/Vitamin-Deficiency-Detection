from flask import Flask, render_template, request, redirect, url_for, session, flash
import mysql.connector
from werkzeug.security import generate_password_hash, check_password_hash
import os
from werkzeug.utils import secure_filename
import numpy as np
import glob
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import json
import random
from keras.models import load_model
from nltk.stem import WordNetLemmatizer
import pickle
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import cv2
import pickle
from flask import Flask, render_template, request, flash, redirect, url_for, session
from keras.models import load_model
from nltk.stem import WordNetLemmatizer



nltk.download('punkt')
nltk.download('wordnet')


# Load dataset
with open('intents1.json', 'r') as file:
    intents = json.load(file)

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Preprocessing
def preprocess_data(intents):
    words = []
    classes = []
    documents = []
    ignore_words = ['?', '!', ',', '.']

    for intent in intents['intents']:
        for pattern in intent['patterns']:
            # Tokenize each word
            word_list = nltk.word_tokenize(pattern)
            words.extend(word_list)
            # Add to documents
            documents.append((word_list, intent['tag']))
            # Add to classes if not already there
            if intent['tag'] not in classes:
                classes.append(intent['tag'])

    # Lemmatize and remove duplicates
    words = [lemmatizer.lemmatize(word.lower()) for word in words if word not in ignore_words]
    words = sorted(set(words))
    classes = sorted(set(classes))

    return words, classes, documents

words, classes, documents = preprocess_data(intents)

# Create training data
training = []
output_empty = [0] * len(classes)

for document in documents:
    bag = []
    word_patterns = document[0]
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
    for word in words:
        bag.append(1 if word in word_patterns else 0)

    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1
    training.append([bag, output_row])

random.shuffle(training)
training = np.array(training, dtype=object)

train_x = np.array(list(training[:, 0]))
train_y = np.array(list(training[:, 1]))
# Load the trained model
model = load_model('chatbot_model.h5')
def load_patterns_from_file():
    with open('patterns.txt', 'r') as file:
        patterns = file.read().split(', ')
    return patterns

patterns_from_file = load_patterns_from_file()

# Chatbot response
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, word in enumerate(words):
            if word == s:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence, model):
    bow_input = bow(sentence, words)
    res = model.predict(np.array([bow_input]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    for i in intents_json['intents']:
        if i['tag'] == tag:
            return random.choice(i['responses'])

# Chat

app = Flask(__name__)
app.secret_key = 'wverihdfuvuwi2482'

app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']




loaded_model = load_model('trained_model_DNN1.h5')
class_names = ["vitaminA", "vitaminB", "vitaminD", "vitamink"] 


def path_to_tensor(img_path, width=224, height=224):
    img = image.load_img(img_path, target_size=(width, height))
    x = image.img_to_array(img)
    return np.expand_dims(x, axis=0)


def read_file_contents(file_path):
    try:
        with open(file_path, 'r') as file:
            return file.read()
    except FileNotFoundError:
        return f"File not found: {file_path}"





def get_db_connection():
    return mysql.connector.connect(
        host='localhost',
        user='root',
        password='', 
        database='vitamin_db'  
    )
doctor_info = []


def fetch_doctor_info():
    """Fetch all doctor details from the database."""
    global doctor_info  
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        
        cursor.execute("SELECT doctor_name, hospital_name, location FROM doctor")
        doctor_info = cursor.fetchall()  

    except mysql.connector.Error as err:
        print(f"Error: {err}")
        doctor_info = []

    finally:
        cursor.close()
        conn.close()

@app.route('/view_doctors')
def view_doctors():
    
    conn = get_db_connection()
    cursor = conn.cursor()

   
    cursor.execute('SELECT doctor_name, hospital_name, location FROM doctor')


    doctor_infos = cursor.fetchall()


    cursor.close()
    conn.close()


    return render_template('view_doctors.html', doctor_infos=doctor_infos)


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/contact')
def contact():
    return render_template('contact.html')


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        number = request.form['number']
        password = request.form['password']
        location = request.form['location']
        

        hashed_password = generate_password_hash(password)

       
        conn = get_db_connection()
        cursor = conn.cursor()
        try:
            cursor.execute(
                'INSERT INTO users (name, email, number, password, location) VALUES (%s, %s, %s, %s, %s)',
                (name, email, number, hashed_password, location)
            )
            conn.commit()
            flash('Registration successful. Please login.', 'success')
            return redirect(url_for('login'))
        except mysql.connector.IntegrityError:
            flash('Email already exists.', 'danger')
        finally:
            cursor.close()
            conn.close()

    return render_template('register.html')



@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
    
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute('SELECT * FROM users WHERE email = %s', (email,))
        user = cursor.fetchone()
        
        if user and check_password_hash(user['password'], password):
            session['email'] = user['email']  
            flash('Login successful!', 'success')
            return redirect(url_for('check'))
        else:
            flash('Invalid email or password', 'danger')
    
    return render_template('login.html')


@app.route('/docregister', methods=['GET', 'POST'])
def register_doctor():
    if request.method == 'POST':
        doctor_name = request.form.get('doctor_name')
        hospital_name = request.form.get('hospital_name')
        address       =  request.form.get('address')
        location = request.form.get('location')

        if doctor_name and hospital_name and location and address:
            try:
            
                conn = get_db_connection()
                cursor = conn.cursor()

            
                cursor.execute("""
                    SELECT * FROM doctor 
                    WHERE doctor_name = %s AND hospital_name = %s 
                    AND location = %s
                      AND  address = %s
                """, (doctor_name, hospital_name, location,  address))
                
                existing_doctor = cursor.fetchone()

                if existing_doctor:
                    flash('This doctor is already registered with the same hospital and location.', 'danger')
                    return redirect(url_for('register_doctor'))

               
                cursor.execute("""
                    INSERT INTO doctor (doctor_name, hospital_name, address, location) 
                    VALUES (%s, %s, %s, %s)
                """, (doctor_name, hospital_name, address, location))
                
                
                conn.commit()

                flash('Doctor details registered successfully!', 'success')
                return redirect(url_for('register_doctor'))

            except mysql.connector.Error as err:
                flash(f'Error: {err}', 'danger')

            finally:
                cursor.close()
                conn.close()

        else:
            flash('Please fill in all fields.', 'danger')

    return render_template('register_doctor.html')



@app.route('/check')
def check():
    if 'email' not in session:
        flash("Please login to access this page", "info")
        return redirect(url_for('login'))
    return render_template('check.html')


@app.route('/detection', methods=['GET', 'POST'])
def detection():
    if 'email' not in session:
        flash("Please login to access this page", "info")
        return redirect(url_for('login'))

    conn = get_db_connection()  
    cursor = conn.cursor()

    user_email = session['email']
    cursor.execute("SELECT location FROM users WHERE email = %s", (user_email,))
    user_location = cursor.fetchone()[0]  

   
    cursor.execute("SELECT doctor_name, hospital_name, address, location FROM doctor WHERE location = %s", (user_location,))
    doctor_info = cursor.fetchone()

   
    if doctor_info:
       
        doctor_name, hospital_name, doctor_address, location = doctor_info
    else:
       
        doctor_name = "Dr. C Sharma"
        hospital_name = "General Hospital"
        doctor_address = "India"
        location = user_location 

    image_url = None
    result_text = None
    confidence_threshold = 0.85 
    
    if request.method == 'POST':
        if 'image' not in request.files:
            flash("No image part", "danger")
            return redirect(request.url)
        
        file = request.files['image']
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            image_url = url_for('static', filename=f'uploads/{filename}')
            
            img_array = path_to_tensor(file_path, width=224, height=224) / 255.0

            pred = loaded_model.predict(img_array)
            predicted_index = np.argmax(pred)
            confidence_score = pred[0][predicted_index]
            predicted_class_name = class_names[predicted_index]

            if confidence_score >= confidence_threshold and predicted_class_name in class_names:
                result_text = f"Predicted Class: {predicted_class_name} \n"

                file_paths = {
                    "vitaminA": ["./files/char.txt", "./files/PREC.txt"],
                    "vitaminB": ["./files2/charra.txt", "./files2/PRECCC.txt"],
                    "vitaminD": ["./files3/charrac.txt", "./files3/PRECCCa.txt"],
                    "vitamink": ["./files1/charr.txt", "./files1/PRECC.txt"]
                }

                for file_path in file_paths.get(predicted_class_name, []):
                    result_text += read_file_contents(file_path) + "\n\n"
                
            else:
                result_text = (
                    "Could not identify deficiency, \n\n"
                    f"Please consult with nearby doctor {doctor_name} at {hospital_name}, {doctor_address}, {location} for further assistance."
                )
        
        else:
            result_text = "Please upload a valid image file (jpg, jpeg, png)."

    conn.close()

    return render_template('detection.html', image_url=image_url, result_text=result_text)


@app.route('/models', methods=['GET', 'POST'])
def models():
    if 'email' not in session:
        flash("Please login to access this page", "info")
        return redirect(url_for('login'))

    # Initialize variables
    result_message = None  
    doctor_name = None  
    doctor_address = None  
    hospital_name = None  
    location = None  
    content = None
    vitamin_content = None

    if request.method == 'POST':

        message = request.form.get('textInput')
        print(f"Received message: {message}")

        if message.lower() not in [pattern.lower() for pattern in patterns_from_file]:
            result_message = "Chatbot: Could not identify deficiency. Please consult a nearby doctor."
        else:
            # Predict the class and get a response
            ints = predict_class(message, model)
            if ints:
                res = get_response(ints, intents)
                print(f"Chatbot response: {res}")  # Debug response
                result_message = f"Chatbot: {res}"
                print(f"Response key (res): {res}")
                if res== "Vitamin L": 
                    print("deficiency")
                    with open("./files/char.txt", "r") as file:
                        content = file.read()
                        print(content)
                    with open("./files/PREC.txt", "r") as file:
                        content = file.read()
                        print(content)
                        
                        
                elif res== "Vitamin K":
                    print("Vitamin K deficiency")
                    with open("./files1/charr.txt", "r") as file:
                        content = file.read()
                        print(content)
                    with open("./files1/PRECC.txt", "r") as file:
                        content = file.read()
                        print(content)
                        
                elif res== "Vitamin D":
                    print("Vitamin D deficiency")
                    with open("./files3/charrac.txt", "r") as file:
                        content = file.read()
                        print(content)
                    with open("./files3/PRECCCa.txt", "r") as file:
                        content = file.read()
                        print(content)
                    
                elif res== "Vitamin B":
                    print("Vitamin B deficiency")
                    with open("./files2/charra.txt", "r") as file:
                        content = file.read()
                        print(content)
                    with open("./files2/PRECCC.txt", "r") as file:
                        content = file.read()
                        print(content)

                elif res== "Vitamin A":
                    print("deficiency in ")
                    print("./files/char.txt")
                    with open("./files/char.txt", "r") as file:
                        content = file.read()
                        print(content)
                        with open("./files/PREC.txt", "r") as file:
                            content = file.read()
                            print(content)

        # Fetch doctor info if needed
        if result_message == "Chatbot: Could not identify deficiency. Please consult a nearby doctor.":
            user_email = session['email']
            conn = get_db_connection()
            cursor = conn.cursor()

            # Fetch user location from the database
            cursor.execute("SELECT location FROM users WHERE email = %s", (user_email,))
            user_location = cursor.fetchone()
            print(f"User location from DB: {user_location}")  # Debug user location

            if user_location:
                user_location = user_location[0].lower()

                # Fetch doctor details based on the user's location
                cursor.execute("SELECT doctor_name, hospital_name, address, location FROM doctor WHERE location = %s", (user_location,))
                doctor_info = cursor.fetchone()
                print(f"Doctor info for {user_location}: {doctor_info}") 

                if doctor_info:
                    doctor_name, hospital_name, doctor_address, location = doctor_info
                else:
                    # Fallback if no doctor found for the location
                    doctor_name = "Dr. C Sharma"
                    hospital_name = "General Hospital"
                    doctor_address = "India"
                    location = user_location

            conn.close()

    # Render the template with all the variables
    return render_template(
        'models.html', 
        result_message=result_message, 
        doctor_name=doctor_name, 
        doctor_address=doctor_address,
        hospital_name=hospital_name,  
        location=location,
        content=content, 
        vitamin_content=vitamin_content
    )

@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out.', 'info')
    return redirect(url_for('login'))


if __name__ == '__main__':
    app.run(debug=True)

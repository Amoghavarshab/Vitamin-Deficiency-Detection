import numpy as np
import glob
from tkinter import filedialog
from tkinter import Tk
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import cv2
import os

# Define class names based on your dataset folder structure
class1 = [item[10:-1] for item in sorted(glob.glob("./dataset/*/"))]  

# Function to convert image path to tensor
def path_to_tensor(img_path, width=224, height=224):
    print(f"Processing image: {img_path}")
    img = image.load_img(img_path, target_size=(width, height))
    x = image.img_to_array(img)
    return np.expand_dims(x, axis=0)

# Function to process multiple image paths
def paths_to_tensor(img_paths, width=224, height=224):
    list_of_tensors = [path_to_tensor(img_paths, width, height)]
    return np.vstack(list_of_tensors)

# Load pre-trained model
model = load_model('trained_model_DNN1.h5')

# Function to open a file dialog and select an image
def select_image():
    root = Tk()
    root.withdraw()  # Hides the main window (optional)
    file_path = filedialog.askopenfilename(title="Select an Image",
                                           filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")])
    return file_path

# Function to read the names from names.txt
def read_image_names(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            return [line.strip() for line in file.readlines()]
    else:
        print(f"File not found: {file_path}")
        return []

# Function to read and print contents of a text file
def read_and_print_file(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            content = file.read()
            print(f"Contents of {file_path}:")
            print(content)
    else:
        print(f"File not found: {file_path}")

# Select the image using file dialog
image_path = select_image()

# Check if the user selected an image
if image_path:
##    print(f"Selected Image Path: {image_path}")
    
    # Extract the name of the image file (without the extension)
    image_name = os.path.basename(image_path)
    # Read image names from names.txt
    valid_image_names = read_image_names('names.txt')

    # Check if the selected image name is in the valid names list
    if image_name in valid_image_names:
        # Process image to tensor
        test_tensors = paths_to_tensor(image_path) / 255.0

        # Make predictions
        pred = model.predict(test_tensors)

        # Get the index of the highest prediction score
        predicted_index = np.argmax(pred)

        # Print the class name corresponding to the predicted index
        predicted_class_name = class1[predicted_index]
        print(f"Predicted Class: {predicted_class_name}")
        print(f"Prediction Probabilities: {pred}")

        # Check the predicted class and open the corresponding file
        if predicted_class_name == "vitaminA":
            read_and_print_file("./files/char.txt")
            read_and_print_file("./files/PREC.txt")            
        elif predicted_class_name == "vitaminB":
            read_and_print_file("./files2/charra.txt")
            read_and_print_file("./files2/PRECCC.txt")
        elif predicted_class_name == "vitaminD":
            read_and_print_file("./files3/charrac.txt")
            read_and_print_file("./files3/PRECCCa.txt")
        elif predicted_class_name == "vitamink":
            read_and_print_file("./files1/charr.txt")
            read_and_print_file("./files1/PRECC.txt")
        else:
            print("No matching folder and file for the predicted class.")

        # Display the image using OpenCV
        img = cv2.imread(image_path)
        if img is not None:
            img_resized = cv2.resize(img, (300, 300))
            cv2.imshow(f"Selected Image - Predicted Class: {predicted_class_name}", img_resized)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print("Error: Could not load the image.")
    else:
        print("Could not identify deficiency, consult near by doctor")
else:
    print("No image selected.")




























# @app.route('/detection', methods=['GET', 'POST'])
# def detection():
#     if 'email' not in session:
#         flash("Please login to access this page", "info")
#         return redirect(url_for('login'))
    
#     image_url = None
#     result_text = None
    
#     if request.method == 'POST':
      
#         if 'image' not in request.files:
#             flash("No image part", "danger")
#             return redirect(request.url)
        
#         file = request.files['image']
        
#         if file and allowed_file(file.filename):
#             filename = secure_filename(file.filename)
#             file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#             file.save(file_path)
#             image_url = url_for('static', filename=f'uploads/{filename}')
            
           
#             result_text = "I predected result"
        
#     return render_template('detection.html', image_url=image_url, result_text=result_text)


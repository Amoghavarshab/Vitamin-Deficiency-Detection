from tkinter import *
import tkinter as tk
import cv2
import os
from tkinter import filedialog
from glob import glob
import numpy as np
from PIL import ImageFile                            
from PIL import Image
from PIL import ImageTk
# global variables
import time
from PIL import ImageTk, Image
from skimage.filters import median
global rep
import copy
import random
from numpy import load
from numpy import save
from skimage.color import rgb2gray
from tkinter import messagebox
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from random import randint
from tkinter import *
from PIL import Image, ImageTk

class Window(Frame):
    def __init__(self, master=None):
        Frame.__init__(self, master)                 
        self.master = master
        self.pack(fill=BOTH, expand=1)  # Fill the entire window

        # Load the background image
        image_path = "img.jpeg"  # Replace this with the path to your image
        img = Image.open(image_path)
        img = img.resize((1200, 800), Image.ANTIALIAS)  # Resize the image to fit the window size
        self.background_image = ImageTk.PhotoImage(img)

        # Create a label with the background image
        self.background_label = Label(self, image=self.background_image)
        self.background_label.place(x=0, y=0, relwidth=1, relheight=1)  # Place the label to cover the entire window

        # Changing the title of our master widget      
        self.master.title("Window with Background Image")

        quitButton = Button(self,command=self.query,text="browse input",fg="blue",activebackground="dark red",width=20)
        quitButton.place(x=700, y=300)

        quitButton = Button(self,command=self.classification,text="prediction",fg="blue",activebackground="dark red",width=20)
        quitButton.place(x=700, y=400)

        load = Image.open("logo.jfif")
        render = ImageTk.PhotoImage(load)

        image=Label(self, image=render,borderwidth=15, highlightthickness=5, height=150, width=150, bg='white')
        image.image = render
        image.place(x=450, y=300)


    def query(self, event=None):
        global rep
        rep = filedialog.askopenfilenames()
        img = cv2.imread(rep[0])
        
        # Resize the image and create a Tkinter-compatible photo image
        resized_img = cv2.resize(img, (200, 200))
        resized_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
        self.photo = ImageTk.PhotoImage(image=Image.fromarray(resized_img))

        # Place the image label at the specified location
        image_label = Label(self, image=self.photo, borderwidth=15, highlightthickness=5, bg='pink')
        image_label.place(x=450, y=300)


   

    def classification(self, event=None):
        global T, rep
        clas1 = [item[10:-1] for item in sorted(glob("./dataset/*/"))]
        from keras.preprocessing import image                  
        from tqdm import tqdm
        def path_to_tensor(img_path, width=224, height=224):
            print(img_path)
            img = image.load_img(img_path, target_size=(width, height))
            x = image.img_to_array(img)
            return np.expand_dims(x, axis=0)
        def paths_to_tensor(img_paths, width=224, height=224):
            list_of_tensors = [path_to_tensor(img_paths, width, height)]
            return np.vstack(list_of_tensors)
        from tensorflow.keras.models import load_model
        model = load_model('trained_model_DNN1.h5')
        test_tensors = paths_to_tensor(rep[0])/255
        pred = model.predict(test_tensors)
        x = np.argmax(pred)
        
        # Determine the predicted folder
        predicted_folder = clas1[x]
        
        # Display information based on the predicted folder
        vitamin_info = ""
        if predicted_folder == 'vitaminA':
            vitamin_info = "vitamin A is found in fish, organ meats (such as liver), dairy products, and eggs Provitamin A carotenoids are turned into vitamin A by your body. They are found in fruits, vegetables, and other plant-based products...."
            
                          
        elif predicted_folder == 'vitaminB':
            vitamin_info = "B vitamins are found in abundance in meat, eggs, and dairy products.Processed carbohydrates such as sugar and white flour tend to have lower B vitamin than their unprocessed counterparts..."

        elif predicted_folder == 'vitaminb1':
            vitamin_info = "Vitamin B1 is also known as thiamine. Vitamin B1 is found in foods such as cereals, whole grains, meat, nuts, beans, and peas. Vitamin B1 is important in the breakdown of carbohydrates from foods into products needed by the body...."

        elif predicted_folder == 'vitaminb12':
            vitamin_info = "Vitamin B12 (cobalamin) is essential for the health of nerve tissue, brain function, and red blood cells. Sources include meat, eggs, and some yeast products. People with a B12 deficiency may need supplements. Signs of a deficiency include headaches and fatigue...."


        elif predicted_folder == 'vitaminD':
            vitamin_info = "Vitamin D isn't naturally found in many foods, but you can get it from fortified milk, fortified cereal, and fatty fish such as salmon, mackerel and sardines. Your body also makes vitamin D when direct sunlight converts a chemical in your skin into an active form of the vitamin (calciferol)...."

        elif predicted_folder == 'vitamink':
            vitamin_info = "Vitamin K is a group of vitamins found in some green vegetables. Vitamins K1 (phytonadione) and K2 (menaquinone) are commonly available as supplements. Vitamin K is an essential vitamin needed by the body for blood clotting, bone building, and other important processes. It's found in leafy green vegetables, broccoli, and Brussels sprouts...."     

            


            
        # Add more conditions for other folders
        
        print('Given image is  = ' + predicted_folder)
        res = predicted_folder
        T = Text(self, height=10, width=80)
        T.place(x=400, y=600)
        T.insert(END, res)

        # Display information about the vitamin
        T.insert(END, "\n\n" + vitamin_info)


    
    

    

        
        



        
# Create a Tkinter window
root = Tk()
root.geometry("800x600")  # Set the size of the window
app = Window(root)
root.mainloop()






        

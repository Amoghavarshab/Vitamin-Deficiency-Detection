import os

# Specify the directory containing your images
data_directory = './data'  # Change this to your actual directory path

# List to hold names of the image files
image_names = []

# Iterate through the files in the specified directory
for filename in os.listdir(data_directory):
    # Check if the file is an image (you can add more extensions if needed)
    if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')):
        image_names.append(filename)
        print(filename)  # Print the name of the image file

# Save the names to names.txt
with open('names.txt', 'w') as file:
    for name in image_names:
        file.write(name + '\n')

print(f"Image names saved to 'names.txt'.")

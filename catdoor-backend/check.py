import os

directory_path = "cropped"  # Replace with the path to your directory

# Get a list of all files in the directory
file_names = os.listdir(directory_path)

# Print the list of file names
for file_name in file_names:
    print(file_name)
import zipfile
import os
import argparse

# Set up argument parsing
parser = argparse.ArgumentParser(
    description="Unzip a file to the same directory with the same name."
)
parser.add_argument("zip_file", help="Path to the zip file")

# Parse the command-line arguments
args = parser.parse_args()

# Get the path to the zip file and its directory
zip_file_path = args.zip_file
directory = os.path.dirname(zip_file_path)

# Remove the .zip extension to get the new directory/folder name
folder_name = os.path.splitext(os.path.basename(zip_file_path))[0]
destination_folder = os.path.join(directory, folder_name)

# Ensure the destination folder exists
os.makedirs(destination_folder, exist_ok=True)

# Unzip the file
with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
    zip_ref.extractall(destination_folder)

print(f"Contents extracted to {destination_folder}")

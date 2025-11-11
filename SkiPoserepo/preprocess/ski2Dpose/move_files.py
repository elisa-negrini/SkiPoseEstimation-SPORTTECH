import os
import shutil

def move_files_to_parent_folder(folder_path):
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            # Create the source and destination paths
            source_path = os.path.join(root, file)
            destination_path = os.path.join(folder_path, file)

            # Move the file to the parent folder
            shutil.move(source_path, destination_path)

if __name__ == "__main__":
    # Specify the folder path where you want to move files
    folder_path = '/path/to/your/folder'

    # Call the function to move files
    move_files_to_parent_folder(folder_path)
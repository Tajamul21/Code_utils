import os

def rename_files_with_spaces(folder_path):
    files = os.listdir(folder_path)
    for filename in files:
        if ' ' in filename:
            new_filename = filename.replace(' ', '_')
            os.rename(os.path.join(folder_path, filename), os.path.join(folder_path, new_filename))
            print(f'Renamed: {filename} -> {new_filename}')


# Call the function to rename all files with spaces in their names




def remove_dcm_extension_from_folder_files(folder_path):
    # List all files in the folder
    files = os.listdir(folder_path)

    # Loop through the files and rename them
    for filename in files:
        if filename.endswith('.dcm.png'):
            new_filename = filename.replace('.dcm', '')
            old_path = os.path.join(folder_path, filename)
            new_path = os.path.join(folder_path, new_filename)
            os.rename(old_path, new_path)
            print(f'Renamed: {filename} -> {new_filename}')



# Specify the path to the folder containing the files
folder_path = '/home/tajamul/scratch/DA/Datasets/Voc_Data/INBreast/test2017'



# Call the function to remove ".dcm" extension from filenames in the folder
remove_dcm_extension_from_folder_files(folder_path)
# rename_files_with_spaces(folder_path)
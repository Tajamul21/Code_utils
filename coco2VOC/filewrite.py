import os

def write_image_paths_to_file(root_folder, output_file):
    with open(output_file, 'w') as file:
        for subdir, _, files in os.walk(root_folder):
            for name in files:
                if name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                    relative_path = os.path.relpath(os.path.join(subdir, name), root_folder)
                    file.write(relative_path + '\n')

if __name__ == "__main__":
    root_folder = '/home/tajamul/scratch/DA/DATA/Voc_Data/cityscapes/JPEGImages'  # Replace with your folder path
    output_file = '/home/tajamul/scratch/DA/DATA/Voc_Data/cityscapes/ImageSets/Main/val.txt'
    write_image_paths_to_file(root_folder, output_file)
import imageio
import os
from PIL import Image

# Directory containing your images
image_folder = '/home/tajamul/scratch/Domain_Adaptation/DA_expts/MRT/MAE_Analysis/mr_0.8_no_flip/visuals'

# Output directory for GIFs
output_gif_dir = '/home/tajamul/scratch/Domain_Adaptation/DA_expts/MRT/MAE_Analysis/mr_0.8_no_flip/gifs'

# Desired range of epochs
start_epoch = 0
end_epoch = 40

# Get the list of image files in the directory
images = [
    os.path.join(image_folder, img)
    for img in sorted(os.listdir(image_folder))
    if img.endswith(".png") and "visualizations_combined_epoch_" in img
]

# Filter out invalid filenames
valid_images = []
for img in images:
    try:
        # Extract epoch and image number from the filename
        epoch = int(img.split("_")[-3])
        img_number = int(img.split("_")[-1].split(".")[0])
        valid_images.append((img, epoch, img_number))
    except ValueError:
        print(f"Invalid filename: {img}")

# Sort the valid images by epoch and image number
valid_images.sort(key=lambda x: (x[1], x[2]))

# Create GIFs for each image number
for img_number in set(x[2] for x in valid_images):
    # Filter valid images for the current image number
    filtered_images = [
        img[0] for img in valid_images if img[2] == img_number
    ]

    # Output GIF file path
    output_gif_path = os.path.join(output_gif_dir, f"image_{img_number}_result.gif")

    # Create the GIF
    with imageio.get_writer(output_gif_path, fps=2.5) as gif_writer:
        for img_path in filtered_images:
            img = Image.open(img_path)
            print(img_number)
            gif_writer.append_data(img)

    print(f"GIF created successfully at: {output_gif_path}")

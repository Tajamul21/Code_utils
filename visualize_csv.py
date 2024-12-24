import cv2
import numpy as np
import csv
import os

# Function to draw bounding boxes on images and save them
def visualize_bounding_boxes(csv_file, image_folder, output_folder):
    with open(csv_file, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            file_name = row['img_path']
            confidence_score = float(row['mal_score'])
            bounding_box = np.fromstring(row['mal_box'].replace('[', '').replace(']', ''), sep=' ')
            image_path = os.path.join(image_folder, file_name)
            
            # Load the image
            image = cv2.imread(image_path)
            
            # Draw the bounding box on the image
            x1, y1, x2, y2 = bounding_box.astype(int)
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Save the new image with bounding box drawn
            output_path = os.path.join(output_folder, file_name)
            cv2.imwrite(output_path, image)
            
            print(f"Visualized bounding box for {file_name} and saved to {output_path}")

# Example usage
csv_file = '/home/tajamul/scratch/DA/Datasets/Dmaster_Data/IRCH/dmaster_pred/test_irch_pred_dmaster.csv'
image_folder = '/home/tajamul/scratch/DA/Datasets/Dmaster_Data/IRCH/Mammo_PNG'
output_folder = '/home/tajamul/scratch/DA/Datasets/Dmaster_Data/IRCH/Mammo_PNG_check'




visualize_bounding_boxes(csv_file, image_folder, output_folder)


# import pandas as pd
# import numpy as np

# # Load the first CSV file with uhid, laterality, and img_path
# df1 = pd.read_csv('/home/tajamul/scratch/DA/Datasets/Dmaster_Data/IRCH/tajamul_data/irch_gt.csv')

# # Load the second CSV file with file_name_image_id, confidence_score, and bounding_box
# df2 = pd.read_csv('/home/tajamul/scratch/DA/UDA/MRT/MRT-release-froc/evaluation_irch_with_names.csv')

# # Merge the two dataframes on the common column 'img_path'
# merged_df = pd.merge(df1, df2, left_on='img_path', right_on='filename')

# # Assign 'mal_score' as 'confidence_score' directly
# merged_df['mal_score'] = merged_df['confidence_score']

# # Convert bounding_box string to NumPy array and assign it to 'mal_box'
# merged_df['mal_box'] = merged_df['bounding_box'].apply(lambda x: np.fromstring(x.replace('[', '').replace(']', ''), sep=' '))

# # Select columns for the final CSV file
# final_df = merged_df[['uhid', 'laterality', 'img_path', 'mal_score', 'mal_box']]

# # Save the final dataframe to a new CSV file
# final_df.to_csv('/home/tajamul/scratch/DA/Datasets/Dmaster_Data/IRCH/dmaster_pred/test_irch_pred_dmaster.csv', index=False)

# print(f"New CSV file has been created.")
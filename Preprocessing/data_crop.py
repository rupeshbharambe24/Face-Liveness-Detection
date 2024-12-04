import os
from PIL import Image

def crop_images_in_subfolders(main_folder, target_width, target_height):
    """
    Recursively crop images in all subfolders of the main folder to the specified size (center crop)
    and save them back to their respective folders.

    Parameters:
    - main_folder: Path to the main folder containing subfolders with images.
    - target_width: Width of the cropped image.
    - target_height: Height of the cropped image.
    """
    for root, _, files in os.walk(main_folder):
        for file_name in files:
            file_path = os.path.join(root, file_name)

            # Check if file is an image
            if file_name.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'tiff', 'gif')):
                try:
                    with Image.open(file_path) as img:
                        # Ensure the image is large enough
                        width, height = img.size
                        if width < target_width or height < target_height:
                            print(f"Image {file_path} is too small to crop to {target_width}x{target_height}")
                            continue

                        # Calculate cropping box for center crop
                        left = (width - target_width) // 2
                        upper = (height - target_height) // 2
                        right = left + target_width
                        lower = upper + target_height

                        # Perform cropping
                        cropped_img = img.crop((left, upper, right, lower))

                        # Overwrite the original image
                        cropped_img.save(file_path)
                        print(f"Cropped and replaced: {file_path}")
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
            else:
                print(f"Skipped non-image file: {file_path}")

# Configuration
main_folder = r"C:\Users\rupes\Downloads\Telegram Desktop\facial_action_final"  # Replace with your main folder path
target_width = 640
target_height = 324

# Execute cropping
crop_images_in_subfolders(main_folder, target_width, target_height)
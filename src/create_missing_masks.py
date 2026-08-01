import os
import numpy as np
from PIL import Image
from tqdm import tqdm

def create_missing_masks(image_dir, mask_dir, image_set_file, img_ext=".jpg", mask_ext=".png"):
    """
    Creates black masks for images that don't have corresponding segmentation masks.
    
    Args:
        image_dir (str): Path to directory containing original images
        mask_dir (str): Path to directory containing segmentation masks
        image_set_file (str): Path to file containing image names
        img_ext (str): Extension for image files (default: ".jpg")
        mask_ext (str): Extension for mask files (default: ".png")
    
    Returns:
        dict: Statistics about the operation
    """
    
    # Verify directories exist
    if not os.path.exists(image_dir):
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    if not os.path.exists(mask_dir):
        raise FileNotFoundError(f"Mask directory not found: {mask_dir}")
    if not os.path.exists(image_set_file):
        raise FileNotFoundError(f"Image set file not found: {image_set_file}")
    
    # Load image names from the text file
    with open(image_set_file, 'r') as f:
        image_names = [line.strip() for line in f.readlines() if line.strip()]
    
    print(f"Found {len(image_names)} image names in {image_set_file}")
    
    # Check which images exist and which masks are missing
    existing_images = []
    missing_masks = []
    existing_masks = []
    missing_images = []
    
    for name in image_names:
        img_path = os.path.join(image_dir, name + img_ext)
        mask_path = os.path.join(mask_dir, name + mask_ext)
        
        if os.path.exists(img_path):
            existing_images.append(name)
            if os.path.exists(mask_path):
                existing_masks.append(name)
            else:
                missing_masks.append(name)
        else:
            missing_images.append(name)
    
    print(f"Existing images: {len(existing_images)}")
    print(f"Existing masks: {len(existing_masks)}")
    print(f"Missing masks: {len(missing_masks)}")
    print(f"Missing images: {len(missing_images)}")
    
    if missing_images:
        print(f"\nWarning: {len(missing_images)} images listed in {image_set_file} are missing from {image_dir}")
        print("First few missing images:", missing_images[:5])
    
    # Get image dimensions from the first existing image
    if existing_images:
        first_img_path = os.path.join(image_dir, existing_images[0] + img_ext)
        with Image.open(first_img_path) as img:
            img_width, img_height = img.size
        print(f"Using image dimensions: {img_width}x{img_height}")
    else:
        raise ValueError("No existing images found to determine dimensions")
    
    # Create black masks for missing ones
    created_masks = 0
    skipped_masks = 0
    
    if missing_masks:
        print(f"\nCreating {len(missing_masks)} black masks...")
        
        for name in tqdm(missing_masks, desc="Creating masks"):
            img_path = os.path.join(image_dir, name + img_ext)
            mask_path = os.path.join(mask_dir, name + mask_ext)
            
            if os.path.exists(img_path):
                # Get the actual image dimensions
                with Image.open(img_path) as img:
                    current_width, current_height = img.size
                
                # Create a black mask (all pixels = 0, representing background class)
                black_mask = np.zeros((current_height, current_width), dtype=np.uint8)
                
                # Convert to PIL Image and save
                mask_image = Image.fromarray(black_mask, mode='L')  # 'L' for grayscale
                mask_image.save(mask_path)
                created_masks += 1
            else:
                skipped_masks += 1
                print(f"Skipped {name}: corresponding image not found")
    
    # Final statistics
    stats = {
        'total_names_in_file': len(image_names),
        'existing_images': len(existing_images),
        'existing_masks_before': len(existing_masks),
        'missing_masks_before': len(missing_masks),
        'created_masks': created_masks,
        'skipped_masks': skipped_masks,
        'missing_images': len(missing_images),
        'existing_masks_after': len(existing_masks) + created_masks
    }
    
    print(f"\n=== SUMMARY ===")
    print(f"Total image names in file: {stats['total_names_in_file']}")
    print(f"Existing images: {stats['existing_images']}")
    print(f"Existing masks (before): {stats['existing_masks_before']}")
    print(f"Missing masks (before): {stats['missing_masks_before']}")
    print(f"Created black masks: {stats['created_masks']}")
    print(f"Skipped (no corresponding image): {stats['skipped_masks']}")
    print(f"Total masks after operation: {stats['existing_masks_after']}")
    
    return stats

def verify_mask_creation(image_dir, mask_dir, image_set_file, img_ext=".jpg", mask_ext=".png"):
    """
    Verifies that all images now have corresponding masks.
    
    Args:
        image_dir (str): Path to directory containing original images
        mask_dir (str): Path to directory containing segmentation masks
        image_set_file (str): Path to file containing image names
        img_ext (str): Extension for image files (default: ".jpg")
        mask_ext (str): Extension for mask files (default: ".png")
    
    Returns:
        bool: True if all images have masks, False otherwise
    """
    
    with open(image_set_file, 'r') as f:
        image_names = [line.strip() for line in f.readlines() if line.strip()]
    
    missing_pairs = []
    
    for name in image_names:
        img_path = os.path.join(image_dir, name + img_ext)
        mask_path = os.path.join(mask_dir, name + mask_ext)
        
        if os.path.exists(img_path) and not os.path.exists(mask_path):
            missing_pairs.append(name)
    
    if missing_pairs:
        print(f"Still missing {len(missing_pairs)} masks:")
        print(missing_pairs[:10])  # Show first 10
        return False
    else:
        print("✅ All images now have corresponding masks!")
        return True

    

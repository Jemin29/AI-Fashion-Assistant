import os
import sys
from pathlib import Path
import numpy as np
from PIL import Image

try:
    import h5py
except ImportError:
    print("h5py not installed. Installing it or installing requirements first.")
    sys.exit(1)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEEPFASHION_SAMPLE_DIR = PROJECT_ROOT / "datasets" / "deepfashion" / "sample"
FASHIONGEN_SAMPLE_DIR = PROJECT_ROOT / "datasets" / "fashiongen" / "sample"


def generate_deepfashion_sample(num_records=250):
    print("Generating DeepFashion sample structure...")
    img_dir = DEEPFASHION_SAMPLE_DIR / "img"
    anno_dir = DEEPFASHION_SAMPLE_DIR / "Anno"
    eval_dir = DEEPFASHION_SAMPLE_DIR / "Eval"

    img_dir.mkdir(parents=True, exist_ok=True)
    anno_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)

    # Categories
    categories = [
        "Anorak", "Blazer", "Blouse", "Bomber", "Button-Down", "Cardigan", "Flannel", "Halter", "Henley", "Hoodie",
        "Jacket", "Jersey", "Parka", "Peacoat", "Poncho", "Sweater", "Tank", "Tee", "Top", "Vest",
        "Cagoule", "Cape", "Coat", "Coverup", "Robe", "Shawl", "Kimono", "Nightgown", "Onesie", "Jumpsuit",
        "Romper", "Sarong", "Suit", "Tunic", "Dress", "Skort", "Skirt", "Shorts", "Jeans", "Pants",
        "Leggings", "Joggers", "Sweatpants", "Sweatshorts", "Trunks", "Underwear", "Socks", "Tights", "Footwear", "Accessories"
    ]
    
    # Write Category Cloth list
    with open(anno_dir / "list_category_cloth.txt", "w") as f:
        f.write(f"{len(categories)}\n")
        f.write("category_name  category_type\n")
        for i, cat in enumerate(categories):
            f.write(f"{cat}  1\n")

    # Write Attribute Cloth list
    attributes = [f"attr_{i:04d}" for i in range(1000)]
    with open(anno_dir / "list_attr_cloth.txt", "w") as f:
        f.write(f"{len(attributes)}\n")
        f.write("attribute_name  attribute_type\n")
        for i, attr in enumerate(attributes):
            f.write(f"{attr}  1\n")

    # Create images and entries
    cat_img_lines = []
    attr_img_lines = []
    bbox_lines = []
    landmarks_lines = []
    eval_lines = []

    for i in range(1, num_records + 1):
        # Pick category
        cat_idx = (i - 1) % len(categories)
        cat_name = categories[cat_idx]
        
        # Directory for category
        cat_subdir = img_dir / cat_name
        cat_subdir.mkdir(parents=True, exist_ok=True)
        
        img_filename = f"img_{i:08d}.jpg"
        img_path = cat_subdir / img_filename
        
        # Save a solid color JPEG image (100x100)
        img = Image.new("RGB", (256, 256), color=(i * 10 % 255, i * 20 % 255, i * 30 % 255))
        img.save(img_path, "JPEG")
        
        # Rel path from datasets/deepfashion/sample/
        rel_img_path = f"img/{cat_name}/{img_filename}"
        
        # Accumulate lists
        cat_img_lines.append(f"{rel_img_path}  {cat_idx + 1}")
        
        # Attribute vector: random 1 present, rest -1
        attr_vals = [-1] * 1000
        attr_vals[i % 1000] = 1
        attr_str = " ".join(map(str, attr_vals))
        attr_img_lines.append(f"{rel_img_path}  {attr_str}")
        
        # Bbox: x1 y1 x2 y2
        bbox_lines.append(f"{rel_img_path}  10 10 240 240")
        
        # Landmarks: 6 points of x y vis
        # Order: left_collar, right_collar, left_sleeve, right_sleeve, left_hem, right_hem
        lm_str = " ".join(f"{10 + idx * 20:03d} {50 + idx * 20:03d} 1" for idx in range(6))
        landmarks_lines.append(f"{rel_img_path}  {lm_str}")
        
        # Split partition: train | val | test
        split = "train" if i % 3 == 0 else ("val" if i % 3 == 1 else "test")
        eval_lines.append(f"{rel_img_path}  {split}")

    # Save to files
    with open(anno_dir / "list_category_img.txt", "w") as f:
        f.write(f"{num_records}\n")
        f.write("image_name  category_label\n")
        f.write("\n".join(cat_img_lines) + "\n")

    with open(anno_dir / "list_attr_img.txt", "w") as f:
        f.write(f"{num_records}\n")
        f.write(f"{len(attributes)}\n")
        f.write("image_name  attribute_labels\n")
        f.write("\n".join(attr_img_lines) + "\n")

    with open(anno_dir / "list_bbox.txt", "w") as f:
        f.write(f"{num_records}\n")
        f.write("image_name  x1 y1 x2 y2\n")
        f.write("\n".join(bbox_lines) + "\n")

    with open(anno_dir / "list_landmarks.txt", "w") as f:
        f.write(f"{num_records}\n")
        f.write("image_name  landmark_info\n")
        f.write("\n".join(landmarks_lines) + "\n")

    with open(eval_dir / "list_eval_partition.txt", "w") as f:
        f.write(f"{num_records}\n")
        f.write("image_name  evaluation_status\n")
        f.write("\n".join(eval_lines) + "\n")

    print(f"Successfully generated {num_records} DeepFashion sample records.")


def generate_fashiongen_sample(num_records=250):
    print("Generating FashionGen sample HDF5 dataset...")
    FASHIONGEN_SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    hdf5_file_path = FASHIONGEN_SAMPLE_DIR / "fashiongen_256_256_train.h5"

    # Prepare datasets
    images = np.random.randint(0, 255, size=(num_records, 256, 256, 3), dtype=np.uint8)
    
    categories = [
        b"T-Shirts", b"Shirts", b"Sweaters", b"Coats", b"Jackets", b"Jeans", b"Pants", b"Shorts", b"Skirts", b"Dresses"
    ]
    subcategories = [
        b"Crewneck", b"Formal Shirt", b"V-Neck", b"Trench Coat", b"Leather Jacket", b"Slim Jeans", b"Chinos", b"Denim Shorts", b"Mini Skirt", b"Maxi Dress"
    ]
    descriptions = [
        f"Description for fashion item {i}".encode("utf-8") for i in range(num_records)
    ]
    genders = [
        b"men" if i % 2 == 0 else b"women" for i in range(num_records)
    ]
    
    selected_categories = [categories[i % len(categories)] for i in range(num_records)]
    selected_subcategories = [subcategories[i % len(subcategories)] for i in range(num_records)]

    with h5py.File(hdf5_file_path, "w") as f:
        f.create_dataset("input_image", data=images)
        f.create_dataset("input_description", data=descriptions)
        f.create_dataset("input_category", data=selected_categories)
        f.create_dataset("input_subcategory", data=selected_subcategories)
        f.create_dataset("input_gender", data=genders)

    print(f"Successfully generated {num_records} FashionGen sample records in HDF5 format.")


if __name__ == "__main__":
    generate_deepfashion_sample()
    generate_fashiongen_sample()

import os
import numpy as np
from PIL import Image
from tqdm import tqdm

# ------------------------------
# Define part-to-class mapping
# ------------------------------
PART_CLASS_MAP = {
    "skin": 1,
    "hair": 2,
    "eye": 3,
    "nose": 4,
    "mouth": 5,
    "ear": 6,
    "hat": 7,
    "glasses": 8,
    "eyebrow": 9,
    "lip": 10,
    "neck": 11,
    "cloth": 12,
    "beard": 13,
    # Add more mappings as needed
}

# ------------------------------
# Function to process split
# ------------------------------
def process_split(split_name: str, num_ids: int):
    input_mask_dir = f"data/CelebAMask-HQ-SPLIT/images_and_masks/{split_name}/masks"
    output_mask_dir = f"data/CelebAMask-HQ-SPLIT/images_and_masks/{split_name}/semantic_masks"
    os.makedirs(output_mask_dir, exist_ok=True)

    print(f"\n Processing split: {split_name} → generating {num_ids} semantic masks")

    for img_id in tqdm(range(num_ids), desc=f"{split_name} split"):
        base_id = f"{img_id:05d}"
        semantic_mask = np.zeros((512, 512), dtype=np.uint8)

        for fname in os.listdir(input_mask_dir):
            if fname.startswith(base_id):
                for part_name, class_id in PART_CLASS_MAP.items():
                    if part_name in fname.lower():
                        path = os.path.join(input_mask_dir, fname)
                        part_mask = Image.open(path).convert("L")
                        part_arr = np.array(part_mask)
                        semantic_mask[part_arr > 0] = class_id
                        break  # Stop after the first matching part

        out_path = os.path.join(output_mask_dir, f"{base_id}.png")
        Image.fromarray(semantic_mask).save(out_path)

    print(f"✅ Done writing semantic masks for '{split_name}' to: {output_mask_dir}")

# ------------------------------
# Run for both train and test
# ------------------------------
if __name__ == "__main__":
    process_split("train", 2700)
    process_split("test", 300)
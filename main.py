import cv2
import numpy as np
import os

# =========================
# Step 1: Load images
# =========================
img_paths = [
    "data/dark.png",
    "data/mild.png",
    "data/bright.png"
]

images = []

for path in img_paths:
    img = cv2.imread(path)
    
    if img is None:
        print(f"Error loading {path}")
        exit()
    
    img = img.astype(np.float32) / 255.0
    images.append(img)

print("✅ Images loaded successfully")


# =========================
# Resize all images
# =========================
base_shape = images[0].shape[:2]

resized_images = []
for img in images:
    resized = cv2.resize(img, (base_shape[1], base_shape[0]))
    resized_images.append(resized)

images = resized_images


# =========================
# Step 2: Simple Average Fusion
# =========================
hdr = np.zeros_like(images[0])

for img in images:
    hdr += img

hdr = hdr / len(images)


# =========================
# Step 3: Convert back
# =========================
hdr_output = (hdr * 255).astype(np.uint8)


# =========================
# Step 4: Save + Display
# =========================
os.makedirs("outputs", exist_ok=True)

cv2.imwrite("outputs/hdr_day1.jpg", hdr_output)

cv2.imshow("LumiFuse - Day 1 Output", hdr_output)
cv2.waitKey(0)
cv2.destroyAllWindows()
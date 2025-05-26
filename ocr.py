import os
import cv2
import numpy as np
import easyocr
import pandas as pd
import matplotlib.pyplot as plt
import re

# === Setup Paths ===
image_path = "images/map.png"  # Path to your map image
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

# === Step 1: Load and Resize Image ===
image = cv2.imread(image_path) 
if image is None:
    raise FileNotFoundError("Image not found at path: " + image_path)

# Optional resize (scale down if large)
scale_percent = 100  # Set to 50 if image is too big
width = int(image.shape[1] * scale_percent / 100)
height = int(image.shape[0] * scale_percent / 100)
resized_image = cv2.resize(image, (width, height))
cv2.imwrite(os.path.join(output_dir, "1_resized_input.png"), resized_image)

# === Step 2: Run CRAFT + CRNN (EasyOCR) ===
reader = easyocr.Reader(['en'], gpu=False, verbose=False)  # Fix: disable Unicode progress bar
results = reader.readtext(resized_image, detail=1, paragraph=False)


# === Step 3: Filter and Classify Text ===
characters = []
numbers = []

for bbox, text, conf in results:
    if conf < 0.4 or text.strip() == "":
        continue
    clean_text = text.strip(",. ")
    
    # Heuristic filters
    if re.fullmatch(r'[A-Za-z][A-Za-z0-9_\- ]{2,}', clean_text):  # Names
        characters.append(clean_text)
    elif re.fullmatch(r'\d{2,4}', clean_text):  # Numbers like 123, 2023
        numbers.append(clean_text)

# === Step 4: Save Extracted Data to CSV ===
characters = sorted(set(characters))
numbers = sorted(set(numbers), key=lambda x: int(x))

# Padding to equal length
max_len = max(len(characters), len(numbers))
characters += [None] * (max_len - len(characters))
numbers += [None] * (max_len - len(numbers))

df = pd.DataFrame({
    "Character Name": characters,
    "Number": numbers
})
csv_path = os.path.join(output_dir, "2_extracted_text.csv")
df.to_csv(csv_path, index=False)
print("CSV saved at:", csv_path)

# === Step 5: Annotate Image with Detected Text ===
annotated = resized_image.copy()
for bbox, text, conf in results:
    if conf < 0.4:
        continue
    pts = np.array([tuple(map(int, pt)) for pt in bbox], dtype=np.int32)
    cv2.polylines(annotated, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
    cv2.putText(annotated, text, pts[0], cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

annotated_path = os.path.join(output_dir, "3_annotated_result.png")
cv2.imwrite(annotated_path, annotated)
print("Annotated image saved at:", annotated_path)

# === Optional: Display the Annotated Result ===
plt.figure(figsize=(20, 12))
plt.imshow(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
plt.title("OCR Result: CRAFT + CRNN via EasyOCR")
plt.axis('off')
plt.show()
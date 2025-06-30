import os
from PIL import Image

img_path = "/shared/ssd_30T/home/seid/projects/VLM_Homework_627/data/test/0a8d486f-1aa6-4fcf-b7be-4bf04fc8628b.png"
img = Image.open(img_path)
print("Image mode:", img.mode)
print("Image size:", img.size)

test_dir = "./data/test"
image_files = [f for f in os.listdir(test_dir) if f.endswith(".png")]
#print("First Image: ", image_files[0])

for fname in image_files[:3]:
    path = os.path.join(test_dir, fname)
    print(f"Path of {fname}", path)
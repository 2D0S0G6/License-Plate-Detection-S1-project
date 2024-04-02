import subprocess

modules = ["opencv-python", "Pillow", "pytesseract"]

for module in modules:
    subprocess.run(["pip", "install", module])

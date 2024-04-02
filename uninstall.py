import subprocess

modules = ["opencv-python", "Pillow", "pytesseract"]

for module in modules:
    subprocess.run(["pip", "uninstall", "-y", module])

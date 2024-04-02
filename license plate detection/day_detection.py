import cv2  # OpenCV for computer vision tasks
from PIL import Image  # Image processing library
import pytesseract  # Python wrapper for Tesseract OCR engine
import os  # Operating system dependent functionality

# Get the current directory path
current_dir = os.path.dirname(os.path.abspath(__file__))

# Loading the pre-trained cascade classifier for detecting license plates
plate_cascade = cv2.CascadeClassifier(os.path.join(current_dir, 'num_plate.xml'))

# Setting the Tesseract executable path based on the operating system
if os.name == 'posix':  # For Linux or Mac OS
    pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'
else:  # For Windows
    pytesseract.pytesseract.tesseract_cmd = os.path.join(current_dir, 'tesseract.exe')

# Function to detect license plates in an image
def detect_plate_img(img):
    plate_img = img.copy()  # Create a copy of the input image
    plate_rect = plate_cascade.detectMultiScale(plate_img, scaleFactor=1.3, minNeighbors=7)
    # Detect potential plate regions in the image using the pre-trained cascade classifier

    for (x, y, w, h) in plate_rect:
        # Iterate through the detected plate regions
        cv2.rectangle(plate_img, (x, y), (x+w, y+h), (0, 255, 255), 3)
        # Draw yellow rectangles around the detected plate regions on the image

    return plate_img, plate_rect
    # Return the modified image with rectangles drawn around detected plates and the plate regions' coordinates

# Function to detect license plates in a video
def detect_plate_video(video_path):
    # Capture video using OpenCV
    cam = cv2.VideoCapture(video_path)

    # Create base directory 'Number_Plate_detection' in the current folder
    base_directory = os.path.join(current_dir, 'Number_Plate_detection')
    os.makedirs(base_directory, exist_ok=True)

    # Directories for detected frames and text file within the base directory
    output_folder = os.path.join(base_directory, 'detected_frames')
    os.makedirs(output_folder, exist_ok=True)
    output_file = os.path.join(base_directory, 'number_plates.txt')

    # Set to store unique plate codes
    unique_plates = set()

    while cam.isOpened():
        ret, frame = cam.read()  # Read frames from the video capture

        if ret:  # If a frame is successfully read
            frame = cv2.resize(frame, (800, 600))  # Resize the frame for better processing

            # Detect license plates in the resized frame
            fr, plate_rect = detect_plate_img(frame)
            cv2.imshow('video', fr)  # Display the frame with detected plates
            key = cv2.waitKey(1) & 0xFF  # Capture keyboard input

            # Check for 'q' key or 'Esc' key (27) to exit the video loop
            if key == ord('q') or key == 27:
                break

            for (x, y, w, h) in plate_rect:  # Iterate through detected plate coordinates
                plate_roi = frame[y:y + h, x:x + w]  # Extract the region of interest (plate)
                plate_code = str(x) + '_' + str(y)  # Unique code based on plate coordinates

                # Process unique license plates not previously encountered
                if plate_code not in unique_plates:
                    output_path = os.path.join(output_folder, f'detected_plate_{plate_code}.jpg')  # File path for saving the detected plate image
                    cv2.imwrite(output_path, plate_roi)  # Save the detected plate as an image
                    unique_plates.add(plate_code)  # Add the plate code to track uniqueness

                    try:
                        image = Image.fromarray(plate_roi)  # Convert plate ROI to PIL Image format
                        text = pytesseract.image_to_string(image)  # Perform OCR on the plate image
                        with open(output_file, 'a') as output_text:
                            output_text.write(f"'detected_plate_{plate_code}.jpg' : '{text.strip()}'\n")  # Write plate text to a file
                    except Exception as e:
                        print(f"Error processing 'detected_plate_{plate_code}.jpg': {e}")  # Handle OCR processing errors
        else:
            break  # Break the loop if no more frames are available in the video


    cam.release()
    cv2.destroyAllWindows()

    # Additional code for file processing and user interaction
    # (Prompting user input for text file processing)

    # Processing the text file to remove invalid lines
    try:
        with open(output_file, 'r') as file:
            lines = file.readlines()  # Read lines from the file

        # Filtering out invalid lines from the file
        filtered_lines = [line for line in lines if not (line.endswith("''\n") or '()' in line)]

        # Remove unwanted parentheses from each line
        filtered_lines = [line.replace('(', '').replace(')', '') for line in filtered_lines]

        with open(output_file, 'w') as file:
            file.writelines(filtered_lines)  # Write filtered lines back to the file

        # Added print statement to indicate successful creation of the text file
        print("Text file with all the number are created successfully!")

    except FileNotFoundError:
        print("File not found. Please provide a valid file path.")
    except Exception as e:
        print(f"An error occurred: {e}")  # Handle other file processing errors


# User input to choose between image or video for plate detection
input_type = input("Enter 'image' or 'video' to detect number plates: ").lower()

# Conditionals based on user input
if input_type == 'image':
    # If user chooses image: prompt for image path and perform plate detection
    image_path = input("Enter the path of the image: ")
    img = cv2.imread(image_path)  # Read the image
    detected_img, plate_rect = detect_plate_img(img)  # Detect plates in the image

    # Display detected plates on the image
    cv2.imshow('Detected Plates', detected_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print("Number plates detected successfully!")

    # Extract text from plates and write to a text file
    output_file = os.path.join(current_dir, 'detected_plate_text.txt')  # Define output file path
    with open(output_file, 'w') as file:
        for (x, y, w, h) in plate_rect:
            plate_roi = img[y:y + h, x:x + w]  # Extract region of interest (ROI) of the plate
            plate_code = str(x) + '_' + str(y)  # Generate a unique code for the plate

            try:
                image = Image.fromarray(plate_roi)  # Convert plate ROI to PIL Image
                text = pytesseract.image_to_string(image)  # Perform OCR to extract text
                file.write(f"'detected_plate_{plate_code}.jpg' : '{text.strip()}'\n")  # Write plate text to file
            except Exception as e:
                print(f"Error processing 'detected_plate_{plate_code}.jpg': {e}")  # Handle OCR errors

    print(f"Text extracted from plates saved in '{output_file}'.")  # Indicate successful text extraction

elif input_type == 'video':
    # If user chooses video: prompt for video path and perform plate detection
    video_path = input("Enter the path of the video: ")
    detect_plate_video(video_path)  # Detect plates in the video
    print("Number plates detected in the video. Frames saved successfully!")

else:
    # If user provides invalid input
    print("Invalid input. Please enter 'image' or 'video'.")
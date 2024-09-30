import cv2
import face_recognition
import pickle
import pyttsx3
import time

# Initialize text-to-speech engine
engine = pyttsx3.init()

# Load encodings from pickle file
pickleFilePath = 'EncodeFile.pkl'
with open(pickleFilePath, 'rb') as file:
    encodeListKnownWithPersonName = pickle.load(file)

# Extract encodings and person names from the loaded data
encodeListKnown, personNameList = zip(*encodeListKnownWithPersonName)

# Initialize video capture with RTSP URL
rtsp_url = 0
videoCapture = cv2.VideoCapture(rtsp_url)

if not videoCapture.isOpened():
    print(f"Error: Failed to open video source: {rtsp_url}")
    exit()

while True:
    # Capture frame-by-frame
    ret, frame = videoCapture.read()

    if not ret:
        print("Error: Failed to capture frame from video source.")
        break

    # Resize frame for faster processing
    smallFrame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgbSmallFrame = cv2.cvtColor(smallFrame, cv2.COLOR_BGR2RGB)

    # Find all the faces and face encodings in the current frame of video
    faceLocations = face_recognition.face_locations(rgbSmallFrame)
    faceEncodings = face_recognition.face_encodings(rgbSmallFrame, faceLocations)

    faceNames = []
    for faceEncoding, (top, right, bottom, left) in zip(faceEncodings, faceLocations):
        # Compare face encoding with the encodings from pickle file
        matches = face_recognition.compare_faces(encodeListKnown, faceEncoding, tolerance=0.6)
        name = "Unknown"

        # Find the index of the face encoding with the smallest distance
        faceDistances = face_recognition.face_distance(encodeListKnown, faceEncoding)
        bestMatchIndex = faceDistances.argmin()

        if matches[bestMatchIndex]:
            name = personNameList[bestMatchIndex]
            print(f"True: {name}")
            # Speak the name using text-to-speech
            engine.say(f"Hello, {name}, is in front of the camera")
            engine.runAndWait()

            # Scale up face locations to original size
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            # Draw a rectangle around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # Display the resulting frame
    cv2.imshow('Face Recognition', frame)

    # Introduce a delay to process one frame per second
    time.sleep(1)

    # Exit loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture
videoCapture.release()
cv2.destroyAllWindows()

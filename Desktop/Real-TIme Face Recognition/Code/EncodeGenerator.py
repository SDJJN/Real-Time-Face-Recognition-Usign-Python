import cv2
import face_recognition
import os
import pickle

folderPath = "C:/Users/Sachin/Desktop/New folder/Images"
imagePathList = os.listdir(folderPath)

imgList = []
personNameList = []

# Load images and corresponding person names
for imagePath in imagePathList:
    img = cv2.imread(os.path.join(folderPath, imagePath))
    imgList.append(img)
    personName = os.path.splitext(imagePath)[0]
    personNameList.append(personName)

print(f"Number of images found: {len(imgList)}")
print("Person names:", personNameList)

def findEncodings(imagesList):
    encodeList = []
    for img in imagesList:
        # Convert image to RGB (face_recognition uses RGB)
        rgbImg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Find face encodings in the image
        faces = face_recognition.face_encodings(rgbImg)
        if len(faces) > 0:
            encodeList.append(faces[0])  # Assuming only one face per image
        else:
            encodeList.append(None)  # Handle cases where no face is found
    return encodeList

print("Encoding Started")
encodeListKnown = findEncodings(imgList)
print("Encoding Completed")

# Combine encodings with person names
encodeListKnownWithPersonName = list(zip(encodeListKnown, personNameList))

# Save encodings to a pickle file
pickleFilePath = 'EncodeFile.pkl'
with open(pickleFilePath, 'wb') as file:
    pickle.dump(encodeListKnownWithPersonName, file)

print(f"Encodings saved to {pickleFilePath}")
print(encodeListKnown)

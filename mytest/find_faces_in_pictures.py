import face_recognition
import cv2, glob

imgfiles = glob.glob("./pict/*")
for imgfile in imgfiles:
    image = face_recognition.load_image_file(imgfile)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Find all the faces in the image using the default HOG-based model.
    # This method is fairly accurate, but not as accurate as the CNN model and not GPU accelerated.
    # See also: find_faces_in_picture_cnn.py
    face_locations = face_recognition.face_locations(image)

    for face_location in face_locations:
        top, right, bottom, left = face_location
        cv2.rectangle(image, (left, top), (right, bottom), (255, 255, 255), 2)

    cv2.imshow(imgfile, image)
k = cv2.waitKey(0)
if k == 27:         # wait for ESC key to exit
    cv2.destroyAllWindows()


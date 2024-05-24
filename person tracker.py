import cv2 
import mediapipe as mp

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
draw = mp.solutions.drawing_utils
# below is what we use to detect the fist 
facecascade = cv2.CascadeClassifier('map\haarcascade_frontalface_default.xml')
OUTPUT_SIZE_WIDTH = 775
OUTPUT_SIZE_HEIGHT = 600
boxing_vid = 'map\Boxing Stock Footage _ No Copyright Footage (240p).mp4'
vid = cv2.VideoCapture(0)




while True:
    ret, frame = vid.read()
    if not ret:
        break
    results = pose.process(frame)
    # below line dter,ines quality of the video
    baseImage = cv2.resize(frame, (300, 200))

    gray = cv2.cvtColor(baseImage, cv2.COLOR_BGR2GRAY)
    # below we 
    if results.pose_landmarks:
        for landmark in results.pose_landmarks.landmark:
            h,w,c = frame.shape
            cx, cy = int(landmark.x * w), int(landmark.y * h)
            cv2.circle(frame, (cx, cy), 5, (225,0,0), -1)

    print(results.pose_landmarks)


    faces = facecascade.detectMultiScale(gray, 1.1, 4)
    resultImage = baseImage.copy()

    maxArea = 0
    x = 0
    y = 0
    w = 0
    h = 0


    #Loop over all faces and check if the area for this face is
    #the largest so far
    for (_x,_y,_w,_h) in faces:
        
            x = _x
            y = _y
            w = _w
            h = _h
            #cv2.circle(frame, (cx, cy), 5, (225,0,0), -1)

            

        #If one or more faces are found, draw a rectangle around the
        #largest face present in the picture
    rectangleColor = (0,165,255)

    if maxArea > 0 :
         cv2.rectangle(resultImage,  (x-10, y-20),(x + w+10 , y + h+20),rectangleColor,2)



   
    largeResult = cv2.resize(resultImage,(OUTPUT_SIZE_WIDTH,OUTPUT_SIZE_HEIGHT))

    cv2.imshow("Frame", frame)

# below press 'q' to exit
    key = cv2.waitKey(1)
    if key ==ord('q'):
        break
cv2.destroyAllWindows()
vid.release()
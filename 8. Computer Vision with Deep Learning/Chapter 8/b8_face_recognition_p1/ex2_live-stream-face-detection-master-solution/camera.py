import cv2

cascPath = 'haarcascade_frontalface_dataset.xml'  # dataset
faceCascade = cv2.CascadeClassifier(cascPath)

video_capture = cv2.VideoCapture(0)  # 0 for web camera live stream
#  for cctv camera'rtsp://username:password@ip_address:554/user=username_password='password'_channel=channel_number_stream=0.sdp'
#  example of cctv or rtsp: 'rtsp://mamun:123456@101.134.16.117:554/user=mamun_password=123456_channel=1_stream=0.sdp'


def camera_stream():
    font = cv2.FONT_HERSHEY_PLAIN

     # Capture frame-by-frame
    ret, frame = video_capture.read()
    img_h, img_w = frame.shape[:2]
    img = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #define a hat img
    witch = cv2.imread('witch.png')

    #get shape of witch
    original_witch_h,original_witch_w,witch_channels = witch.shape
    witch_gray = cv2.cvtColor(witch, cv2.COLOR_BGR2GRAY)

    #create mask and inverse mask of witch
    ret, original_mask = cv2.threshold(witch_gray, 10, 255, cv2.THRESH_BINARY_INV)
    original_mask_inv = cv2.bitwise_not(original_mask)
    counter = 0
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        #coordinates of face region
        face_w = w
        face_h = h
        face_x1 = x
        face_x2 = face_x1 + face_w
        face_y1 = y
        face_y2 = face_y1 + face_h

        #witch size in relation to face by scaling
        witch_width = int(1.5 * face_w)
        witch_height = int(witch_width * original_witch_h / original_witch_w)
        
        #setting location of coordinates of witch
        witch_x1 = face_x2 - int(face_w/2) - int(witch_width/2)
        witch_x2 = witch_x1 + witch_width
        witch_y1 = face_y1 - int(face_h*1.25)
        witch_y2 = witch_y1 + witch_height 

        #check to see if out of frame
        if witch_x1 < 0:
            witch_x1 = 0
        if witch_y1 < 0:
            witch_y1 = 0
        if witch_x2 > img_w:
            witch_x2 = img_w
        if witch_y2 > img_h:
            witch_y2 = img_h

        #Account for any out of frame changes
        witch_width = witch_x2 - witch_x1
        witch_height = witch_y2 - witch_y1

        #resize witch to fit on face
        witch = cv2.resize(witch, (witch_width,witch_height), interpolation = cv2.INTER_AREA)
        mask = cv2.resize(original_mask, (witch_width,witch_height), interpolation = cv2.INTER_AREA)
        mask_inv = cv2.resize(original_mask_inv, (witch_width,witch_height), interpolation = cv2.INTER_AREA)

        #take ROI for witch from background that is equal to size of witch image
        roi = img[witch_y1:witch_y2, witch_x1:witch_x2]

        #original image in background (bg) where witch is not
        roi_bg = cv2.bitwise_and(roi,roi,mask = mask)
        roi_fg = cv2.bitwise_and(witch,witch,mask=mask_inv)
        dst = cv2.add(roi_bg,roi_fg)

        #put back in original image
        img[witch_y1:witch_y2, witch_x1:witch_x2] = dst

        counter += 1
    # Display the resulting frame in browser
    # cv2.putText(img,'Face recognized: '+str(counter),(10,40), font, 1.5, (0, 255, 0), 1, cv2.LINE_AA)
    return cv2.imencode('.jpg', img)[1].tobytes()

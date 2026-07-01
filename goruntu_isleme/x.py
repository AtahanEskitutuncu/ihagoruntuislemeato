import cv2
import numpy as np 
import time 


def nothing(x):
    pass
#hsv'ye göre renkleri algıla
def detect_red_blue_hsv(h, s, v):
   
    if v < 50 or s < 50:
        return "Other"

    if h < 10 or h > 170:
        return "Red"

    if 90 <= h <= 130:
        return "Blue"

    return "Other"


previousTime = 0
currentTime  = 0

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)



altKirmizi1 = np.array([0, 120, 70])
ustKirmizi1 = np.array([10, 255, 255])
altKirmizi2 = np.array([170, 120, 70])
ustKirmizi2 = np.array([180, 255, 255])

altMavi = np.array([100, 80, 40])
ustMavi = np.array([140, 255, 220])


font = cv2.FONT_HERSHEY_SIMPLEX


prev_time = time.time()
fps = 0.0

previousTime = time.time()

kernel = np.ones((3,3), np.uint8)

while 1:
    ret,frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame,1)
    frame = cv2.blur(frame, (3, 3))

    hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)#bgr'den hsvye çevirme işi

    k1 = cv2.inRange(hsv, altKirmizi1, ustKirmizi1)
    k2 = cv2.inRange(hsv, altKirmizi2, ustKirmizi2)
    kirmiziMaske = cv2.bitwise_or(k1, k2)

    maviMaske = cv2.inRange(hsv, altMavi, ustMavi)

    kirmiziMaske = cv2.morphologyEx(kirmiziMaske, cv2.MORPH_OPEN, kernel, iterations=1)
    kirmiziMaske = cv2.morphologyEx(kirmiziMaske, cv2.MORPH_CLOSE, kernel, iterations=1)

    maviMaske = cv2.morphologyEx(maviMaske, cv2.MORPH_OPEN, kernel, iterations=1)
    maviMaske = cv2.morphologyEx(maviMaske, cv2.MORPH_CLOSE, kernel, iterations=1)

    mask = cv2.bitwise_or(kirmiziMaske, maviMaske)



    kirmiziContours,_ = cv2.findContours(kirmiziMaske, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    maviContours,_    = cv2.findContours(maviMaske,    cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


    
    for cnt in kirmiziContours:
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt,True)
        if perimeter == 0:
            continue

        circularity = 4 * np.pi * area / (perimeter * perimeter)
        if area <= 400:
            continue

        epsilon = 0.02 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        x, y = approx.ravel()[:2]


        color_name = "Red"

        shape_name = "Polygon"

#şekillerin tanımı , çemberselde tembellik yapmadım
        if circularity > 0.8:
            shape_name = "Circular"
        elif len(approx) == 3:
            shape_name = "Triangle"
        elif len(approx) == 4:
            shape_name = "Rectangle"
        elif len(approx) == 5:
            shape_name = "Pentagon"
        elif len(approx) == 6:
            shape_name = "Hexagon"
        
        cv2.drawContours(frame, [approx], 0, (0, 0, 0), 3)
        cv2.putText(frame, f"{color_name} {shape_name}",
                    (x, y), font, 1, (0, 0, 0), 2)


    for cnt in maviContours:
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt,True)
        if perimeter == 0:
            continue

        circularity = 4 * np.pi * area / (perimeter * perimeter)
        if area <= 400:
            continue

        epsilon = 0.02 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        x, y = approx.ravel()[:2]


        color_name = "Blue"

        shape_name = "Polygon"

#şekillerin tanımı , çemberselde tembellik yapmadım
        if circularity > 0.8:
            shape_name = "Circular"
        elif len(approx) == 3:
            shape_name = "Triangle"
        elif len(approx) == 4:
            shape_name = "Rectangle"
        elif len(approx) == 5:
            shape_name = "Pentagon"
        elif len(approx) == 6:
            shape_name = "Hexagon"
        
        cv2.drawContours(frame, [approx], 0, (0, 0, 0), 3)
        cv2.putText(frame, f"{color_name} {shape_name}",
                    (x, y), font, 1, (0, 0, 0), 2)




    cv2.putText(frame , str(int(fps)) , (10,70)  ,cv2.FONT_HERSHEY_PLAIN, 3 , (0,255,0) , thickness=2)


    
    current_time = time.time()
    dt = current_time - prev_time
    prev_time = current_time
    if dt > 0:
        fps = 1.0 / dt



        

    cv2.imshow("frame" ,frame)
    cv2.imshow("mask" , mask)
    cv2.imshow("kirmiziMaske", kirmiziMaske)
    cv2.imshow("maviMaske", maviMaske)

    if cv2.waitKey(3) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
import cv2
import HandTracking as ht

def colorBackgroundText(img, text, font, fontScale, textPos, textThickness=1,textColor=(0,255,0), bgColor=(0,0,0), pad_x=3, pad_y=3):
    (t_w, t_h), _= cv2.getTextSize(text, font, fontScale, textThickness) # getting the text size
    x, y = textPos
    cv2.rectangle(img, (x-pad_x, y+ pad_y), (x+t_w+pad_x, y-t_h-pad_y), bgColor,-1) # draw rectangle 
    cv2.putText(img,text, textPos,font, fontScale, textColor,textThickness ) # draw in text

    return img

def main():
    width = 640             # Width of Camera
    height = 480            # Height of Camera

    cap = cv2.VideoCapture(0)   # Getting video feed from the webcam
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)           # Adjusting size
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    detector = ht.handDetector(maxHands=1)                  # Detecting one hand at max
    while True:
        success, img = cap.read()
        img = detector.findHands(img)                       # Finding the hand
        lmlist, bbox = detector.findPosition(img)           # Getting position of hand
        
        if len(lmlist)!=0:
            fingers = detector.fingersUp()      # Checking if fingers are upwards
            colorBackgroundText(img, f'Thumb: {fingers[0]}', cv2.FONT_HERSHEY_COMPLEX, 0.5, (20, 40), 1, (0, 255, 0), (0, 0, 0), 8, 8)
            colorBackgroundText(img, f'First Finger: {fingers[1]}', cv2.FONT_HERSHEY_COMPLEX, 0.5, (20, 80), 1, (0, 255, 0), (0, 0, 0), 8, 8)
            colorBackgroundText(img, f'Second Finger: {fingers[2]}', cv2.FONT_HERSHEY_COMPLEX, 0.5, (20, 120), 1, (0, 255, 0), (0, 0, 0), 8, 8)
            colorBackgroundText(img, f'Third Finger: {fingers[3]}', cv2.FONT_HERSHEY_COMPLEX, 0.5, (20, 160), 1, (0, 255, 0), (0, 0, 0), 8, 8)
            colorBackgroundText(img, f'Fourth Finger: {fingers[4]}', cv2.FONT_HERSHEY_COMPLEX, 0.5, (20, 200), 1, (0, 255, 0), (0, 0, 0), 8, 8)
            
        cv2.imshow("Image", img)
        if cv2.waitKey(5) & 0xFF == 27:
            cv2.destroyAllWindows()
            cap.release()
            break

if __name__ == "__main__":
    main()

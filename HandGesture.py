import cv2
import imutils
import numpy as np
from sklearn.metrics import pairwise
from pynput.mouse import Button, Controller
import wx

bg = None
mouse = Controller()

app = wx.App(False)
(sx,sy) = wx.GetDisplaySize()

def count(thresholded, segmented):
    chull = cv2.convexHull(segmented)

    extreme_top    = tuple(chull[chull[:, :, 1].argmin()][0])
    extreme_bottom = tuple(chull[chull[:, :, 1].argmax()][0])
    extreme_left   = tuple(chull[chull[:, :, 0].argmin()][0])
    extreme_right  = tuple(chull[chull[:, :, 0].argmax()][0])

	# مرکز کف دست    
    cX = int((extreme_left[0] + extreme_right[0]) / 2)
    cY = int((extreme_top[1] + extreme_bottom[1]) / 2)
    
	# یافتن فاصله وسط کف دست و نقاط دیگر با استفاده از کانوکس هول   
    distance = pairwise.euclidean_distances([(cX, cY)], Y=[extreme_left, extreme_right, extreme_top, extreme_bottom])[0]
    maximum_distance = distance[distance.argmax()]
	
    radius = int(0.8 * maximum_distance)
	
    circumference = (2 * np.pi * radius)

	# استخراج یک دایره شامل محل کف دست و انگشت ها    
    circular_roi = np.zeros(thresholded.shape[:2], dtype="uint8")
    
    cv2.circle(circular_roi, (cX, cY), radius, 255, 1)
	
    circular_roi = cv2.bitwise_and(thresholded, thresholded, mask=circular_roi)

    (_, cnts, _) = cv2.findContours(circular_roi.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    count = 0
    
    for c in cnts:
        
        (x, y, w, h) = cv2.boundingRect(c)

		# افزایش تعداد انگشت با شرط زیر        
        if((cY + (cY * 0.25)) > (y + h)) and ((circumference * 0.25) > c.shape[0]):
            count+= 1

    return count



def run_avg(image, aWeight):
    global bg
    if bg is None:
        bg = image.copy().astype("float")
        return

    cv2.accumulateWeighted(image, bg, aWeight)

def segment(image, threshold=25):
    global bg
    diff = cv2.absdiff(bg.astype("uint8"), image)

    # با گرفتن ترشلد پس زمینه را مس یابد    
    thresholded = cv2.threshold(diff,
                                threshold,
                                255,
                                cv2.THRESH_BINARY)[1]

    (_, cnts, _) = cv2.findContours(thresholded.copy(),
                                    cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)

    if len(cnts) == 0:
        return
    else:
        segmented = max(cnts, key=cv2.contourArea)
        return (thresholded, segmented)
if __name__ == "__main__":
    # مقدار دهی اولیه به متوسط اجرا    
    aWeight = 0.5

    camera = cv2.VideoCapture(0)

    top, right, bottom, left = 0, 300, 350, 650
    num_frames = 0
    hamed = 0

    while(True):
        (grabbed, frame) = camera.read()

        frame = imutils.resize(frame, width=700)

        # فلیپ کردن تصویر برای جلوگیری از اثر آینه ای       
        frame = cv2.flip(frame, 1)
        clone = frame.copy()
        (height, width) = frame.shape[:2]

        roi = frame[top:bottom, right:left]

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)

        if num_frames < 30:
            run_avg(gray, aWeight)
        else:
            hand = segment(gray)

            # چک کردن پیدا شدن دست            
            if hand is not None:
                (thresholded, segmented) = hand

                cv2.drawContours(clone, [segmented + (right, top)], -1, (0, 0, 255))
                cX= segmented[0][0][0] + right
                cY= segmented[0][0][1] + 25
                
                # کشیدن خط دور دست، و یک پوینتر در نوک انگشت اشاره                
                cv2.circle(clone, (cX, cY), 7, (0, 0, 255), -1)
                cv2.putText(clone, "center", (cX - 20, cY - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                cv2.drawContours(clone, [segmented+ (right, top) ], -1, (0, 255, 0))
                cv2.imshow("Thesholded", thresholded)
                hamed = count(thresholded,segmented)
                print(hamed)
                # قسمت مربوط به موس(گرفتن یک آیکن و جابجا کردن آن و رها کردن آن)               
                if(hamed > 1):
                    mouse.release(Button.left)
                    mouse.position = (int((cX-right)*sx/(left-right)),int(cY*sy/(bottom-top)))
                if(hamed < 2):
                    mouse.position = (int((cX-right)*sx/(left-right)),int(cY*sy/(bottom-top)))
                    mouse.press(Button.left)
                    

        #محصور کردن قسمت مربوط با دست         
        cv2.rectangle(clone, (left, top), (right, bottom), (0,255,0), 2)

        num_frames += 1
        

        
        #نشان دادن قاب به همراه دست قطعه بندی شده        
        cv2.imshow("Video Feed", clone)

        keypress = cv2.waitKey(1) & 0xFF

        if keypress == ord("q"):
            break

camera.release()
cv2.destroyAllWindows()
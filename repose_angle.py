import cv2
import math

path = r'C:\Users\pedro\OneDrive\Documentos\PIBIC\repose_angle.jpg'
img = cv2.imread(path)
pointslist = []

def mousePoints(event,x ,y,flags,params):
    if event == cv2.EVENT_LBUTTONDOWN:
        size = len(pointslist)
        if size != 0 and size % 3 != 0:
            cv2.line(img,tuple(pointslist[round((size-1)/3)*3]),(x,y),(0,0,255),2)
        cv2.circle(img, (x,y),3,(0,0,255),cv2.FILLED)
        pointslist.append([x,y])


def gradient(pt1,pt2):
    return (pt2[1]-pt1[1])/(pt2[0]-pt1[0])
    
def getAngle(pointslist):
    pt1, pt2, pt3 = pointslist[-3:]
    m1 = gradient(pt1,pt2)
    m2 = gradient(pt1,pt3)
    angR = math.atan((m1-m2)/(1+(m2*m1)))
    angD = round(math.degrees(angR))
    cv2.putText(img,str(angD),(pt1[0]-40,pt1[1]-20),cv2.FONT_HERSHEY_COMPLEX, 1.5,(0,0,255),2)

while True:

    if len(pointslist) % 3 == 0 and len(pointslist) != 0:
        getAngle(pointslist)
    cv2.imshow('Image',img)
    cv2.setMouseCallback('Image',mousePoints)
    if cv2.waitKey(1) & 0xFF == ord('w'):
        pointslist = []
        img = cv2.imread(path)



import cv2
from ultralytics import YOLO
import numpy as np
#from AWSIoTPythonSDK.MQTTLib import AWSIoTMQTTClient
#import json

#_______reading parking coordinates from .txt file___________#

def cord(path):
  with open(path, 'r') as file:
    coordinates =eval( file.read())

  #___________________Parking coordinates____________________#

  cord_dict={}
  for i in range(len(coordinates)):
    cord_dict['slot'+str(i)]=np.array(coordinates[i],np.int32)
  return cord_dict

'''#___________________________________AWS IoT core SETUP______________________________________#

client = AWSIoTMQTTClient("MyClientID")
client.configureEndpoint("a4w2za7t2smbw-ats.iot.us-east-1.amazonaws.com", 8883)
client.configureCredentials(
    "AmazonRootCA1.pem",
    "private.pem.key",
    "certificate.pem.crt"
)

client.configureConnectDisconnectTimeout(15)  # 10 sec
client.configureMQTTOperationTimeout(10)  # 5 sec


print("Connecting to AWS IoT Core...")
if client.connect():
    print("Connected to AWS IoT Core!")
else:
    print("Failed to connect to AWS IoT Core.")


topic = "ParkEase" '''

def yolo_detection(frame):
  detected_car={}
  result=Model(frame)

  #________________________________detected car Coordinates And Class__________________________________________________#

  for re in result:
    box=re.boxes.xyxy
    classes=re.boxes.cls
    for i in range(len(re)):
      if classes[i]==2:

       #_______________________________bounding box__________________________________________________________#

        #cv2.rectangle(frame,(int(box[i][0]),int(box[i][1])),(int(box[i][2]),int(box[i][3])),(0,255,0),2)     
       

       #_________________________________Mid point____________________________________________________________#

        Xcenter=int((box[i][0]+box[i][2])//2)
        Ycenter=int((box[i][1]+box[i][3])//2)
        center=(Xcenter,Ycenter)
        cv2.circle(frame,center,1,(255,0,0),-1)


        for slot, coordinates in parking_coordinates.items():

          result = cv2.pointPolygonTest((coordinates), center, False)   #_____if center inside the polygon returns -1, outside returns +1, in between returns 0_______#

          if result < 0:
            colour=(0,0,255)
            thickness=1
            detected_car[slot]='Occupied'
          else:
            detected_car[slot]='Free'
            colour=(0,255,0)
            thickness=1
          cv2.polylines(frame, [coordinates], True, colour, thickness)
  cv2.imshow('frame',frame)
  return detected_car

'''#______________________________Publish data___________________________________________________________________#

def publush(data):
  payload = {data}
  client.publish(topic, json.dumps(payload), 1) '''

video1=cv2.VideoCapture('/content/drive/MyDrive/Project_v1/Video/leaving.mp4')       #<-------------path to video_______________#

Model=YOLO('yolov8n.pt')  #Model

path='/content/drive/MyDrive/Project_v1/Coordinates/Coordinates(Leaving).txt.txt'     #<------------Path to .txt file (Coordinates)_____________________#
parking_coordinates=cord(path)

while True:
  cap1, frame1=video1.read()
  if cap1==False:
    break

  frame1=cv2.resize(frame1,(640,640))
  occupancy=yolo_detection(frame1)
  print(occupancy)




  if cv2.waitKey(1) & 0xFF == ord('q'):
    break

video1.release()
cv2.destroyAllWindows()
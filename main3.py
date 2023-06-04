from flask import Flask, render_template, Response, request
import cv2
import datetime, time
import os, sys
import numpy as np
from threading import Thread
import tensorflow as tf
import pyttsx3
from ultralytics import YOLO
from ultralytics.yolo.v8.detect.predict import DetectionPredictor

# model = YOLO("C:/Users/Admin/Downloads/best_55.pt")
# model.predict(source="0", show=True, conf=0.15)


# net = cv2.dnn.readNet('F:/clgass/sem8/seminar/weapon-detection/yolov3.cfg', 'F:/clgass/sem8/seminar/weapon-detection/yolov3.weights')

# Load the pretrained model for object classification
model = tf.keras.models.load_model('F:\clgass\sem8\seminar\flask_env\saved_model\cnn-model1.h')

classes = ['Gun', 'NoWeapon']

layer_names = net.getLayerNames()
colors = np.random.uniform(0, 255, size=(len(classes), 3))

try:
    layer_names = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
except IndexError:
    layer_names = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]



global capture,rec_frame, grey, switch, neg, face, rec, out 
capture=0
grey=0
neg=0
face=0
switch=1
rec=0

#make shots directory to save pics
try:
    os.mkdir('./shots')
except OSError as error:
    pass

#Load pretrained face detection model    
# net = tf.keras.models.load_model('./saved_model/cnn-model1.h5')
# net = cv2.dnn.readNet('F:/clgass/sem8/seminar/weapon-detection/yolov3.cfg', 'F:/clgass/sem8/seminar/weapon-detection/yolov3.weights')
# net = cv2.dnn.readNetFromCaffe('./saved_model/deploy.prototxt.txt', './saved_model/res10_300x300_ssd_iter_140000.caffemodel')


#instatiate flask app  
app = Flask(__name__, template_folder='./templates')


camera = cv2.VideoCapture(0)

def record(out):
    global rec_frame
    while(rec):
        time.sleep(0.05)
        out.write(rec_frame)

def detect_face(frame):
    # pred = model.predict(frame)
    # print(pred)
    frame = model.predict(frame, show=True, conf=0.50)
    return frame

def detect_face_disable(frame):
    height, width, channels = frame.shape

    #detect objects
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

    net.setInput(blob)
    outs = net.forward(layer_names)

    #information on screen
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.6:
                #if object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                #box co-ordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                 # Extract the region of interest (ROI) corresponding to the detected object
                # roi = frame[y:y+h, x:x+w]
                try: 
                # Preprocess the ROI for classification using the pretrained model
                    if frame[y:y+h, x:x+w] is not None :
                        resized_roi = cv2.resize(frame[y:y+h, x:x+w], (224, 224))
                        normalized_roi = resized_roi.astype('float') / 255.0
                        expanded_roi = np.expand_dims(normalized_roi, axis=0)

                        # Use the pretrained model to classify the ROI and print the result
                        predictions = model.predict(expanded_roi)
                        class_index = np.argmax(predictions)
                        class_label = classes[class_index]
                        print("Detected object:", class_label)
                except Exception as e:
                    pass
    #             boxes.append([x, y, w, h])
    #             confidences.append(float(confidence))
    #             class_ids.append(class_id)
    # indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.5)
    # font = cv2.FONT_HERSHEY_PLAIN
    # for i in range(len(boxes)):
    #     if i in indices:
    #         x, y, w, h = boxes[i]
    #         label = f"{str(classes[class_ids[i]])}: {confidences[i]:.2f}"
    #         color = colors[class_ids[i]]
    #         cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    #         cv2.putText(frame, label, (x, y + 30), font, 3, color, 3)
    #         print("gun detected")
    #         #voice alarm
    #         engine = pyttsx3.init()
    #         voices = engine.getProperty('voices')
    #         engine.setProperty('voice', voices[1].id)
    #         engine.setProperty('rate', 150)
    #         engine.say("Weapon Detected")
    #         engine.runAndWait()

    # frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
    return frame
 

def gen_frames():  # generate frame by frame from camera
    global out, capture,rec_frame
    while True:
        success, frame = camera.read() 
        if success:
            if(face):                
                frame= detect_face(frame)
            if(grey):
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if(neg):
                frame=cv2.bitwise_not(frame)    
            if(capture):
                capture=0
                now = datetime.datetime.now()
                p = os.path.sep.join(['shots', "shot_{}.png".format(str(now).replace(":",''))])
                cv2.imwrite(p, frame)
            
            if(rec):
                rec_frame=frame
                frame= cv2.putText(cv2.flip(frame,1),"Recording...", (0,25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),4)
                frame=cv2.flip(frame,1)
            
                
            try:
                ret, buffer = cv2.imencode('.jpg', cv2.flip(frame,1))
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            except Exception as e:
                pass
                
        else:
            pass


@app.route('/')
def index():
    return render_template('index.html')
    
    
@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/requests',methods=['POST','GET'])
def tasks():
    global switch,camera
    if request.method == 'POST':
        if request.form.get('click') == 'Capture':
            global capture
            capture=1
        elif  request.form.get('grey') == 'Grey':
            global grey
            grey=not grey
        elif  request.form.get('neg') == 'Negative':
            global neg
            neg=not neg
        elif  request.form.get('face') == 'Detect':
            global face
            face=not face 
            if(face):
                time.sleep(4)   
        elif  request.form.get('stop') == 'Stop/Start':
            
            if(switch==1):
                switch=0
                camera.release()
                cv2.destroyAllWindows()
                
            else:
                camera = cv2.VideoCapture(0)
                switch=1
        elif  request.form.get('rec') == 'Start/Stop Recording':
            global rec, out
            rec= not rec
            if(rec):
                now=datetime.datetime.now() 
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                out = cv2.VideoWriter('vid_{}.avi'.format(str(now).replace(":",'')), fourcc, 20.0, (640, 480))
                #Start new thread for recording the video
                thread = Thread(target = record, args=[out,])
                thread.start()
            elif(rec==False):
                out.release()
                          
                 
    elif request.method=='GET':
        return render_template('index.html')
    return render_template('index.html')


if __name__ == '__main__':
    app.run()
    
camera.release()
cv2.destroyAllWindows()     

import cv2
import numpy as np
import pyttsx3

#load Yolo
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
with open("class.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

layer_names = net.getLayerNames()
colors = np.random.uniform(0, 255, size=(len(classes), 3))

try:
    layer_names = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
except IndexError:
    layer_names = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

def value():
    val = input("Enter file name or press enter to start webcam : \n")
    if val == "":
        val = 0
    return val

#capture video
cap = cv2.VideoCapture(value())

while True:
    _, img = cap.read()
    height, width, channels = img.shape

    #detect objects
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

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

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.5)
    font = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)):
        if i in indices:
            x, y, w, h = boxes[i]
            label = f"{str(classes[class_ids[i]])}: {confidences[i]:.2f}"
            color = colors[class_ids[i]]
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, label, (x, y + 30), font, 3, color, 3)
            print("gun detected")
            #voice alarm
            engine = pyttsx3.init()
            voices = engine.getProperty('voices')
            engine.setProperty('voice', voices[1].id)
            engine.setProperty('rate', 150)
            engine.say("Weapon Detected")
            engine.runAndWait()

    frame = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
    cv2.imshow("Security Feed", img)
    key = cv2.waitKey(5)
    if key == 9:
        break
cap.release()
cv2.destroyAllWindows()

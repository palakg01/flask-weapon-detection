# add preprocecssing

import cv2
import numpy as np
import pyttsx3
import tensorflow as tf
from tensorflow import keras


#load Yolo
net = cv2.dnn.readNet('F:/clgass/sem8/seminar/weapon-detection/yolov3.cfg', 'F:/clgass/sem8/seminar/weapon-detection/yolov3.weights')

# Load the pretrained model for object classification
model = tf.keras.models.load_model('F:/clgass/sem8/seminar/weapon-detection/VGG16/VGG_model_new.h5')

classes = ['Gun', 'NoWeapon']

layer_names = net.getLayerNames()
colors = np.random.uniform(0, 255, size=(len(classes), 3))

try:
    layer_names = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
except IndexError:
    layer_names = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

#capture video
cap = cv2.VideoCapture(0)

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

                 # Extract the region of interest (ROI) corresponding to the detected object
                roi = img[y:y+h, x:x+w]

                # Preprocess the ROI for classification using the pretrained model
                if roi is not None:
                    resized_roi = cv2.resize(roi, (224, 224))
                    normalized_roi = resized_roi.astype('float') / 255.0
                    expanded_roi = np.expand_dims(normalized_roi, axis=0)

                    # Use the pretrained model to classify the ROI and print the result
                    predictions = model.predict(expanded_roi)
                    class_index = np.argmax(predictions)
                    class_label = classes[class_index]
                    print("Detected object:", class_label)

                # boxes.append([x, y, w, h])
                # confidences.append(float(confidence))
                # class_ids.append(class_id)

    # indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.5)
    # font = cv2.FONT_HERSHEY_PLAIN
    # for i in range(len(boxes)):
    #     if i in indices:
    #         x, y, w, h = boxes[i]
    #         label = f"{str(classes[class_ids[i]])}: {confidences[i]:.2f}"
    #         color = colors[class_ids[i]]
    #         cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
    #         cv2.putText(img, label, (x, y + 30), font, 3, color, 3)
    #         print("gun detected")
    #         #voice alarm
    #         engine = pyttsx3.init()
    #         voices = engine.getProperty('voices')
    #         engine.setProperty('voice', voices[1].id)
    #         engine.setProperty('rate', 150)
    #         engine.say("Weapon Detected")
    #         engine.runAndWait()

    # frame = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
    # cv2.imshow("Security Feed", img)
    # key = cv2.waitKey(5)
    # if key == 9:
    #     break

    # Display the original frame with the detected objects and their labels
    cv2.imshow('Object detection', img)

    # Exit the program if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

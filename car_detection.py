import cv2
import numpy as np
import time
# import pytesseract

# physical_devices = tf.config.experimental.list_physical_devices("GPU")
# print('Num Gpus Available:', len(physical_devices))
# tf.config.experimental.set_memory_growth(physical_devices[0], True)


# we call yolov weights with cnn
file_path = "C:/documnetss/my_projects/space_parking_car/"
model = cv2.dnn.readNetFromDarknet(file_path+"yolov4-custom.cfg",
                                   file_path+"yolov4-custom_last.weights")
model.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
model.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
layers = model.getLayerNames()
unconnect = model.getUnconnectedOutLayers()
unconnect = unconnect - 1

output_layers = []
for i in unconnect:
    output_layers.append(layers[i])

classfile = file_path + "classes.names"
classNames = []
with open(classfile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

def detection(frame):
    H, W, channel = frame.shape

    frame_blob = cv2.dnn.blobFromImage(frame, 1/255, (416, 416), swapRB=True)
    model.setInput(frame_blob)
    detection_layers = model.forward(output_layers)

    ids_list = []
    confidence_list = []
    boxes_list = []
    for detection_layer in detection_layers:
        for object_detection in detection_layer:
            scores = object_detection[5:]
            predicted_id = np.argmax(scores)
            confidence = scores[predicted_id]

            if confidence > 0.80:
                label = classNames[predicted_id]
                bounding_box = object_detection[:4]*np.array([W, H, W, H])
                (box_center_x, box_center_y, box_width, box_height) = bounding_box.astype('int')
                startX = int(box_center_x - box_width/2)
                startY = int(box_center_y - box_height/2)

                ids_list.append(predicted_id)
                confidence_list.append(confidence)
                boxes_list.append([startX, startY, box_width, box_height])
    max_ids = cv2.dnn.NMSBoxes(boxes_list, confidence_list, 0.5, 0.4)
    for max_id in max_ids:
        box = boxes_list[max_id]

        x0, y0 = box[0], box[1]
        x1, y1 = x0+box[2], y0+box[3]

        finding_id = ids_list[max_id]
        label = classNames[finding_id]

        accurate = int(confidence_list[max_id]*100)
        cv2.rectangle(frame, (x0, y0), (x1, y1), (0,0,255), 2)
        cv2.putText(frame, str(accurate), (x0, y0-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    return frame

video = cv2.VideoCapture('C:/documnetss/my_projects/space_parking_car/parking.mp4')
prev_time = 0
new_time = 0
while video.isOpened():
    ret, frame = video.read()
    # frame=cv2.flip(frame,0)
    h, w = frame.shape[:2]
    # frame = cv2.resize(frame, (int(w/2), int(h/2)))
    processed_frame = detection((frame))
    new_time = time.time()
    fps = int(1/(new_time - prev_time))
    cv2.putText(processed_frame, str(fps), (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    prev_time = new_time
    cv2.imshow('frame', processed_frame)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    elif key == ord('w'):
        cv2.waitKey(0)
cv2.destroyAllWindows()
video.release()
#
#

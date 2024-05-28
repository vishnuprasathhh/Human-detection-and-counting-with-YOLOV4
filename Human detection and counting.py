import numpy as np
import cv2
import os
import imutils
import argparse

NMS_THRESHOLD = 0.3
MIN_CONFIDENCE = 0.2

def pedestrian_detection(image, model, layer_name, personidz=0):
    (H, W) = image.shape[:2]
    results = []

    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
        swapRB=True, crop=False)
    model.setInput(blob)
    layerOutputs = model.forward(layer_name)

    boxes = []
    centroids = []
    confidences = []

    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            if classID == personidz and confidence > MIN_CONFIDENCE:
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                boxes.append([x, y, int(width), int(height)])
                centroids.append((centerX, centerY))
                confidences.append(float(confidence))

    idzs = np.array(cv2.dnn.NMSBoxes(boxes, confidences, MIN_CONFIDENCE, NMS_THRESHOLD)).flatten()


    count = 0  # Initialize count
    for i in idzs.flatten():
        (x, y) = (boxes[i][0], boxes[i][1])
        (w, h) = (boxes[i][2], boxes[i][3])

        # Update count and draw rectangle with person number
        count += 1
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Display person number above the box in red color
        cv2.putText(image, f'Person {count}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Display status and total count
    status_text = "Status: Detection"
    count_text = f"Count: {count}"
    cv2.putText(image, status_text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.putText(image, count_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return results

def detect_by_input(input_type, input_path, model, layer_name):
    if input_type == "image":
        image = cv2.imread(input_path)
        image = imutils.resize(image, width=700)
        pedestrian_detection(image, model, layer_name, personidz=LABELS.index("person"))
        cv2.imshow("Image Detection", image)
        cv2.waitKey(0)
    elif input_type == "video":
        cap = cv2.VideoCapture(input_path)
        while True:
            (grabbed, image) = cap.read()
            if not grabbed:
                break
            image = imutils.resize(image, width=700)
            pedestrian_detection(image, model, layer_name, personidz=LABELS.index("person"))
            cv2.imshow("Video Detection", image)
            key = cv2.waitKey(1)
            if key == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
    elif input_type == "webcam":
        cap = cv2.VideoCapture(0)
        while True:
            (grabbed, image) = cap.read()
            if not grabbed:
                break
            image = imutils.resize(image, width=700)
            pedestrian_detection(image, model, layer_name, personidz=LABELS.index("person"))
            cv2.imshow("Webcam Detection", image)
            key = cv2.waitKey(1)
            if key == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    labelsPath = "coco.names"
    LABELS = open(labelsPath).read().strip().split("\n")

    weights_path = "yolov4-tiny.weights"
    config_path = "yolov4-tiny.cfg"

    model = cv2.dnn.readNetFromDarknet(config_path, weights_path)

    layer_name = model.getLayerNames()
    layer_name = [layer_name[i - 1] for i in model.getUnconnectedOutLayers()]

    parser = argparse.ArgumentParser(description='Pedestrian Detection')
    parser.add_argument('--input_type', choices=['image', 'video', 'webcam'], default='image', help='Type of input: image, video, webcam')
    parser.add_argument('--input_path', type=str, default='path/to/your/input', help='Path to input file (image, video) or camera index (webcam)')
    args = parser.parse_args()

    detect_by_input(args.input_type, args.input_path, model, layer_name)

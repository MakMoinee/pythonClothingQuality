import cv2
import torch
import sys
from datetime import datetime

username = ''
password = ''
if len(sys.argv) < 3:
    print("Please provide the IP address as a command-line argument.")
    sys.exit(1)

ip = sys.argv[1]
userID = sys.argv[2]
rtsp_url = f"rtsp://{ip}/live/ch00_0"
acceptable_confidence = 0.52


# Load YOLOv5 model
def load_model(weights_path):
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights_path)
    return model

def updateStatus(user_id,ip,status):
    err ={}
    # if len(device_docs) == 0:
    #     err = {'error': 'Device not exist'}
    #     print({'error': 'Device not exist'})

    # for doc in device_docs:
    #     doc.reference.update({'status': status})
        
    return err

def save_activity_history(image_name):
    print("Saving activity history ...")
    activityHistory = {}
    activityHistory['caregiverID'] = userID
    activityHistory['ip'] = ip
    activityHistory['imagePath'] = f"./gallery/{image_name}"
    activityHistory['createdAt'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    activityHistory['status'] = "Unread"
    print("Successfully saved activity history")
    

def save_image_with_boxes(frame, detections):
    detected_objects = []
    for index, detection in detections.iterrows():
        if detection['confidence'] >= acceptable_confidence:
            box = [
                int(detection['xmin']),
                int(detection['ymin']),
                int(detection['xmax']),
                int(detection['ymax'])
            ]
            # Draw bounding box on the frame
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)
            cv2.putText(frame, f"{detection['name']} {detection['confidence']:.2f}",
                        (box[0], box[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            detected_objects.append({
                'name': detection['name'],
                'confidence': detection['confidence'],
                'bbox': box
            })

    if detected_objects:
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        image_name = f"detected_{timestamp}.jpg"
        cv2.imwrite(f"./gallery/{image_name}", frame)
        return image_name, detected_objects

    return None, None

# Initialize the model
model_path = "./best.pt"  # Replace this with the path to your custom YOLOv5 .pt file
model = load_model(model_path)
updateOnce = False
# Set the model to evaluation mode
model.eval()

stream = cv2.VideoCapture(rtsp_url)
detectedCount = 0

try:
    while stream.isOpened():
        ret, frame = stream.read()
        if not ret:
            break

        if (updateOnce is not True):
            updateOnce = True
            print("Updating Device Status")
            err = updateStatus(userID, ip, 'Active')
            if 'error' in err and err['error'] != "":
                stream.release()
                cv2.destroyAllWindows()
                sys.exit()
        
        # Perform inference
        results = model(frame)
        detections = results.pandas().xyxy[0]
        
        for index, detection in detections.iterrows():
            if (detection['confidence'] >= acceptable_confidence):
                print(f"Confidence: {detection['confidence']}, Name: {detection['name']}")
                if "fall" in detection['name']:
                    detectedCount += 1
                if (detectedCount == 100):
                    print("Reached the desired detected count")
                    im,s = save_image_with_boxes(frame,detections)
                    detectedCount = 0
                    save_activity_history(im)
        
        cv2.imshow('Real-time Detection', results.render()[0])

        if cv2.waitKey(1) == ord('q'):
            break

except cv2.error as e:
    print(f"OpenCV error: {e}")
except KeyboardInterrupt:
    print("Keyboard Interrupt detected. Exiting...")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
finally:
    # Release resources
    updateStatus(userID,ip,"Inactive")
    stream.release()
    cv2.destroyAllWindows()


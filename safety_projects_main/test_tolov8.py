import cv2
import threading
from ultralytics import YOLO
import serial
import time

class CameraCapture:
    def __init__(self, video_path):
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise Exception("Error: Could not open video file.")
        self.frame = None
        self.running = True
    
    def get_frame(self):
        return self.frame
    
    def set_frame(self, frame):
        self.frame = frame
    
    def capture_frames(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                self.running = False
                break
            self.set_frame(frame)
    
    def release(self):
        self.cap.release()
        cv2.destroyAllWindows()

class ObjectDetector:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.class_list = self.model.model.names
    
    def predict(self, frame):
        return self.model.predict(frame)
    
    def draw_box(self, img, result):
        xyxy = result.boxes.xyxy.numpy()
        confidence = result.boxes.conf.numpy()
        class_id = result.boxes.cls.numpy().astype(int)
        class_name = [self.class_list[x] for x in class_id]
        sum_output = list(zip(class_name, confidence, xyxy))
        out_image = img.copy()
        for run_output in sum_output:
            label, con, box = run_output
            box_color = (0, 0, 255)
            text_color = (255, 255, 255)
            first_half_box = (int(box[0]), int(box[1]))
            second_half_box = (int(box[2]), int(box[3]))
            cv2.rectangle(out_image, first_half_box, second_half_box, box_color, 2)
            text_print = '{label} {con:.2f}'.format(label=label, con=con)
            text_location = (int(box[0]), int(box[1] - 10))
            labelSize, baseLine = cv2.getTextSize(text_print, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
            cv2.rectangle(out_image,
                          (int(box[0]), int(box[1] - labelSize[1] - 10)),
                          (int(box[0]) + labelSize[0], int(box[1] + baseLine - 10)),
                          box_color, cv2.FILLED)
            cv2.putText(out_image, text_print, text_location,
                        cv2.FONT_HERSHEY_SIMPLEX, 1,
                        text_color, 2, cv2.LINE_AA)
        return out_image

def resize_image(img, scale_percent):
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    return resized

def calculate_custom_areas(img):
    height, width = img.shape[:2]
    
    # Define the areas
    area1_width = int(width * 0.4)
    area1_height = int(height * 0.4)
    area1_x1 = width // 4
    area1_y1 = height // 4
    area1_x2 = area1_x1 + area1_width
    area1_y2 = area1_y1 + area1_height

    area2_width = int(width * 0.4)
    area2_height = int(height * 0.4)
    area2_x1 = width // 2
    area2_y1 = height // 2
    area2_x2 = area2_x1 + area2_width
    area2_y2 = area2_y1 + area2_height
    
    return (area1_x1+300, area1_y1+200, area1_x2-300, area1_y2+120), (area2_x1+120, area2_y1-70, area2_x2-485, area2_y2-150)

def is_in_custom_area(box, custom_area):
    x1, y1, x2, y2 = box
    ca_x1, ca_y1, ca_x2, ca_y2 = custom_area
    return not (x2 < ca_x1 or x1 > ca_x2 or y2 < ca_y1 or y1 > ca_y2)

def alert_message(message):
    print(message)
    
# Function to save the image
def save_image(frame, filename):
    cv2.imwrite(filename, frame)
    print(f"Image saved as {filename}")
    
def draw_custom_area_CPB(img, custom_area):
    ca_x1, ca_y1, ca_x2, ca_y2 = custom_area
    color = (255, 0, 0)  # Blue color for the rectangle
    thickness = 2
    cv2.rectangle(img, (ca_x1, ca_y1), (ca_x2, ca_y2), color, thickness)
    return img

def draw_custom_area_SULP(img, custom_area):
    ca_x1, ca_y1, ca_x2, ca_y2 = custom_area
    color = (255, 255, 255)  # White color for the rectangle
    thickness = 2
    cv2.rectangle(img, (ca_x1, ca_y1), (ca_x2, ca_y2), color, thickness)
    return img

# Paths
video_path = r"C:\Users\Chanatip.S\Web\RecordFiles\2024-08-15\192.168.1.11_01_20240815162937973.mp4"
model_path = r"D:\Users\Chanatip.S\Documents\WORK\safety projects\safety_projects_main\model_version_1\weights\model_version_1.pt"

# Initialize objects
camera = CameraCapture(video_path)
detector = ObjectDetector(model_path)
scale_show = 100

# Initialize serial communication
# serial_port = 'COM3'  # Replace with your ESP32 serial port
# baud_rate = 115200
# ser = serial.Serial(serial_port, baud_rate, timeout=1)

# Dictionary to keep track of alert conditions and their start times
alert_times = {}

def process_frames():
    fps = camera.cap.get(cv2.CAP_PROP_FPS)
    delay = int(1000 / fps)

    while camera.running:
        start_time = time.time()
        frame = camera.get_frame()
        if frame is None:
            continue
        
        # Calculate the custom areas for the current frame
        custom_area1, custom_area2 = calculate_custom_areas(frame)
        
        results = detector.predict(frame)
        labeled_img = detector.draw_box(frame, results[0])
        
        # Draw the custom areas on the image
        display_img = draw_custom_area_CPB(labeled_img, custom_area1)
        display_img = draw_custom_area_SULP(display_img, custom_area2)
        
        # Resize the image for display
        display_img = cv2.resize(display_img, (1280,640))
        
        # Check for objects in the custom areas and trigger alerts
        xyxy = results[0].boxes.xyxy.numpy()
        class_id = results[0].boxes.cls.numpy().astype(int)
        class_name = [detector.class_list[x] for x in class_id]
        
        alert_triggered = False
        
        for box, name in zip(xyxy, class_name):
            current_time = time.time()
            
            if name == 'SULPHURIC':
                if is_in_custom_area(box, custom_area1):
                    if 'SULPHURIC' not in alert_times:
                        alert_times['SULPHURIC'] = current_time
                    if current_time - alert_times['SULPHURIC'] > 2:
                        save_image(display_img, "alert_image_SULPHURIC.jpg")
                        alert_times.pop('SULPHURIC')
                        
                    # Add bold text overlay for misposition at the top right
                    text = "Misposition - Red Lamp"
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 1
                    font_thickness = 2
                    text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
                    text_x = display_img.shape[1] - text_size[0] - 50
                    text_y = 50

                    # Draw the text multiple times to create a bold effect
                    cv2.putText(display_img, text, (text_x, text_y), font, font_scale, (0, 0, 255), font_thickness + 1, cv2.LINE_AA)
                    cv2.putText(display_img, text, (text_x, text_y), font, font_scale, (0, 0, 255), font_thickness + 2, cv2.LINE_AA)
                    
                    alert_message(f"Alert! SULPHURIC detected in custom area: {box}")
                    try:
                        # ser.write(b'1')
                        print("Signal sent to ESP32 successfully.")
                    except serial.SerialException as e:
                        print(f"Error sending signal to ESP32: {e}")
                    alert_triggered = True
            
            elif name == 'CPB-75D':
                if is_in_custom_area(box, custom_area2):
                    if 'CPB-75D' not in alert_times:
                        alert_times['CPB-75D'] = current_time
                    if current_time - alert_times['CPB-75D'] > 2:
                        save_image(display_img, "alert_image_CPB-75D.jpg")
                        alert_times.pop('CPB-75D')
                        
                    # Add bold text overlay for misposition at the top right
                    text = "Misposition - Red Lamp"
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 1
                    font_thickness = 2
                    text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
                    text_x = display_img.shape[1] - text_size[0] - 50
                    text_y = 100

                    # Draw the text multiple times to create a bold effect
                    cv2.putText(display_img, text, (text_x, text_y), font, font_scale, (0, 0, 255), font_thickness + 1, cv2.LINE_AA)
                    cv2.putText(display_img, text, (text_x, text_y), font, font_scale, (0, 0, 255), font_thickness + 2, cv2.LINE_AA)
                    
                    alert_message(f"Alert! CPB-75D detected in custom area: {box}")
                    try:
                        # ser.write(b'1')
                        print("Signal sent to ESP32 successfully.")
                    except serial.SerialException as e:
                        print(f"Error sending signal to ESP32: {e}")
                    alert_triggered = True

        # Reset alert times if no alert is triggered
        if not alert_triggered:
            alert_times.clear()
            try:
                # Add bold text for green lamp condition at the top right
                text = "Good position - Green Lamp"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 1
                font_thickness = 2
                text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
                text_x = display_img.shape[1] - text_size[0] - 50
                text_y = 100

                # Draw the text multiple times to create a bold effect
                cv2.putText(display_img, text, (text_x, text_y), font, font_scale, (0, 255, 0), font_thickness + 1, cv2.LINE_AA)
                cv2.putText(display_img, text, (text_x, text_y), font, font_scale, (0, 255, 0), font_thickness + 2, cv2.LINE_AA)
                
            except serial.SerialException as e:
                print(f"Error sending signal to ESP32: {e}")

        
        # Calculate elapsed time and delay to match the frame rate
        elapsed_time = time.time() - start_time
        if elapsed_time < delay / 1000:
            time.sleep((delay / 1000) - elapsed_time)

        # Exit the video display if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            camera.running = False
            break
        # Display the frame in a window
        cv2.imshow('Video', display_img)
# Start the capture thread
capture_thread = threading.Thread(target=camera.capture_frames)
capture_thread.start()

# Start the processing thread
process_thread = threading.Thread(target=process_frames)
process_thread.start()

# Wait for threads to complete
capture_thread.join()
process_thread.join()

# Release resources
camera.release()
# ser.close()


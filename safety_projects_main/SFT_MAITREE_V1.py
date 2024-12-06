import cv2
import threading
import time
import os
import logging
from ultralytics import YOLO
from pymodbus.client import ModbusTcpClient

# Setup logging configuration
logging.basicConfig(filename='program_log.log', 
                    level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
                    
class CameraCapture:
    def __init__(self, video_path):
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            # logging.error("Error: Could not open video file.")
            raise Exception("Error: Could not open video file.")
        self.frame = None
        self.running = True
        self.frame_rate = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_delay = 1 / self.frame_rate

    def get_frame(self):
        return self.frame
    
    def set_frame(self, frame):
        self.frame = frame
    
    def capture_frames(self):
        while self.running:
            start_time = time.time()
            ret, frame = self.cap.read()
            if not ret:
                self.running = False
                logging.info("End of video file reached.")
                break
            self.set_frame(frame)
            elapsed_time = time.time() - start_time
            if elapsed_time < self.frame_delay:
                time.sleep(self.frame_delay - elapsed_time)
    
    def release(self):
        self.cap.release()
        cv2.destroyAllWindows()

class ObjectDetector:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.class_list = self.model.model.names
    
    def predict(self, frame):
        return self.model.predict(frame, task='detect',imgsz=640, conf=0.75)
    
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
            # CPB-75D Area                                            # SULPHURIC Area
    return (area1_x1+300, area1_y1+200, area1_x2-305, area1_y2+70), (area2_x1+120, area2_y1-70, area2_x2-485, area2_y2-205)

def calculate_intersection_area(box, custom_area):
    # Unpack box and custom area coordinates
    x1, y1, x2, y2 = box
    ca_x1, ca_y1, ca_x2, ca_y2 = custom_area

    # Calculate intersection coordinates
    inter_x1 = max(x1, ca_x1)
    inter_y1 = max(y1, ca_y1)
    inter_x2 = min(x2, ca_x2)
    inter_y2 = min(y2, ca_y2)

    # Calculate intersection area
    inter_width = max(0, inter_x2 - inter_x1)
    inter_height = max(0, inter_y2 - inter_y1)
    inter_area = inter_width * inter_height

    # Calculate bounding box area
    box_area = (x2 - x1) * (y2 - y1)

    # Calculate percentage of intersection
    if box_area > 0:
        intersection_percentage = (inter_area / box_area) * 100
    else:
        intersection_percentage = 0

    return intersection_percentage

def is_in_custom_area(box, custom_area):
    x1, y1, x2, y2 = box
    ca_x1, ca_y1, ca_x2, ca_y2 = custom_area
    return not (x2 < ca_x1 or x1 > ca_x2 or y2 < ca_y1 or y1 > ca_y2)

def alert_message(display_img, message):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_thickness = 1
    text_y = 50

    text_x = display_img.shape[1] - 50  # Position text 50 pixels from the right edge

    if message == 'ALARM':
        text = "Misposition - Red Lamp"
        text_color = (0, 0, 255)
        alert_triggered = 'ALARM'
        
    elif message == 'GOOD':
        text = "Match position - Green Lamp"
        text_color = (0, 255, 0)
        alert_triggered = 'GOOD'
        
    else:
        return display_img, None  # No message to display
    
    text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
    text_x -= text_size[0]  # Adjust x position to ensure text is within image bounds
    
    cv2.putText(display_img, text, (text_x, text_y), font, font_scale, text_color, font_thickness, cv2.LINE_AA)
    
    return display_img, alert_triggered

    
def save_image(frame, filename):
    cv2.imwrite(filename, frame)
    # logging.info(f"Image saved as {filename}")
    
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

def read_coil_status(last_sent_status):
    while True:
        coil_status_result = client.read_coils(0, 1)
        if coil_status_result.bits[0]:  # If coil is True
            if last_sent_status == '0':
                try:
                    client.write_coil(16, False)  # Send signal 0 (reset)
                    print("Signal 0 sent (coil reset)")
                    last_sent_status = '0'  # Reset last sent status to '0'
                except:
                    print('Write 0 fail')
            else:
                print('Please wait for reset')  # Already in a reset state
            break
        time.sleep(1)  # Avoid busy-waiting by adding a delay

def check_last_sent_status(last_status):
    # Here we return the last status, this can be extended with more logic if needed
    return last_status

def process_frames():
    alert_times = {}
    last_sent_status = '0'  # Initialize last_sent_status as '0'
    
    try:
        while camera.running:
            frame = camera.get_frame()
            if frame is None:
                continue

            CPB_AREA, SULP_AREA = calculate_custom_areas(frame)
            results = detector.predict(frame)
            labeled_img = detector.draw_box(frame, results[0])

            display_img = draw_custom_area_CPB(labeled_img, CPB_AREA)
            display_img = draw_custom_area_SULP(display_img, SULP_AREA)
            display_img = cv2.resize(display_img, (1280, 640))

            xyxy = results[0].boxes.xyxy.numpy()
            class_id = results[0].boxes.cls.numpy().astype(int)
            class_name = [detector.class_list[x] for x in class_id]

            status = "GOOD"
            for box, name in zip(xyxy, class_name):
                current_time = time.time()

                if is_in_custom_area(box, CPB_AREA) and (name == 'SULPHURIC' or name == 'NaOH'):
                    if calculate_intersection_area(box, CPB_AREA) > 30:
                        status = "ALARM"
                        if 'SULPHURIC' not in alert_times:
                            alert_times['SULPHURIC'] = current_time
                        if current_time - alert_times['SULPHURIC'] > 2:
                            save_image(display_img, f"Alarm_CPB-75D_Area.jpg")
                            alert_times.pop('SULPHURIC')
                            if last_sent_status != "1":
                                logging.info("1 Alarm !! SULPHURIC in area CPB-75D")
                                last_sent_status = "1"
                                try:
                                    client.write_coil(16, True)  # Send signal 1
                                    print("Signal 1 sent (alarm)")
                                    threading.Thread(target=read_coil_status, args=(last_sent_status,)).start()
                                except:
                                    print('Write 1 fail')
                                
                elif is_in_custom_area(box, SULP_AREA) and (name == 'CPB-75D' or name == 'NaOH'):
                    if calculate_intersection_area(box, SULP_AREA) > 30:
                        status = "ALARM"
                        if 'CPB-75D' not in alert_times:
                            alert_times['CPB-75D'] = current_time
                        if current_time - alert_times['CPB-75D'] > 2:
                            save_image(display_img, f"Alarm_SULPHURIC_Area.jpg")
                            alert_times.pop('CPB-75D')
                            if last_sent_status != "1":
                                logging.info("1 Alarm !! CPB-75D in area SULPHURIC")
                                last_sent_status = "1"
                                try:
                                    client.write_coil(16, True)  # Send signal 1
                                    print("Signal 1 sent (alarm)")
                                    threading.Thread(target=read_coil_status, args=(last_sent_status,)).start()
                                except:
                                    print('Write 1 fail')

            if status == "GOOD":
                alert_times.clear()
                if last_sent_status != "0":
                    logging.info("0 Normal")
                    last_sent_status = "0"
                    threading.Thread(target=read_coil_status, args=(last_sent_status,)).start()

            display_img, alert_triggered = alert_message(display_img, status)

            cv2.imshow('Detections', display_img)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                if camera:
                    camera.release()
                logging.info("Program ended")
                break
    except Exception as e:
        print(e)
        pass


if __name__ == "__main__":
    try:
        # Establish Modbus TCP connection
        client = ModbusTcpClient('192.168.110.102')  # Replace with your device IP
        connection = client.connect()
        if not connection:
            logging.error("Failed to connect to Modbus server.")
            raise Exception("Connection error: Could not connect to Modbus server.")

        # Initialize camera and detector
        video_path =  r"rtsp://admin:PTE3402C@192.168.110.101:554" # Replace with the path to your video
        model_path = "/home/smartfactory/projects/safety_projects/SAFETY_MAITREE_1.1_multi/model/model_v3_1_1/weights/model_v3_1_1.pt"  # Replace with the path to your YOLOv8 model

        logging.info(f"Program started SW: MAITREE 1.0 Model: {os.path.basename(model_path)} MC: M-05-01 RMCH")

        camera = CameraCapture(video_path)
        detector = ObjectDetector(model_path)

        # Start capturing frames in a separate thread
        capture_thread = threading.Thread(target=camera.capture_frames)
        capture_thread.start()

        # Start processing frames
        process_frames()

    except Exception as e:
        logging.error(f"An error occurred: {e}")
        print(f"An error occurred: {e}")
    
    finally:
        # Close Modbus client connection
        if client:
            client.close()
        logging.info("Modbus connection closed.")

        # Release camera resources if the camera is still running
        if camera:
            camera.release()

        logging.info("Program terminated.")


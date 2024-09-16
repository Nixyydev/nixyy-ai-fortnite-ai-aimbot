import ctypes
import cv2
import json
import math
import os
import sys
import time
import torch
import win32api
import pygame
import dxcam
from PyQt5.QtWidgets import QWidget, QApplication
from PyQt5.QtGui import QPainter, QColor, QPen
from PyQt5.QtCore import Qt
from termcolor import colored


#CONFIGURATION

# Pull values from config.json
file = open("lib/config/config.json")
config = json.load(file)

detection_box_size = config['detection_box_size']
detection_box_size_threshold = 150

mouse_delay = config['mouse_delay']
pixel_increment = config['pixel_increment']
aim_height = config["aim_height"]

#AI confidence
confidence_threshold = config['confidence_threshold']
iou_threshold = config['iou_threshold']

#input
keybind = config['keybind'] #0x02 is right click. See: https://learn.microsoft.com/en-us/windows/win32/inputdev/virtual-key-codes for other keycodes
use_controller = config['use_controller']

#visual
crosshair_thickness = config['crosshair_thickness']
crosshair_color = config['crosshair_color']
crosshair_size = config['crosshair_size']
detection_box_opacity = config['detection_box_opacity']

#CONFIGURATION^^


screensize = {'X': ctypes.windll.user32.GetSystemMetrics(0), 'Y': ctypes.windll.user32.GetSystemMetrics(1)}
screen_res_X = screensize['X'] # Horizontal
screen_res_Y = screensize['Y'] # Vertical

print(f"Screen Resolution: {screen_res_X, screen_res_Y}")

screen_x = int(screen_res_X / 2)
screen_y = int(screen_res_Y / 2)

# Array for controller trigger pos values
values = []

#Initialize controller boolean
controller_is_targeted = False

# Initialize crosshair_dist
crosshair_dist = 1


PUL = ctypes.POINTER(ctypes.c_ulong)
class KeyBdInput(ctypes.Structure):
    _fields_ = [("wVk", ctypes.c_ushort),
                ("wScan", ctypes.c_ushort),
                ("dwFlags", ctypes.c_ulong),
                ("time", ctypes.c_ulong),
                ("dwExtraInfo", PUL)]

class HardwareInput(ctypes.Structure):
    _fields_ = [("uMsg", ctypes.c_ulong),
                ("wParamL", ctypes.c_short),
                ("wParamH", ctypes.c_ushort)]

class MouseInput(ctypes.Structure):
    _fields_ = [("dx", ctypes.c_long),
                ("dy", ctypes.c_long),
                ("mouseData", ctypes.c_ulong),
                ("dwFlags", ctypes.c_ulong),
                ("time", ctypes.c_ulong),
                ("dwExtraInfo", PUL)]

class Input_I(ctypes.Union):
    _fields_ = [("ki", KeyBdInput),
                ("mi", MouseInput),
                ("hi", HardwareInput)]

class Input(ctypes.Structure):
    _fields_ = [("type", ctypes.c_ulong),
                ("ii", Input_I)]

class POINT(ctypes.Structure):
    _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]

# Detection area visuals

class DetectionBox(QWidget):
    def __init__(self):
        super().__init__()
        # Set window properties
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setAttribute(Qt.WA_TranslucentBackground)

        self.box_size = detection_box_size

        # Set initial window size
        self.setGeometry(int(screen_res_X - self.box_size) // 2, int(screen_res_Y - self.box_size) // 2,
                         self.box_size, self.box_size)

        self.crosshair_color = QColor(crosshair_color[0], crosshair_color[1], crosshair_color[2])
        self.crosshair_thickness = crosshair_thickness

    def setBoxSize(self, new_size):
        self.box_size = new_size
        self.setGeometry(int(screen_res_X - self.box_size) // 2, int(screen_res_Y - self.box_size) // 2,
                         self.box_size, self.box_size)
        self.update()  # Update the widget to reflect the new size

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        pen = QPen(self.crosshair_color, self.crosshair_thickness, Qt.SolidLine)
        painter.setPen(pen)

        # Draw Detection Box
        center_x = self.width() // 2
        center_y = self.height() // 2
        half_size = self.box_size // 2
        painter.drawRect(center_x - half_size, center_y - half_size, 2 * half_size, 2 * half_size)

        # Draw Crosshair
        painter.setOpacity(detection_box_opacity)
        painter.drawLine(center_x, center_y + crosshair_size, center_x, center_y - crosshair_size)
        painter.drawLine(center_x - crosshair_size, center_y, center_x + crosshair_size, center_y)

app = QApplication([])
overlay = DetectionBox()
overlay.show()

class Aimbot:
    extra = ctypes.c_ulong(0)
    ii_ = Input_I()
    
    with open("lib/config/sens.json") as f:
        sens_config = json.load(f)
    aimbot_status = colored("ENABLED", 'green')

    def __init__(self, box_constant = detection_box_size, collect_data = False, mouse_delay = mouse_delay):
        #controls the initial centered box width and height of the "Lunar Vision" window
        self.box_constant = box_constant #controls the size of the detection box (equaling the width and height)
        self.detection_box_size = detection_box_size
        print("[INFO] Loading the neural network model")

        #self.model = torch.hub.load('ultralytics/yolov5', 'custom', 'best v5.pt', force_reload=True) #uncomment for YoloV5
        self.model = torch.hub.load('yolov7', 'custom', 'lib/best.pt', force_reload=True, source='local')

        os.system('cls') # Clear console

        print(colored('''
 _     _    _ _   _          _____   __      _____  
| |   | |  | | \ | |   /\   |  __ \  \ \    / /__ \ 
| |   | |  | |  \| |  /  \  | |__) |  \ \  / /   ) |
| |   | |  | | . ` | / /\ \ |  _  /    \ \/ /   / / 
| |___| |__| | |\  |/ ____ \| | \ \     \  /   / /_ 
|______\____/|_| \_/_/    \_\_|  \_\     \/   |____|

(Neural Network Aimbot)''', "green"))

        if torch.cuda.is_available():
            print(colored("CUDA ACCELERATION [ENABLED]", "green"))
        else:
            print(colored("[!] CUDA ACCELERATION IS UNAVAILABLE", "red"))
            print(colored("[!] Check your PyTorch installation, else performance will be poor", "red"))

        self.model.conf = confidence_threshold # base confidence threshold (or base detection (0-1)
        self.model.iou = iou_threshold # NMS IoU (0-1)
        self.collect_data = collect_data
        self.mouse_delay = mouse_delay

        print("\n[INFO] PRESS 'F1' TO TOGGLE AIMBOT\n[INFO] PRESS 'F2' TO QUIT")


    def update_status_aimbot():
        if Aimbot.aimbot_status == colored("ENABLED", 'green'):
            Aimbot.aimbot_status = colored("DISABLED", 'red')
        else:
            Aimbot.aimbot_status = colored("ENABLED", 'green')
        sys.stdout.write("\033[K")
        print(f"[!] AIMBOT IS [{Aimbot.aimbot_status}]", end = "\r")

    def sleep(duration, get_now = time.perf_counter):
        if duration == 0: return
        now = get_now()
        end = now + duration
        while now < end:
            now = get_now()

    def is_aimbot_enabled():
        return True if Aimbot.aimbot_status == colored("ENABLED", 'green') else False

    def is_targeted():
        # If holding right click return true and skip controller check
        if win32api.GetKeyState(int(keybind, 16)) in (-127, -128):
            return True

        if use_controller != "false":
            global controller_is_targeted
            global values
            pygame.init()
            pygame.joystick.init()
            # Check for connected controllers
            if pygame.joystick.get_count() == 0:
                print("No controllers detected.")
                return
            
            # Initialize controller
            controller = pygame.joystick.Joystick(0)
            controller.init()

            for event in pygame.event.get():
                if event == b"<Event(1543-Unknown {})>":
                    return
                if event.type == pygame.JOYAXISMOTION and event.axis == 4:
                    values.append(event.value)
                    if len(values) < 2:
                        values.append(event.value + 0.1)

                    if values[-2:][0] < values[-1:][0]: # Get last 2 values to determine trigger status
                        controller_is_targeted = True
                    else:
                        controller_is_targeted = False

            if len(values) > 2:
                values.pop() # Remove items from array after use

            return controller_is_targeted
        else:
            return False

    def move_crosshair(self, x, y):
        if not Aimbot.is_targeted(): return

        scale = Aimbot.sens_config["targeting_scale"]

        for rel_x, rel_y in Aimbot.interpolate_coordinates_from_center((x, y), scale):
            Aimbot.ii_.mi = MouseInput(rel_x, rel_y, 0, 0x0001, 0, ctypes.pointer(Aimbot.extra))
            input_obj = Input(ctypes.c_ulong(0), Aimbot.ii_)
            ctypes.windll.user32.SendInput(1, ctypes.byref(input_obj), ctypes.sizeof(input_obj))
            Aimbot.sleep(self.mouse_delay)

    #generator yields pixel tuples for relative movement
    
    def interpolate_coordinates_from_center(absolute_coordinates, scale):

        #if dist == 0:
        #    dist == 1

        #percent = float((0.3 * dist))

        #if percent == 0:
        #    percent = pixel_increment

        #pixel_increment = percent / 100

        #print(f"dist: {dist} percent: {percent}")
        
        diff_x = (absolute_coordinates[0] - screen_x) * scale/pixel_increment
        diff_y = (absolute_coordinates[1] - screen_y) * scale/pixel_increment
        length = int(math.dist((0,0), (diff_x, diff_y)))

        if length == 0: return
        unit_x = (diff_x/length) * pixel_increment
        unit_y = (diff_y/length) * pixel_increment
        x = y = sum_x = sum_y = 0
        for k in range(0, length):
            sum_x += x
            sum_y += y
            x, y = round(unit_x * k - sum_x), round(unit_y * k - sum_y)
            yield x, y
   

    def start(self):
        print('[!] For best results in Fortnite, use performance mode')
        print('[INFO] Beginning screen capture')
        Aimbot.update_status_aimbot()

        #global detection_box_size_threshold

        left, top = (screen_res_X - detection_box_size) // 2, (screen_res_Y - detection_box_size) // 2
        right, bottom = left + detection_box_size, top + detection_box_size
        region = (left, top, right, bottom)

        camera = dxcam.create(output_color="BGR", region=region)
        #camera.start(video_mode=True, target_fps=120)
        #print(dxcam.device_info())
        
        while True:
            #start_time = time.perf_counter()
            frame = camera.grab()

            if frame is not None:
                results = self.model(frame)

                if len(results.xyxy[0]) != 0: #player detected
                    least_crosshair_dist = closest_detection = player_in_frame = False

                    for *box, conf, cls in results.xyxy[0]: #iterate over each player detected
                        x1y1 = [int(x.item()) for x in box[:2]]
                        x2y2 = [int(x.item()) for x in box[2:]]
                        x1, y1, x2, y2, conf = *x1y1, *x2y2, conf.item()
                        height = y2 - y1
                        relative_head_X, relative_head_Y = int((x1 + x2)/2), int((y1 + y2)/2 - height/aim_height) #offset to roughly approximate the head using a ratio of the height
                        own_player = x1 < 15 or (x1 < self.box_constant/5 and y2 > self.box_constant/1.2) #helps ensure that your own player is not regarded as a valid detection

                        #calculate the distance between each detection and the crosshair at (self.box_constant/2, self.box_constant/2)
                        crosshair_dist = math.dist((relative_head_X, relative_head_Y), (self.box_constant/2, self.box_constant/2))

                        if not least_crosshair_dist: least_crosshair_dist = crosshair_dist #initalize least crosshair distance variable first iteration

                        if crosshair_dist <= least_crosshair_dist and not own_player:
                            least_crosshair_dist = crosshair_dist
                            closest_detection = {"x1y1": x1y1, "x2y2": x2y2, "relative_head_X": relative_head_X, "relative_head_Y": relative_head_Y, "conf": conf}


                        #self.detection_box_size = (y2 - x1) * 2
                        #if self.detection_box_size < detection_box_size_threshold:
                        #    self.detection_box_size = detection_box_size_threshold
                        #overlay.setBoxSize(self.detection_box_size)
                        
                        #print(f"Detection box: {Aimbot.detection_box_size}")

                        if own_player:
                            own_player = False
                            if not player_in_frame:
                                player_in_frame = True

                    if closest_detection: #if valid detection exists
                        #mss
                        #absolute_head_X, absolute_head_Y = closest_detection["relative_head_X"] + detection_box['left'], closest_detection["relative_head_Y"] + detection_box['top']

                        #dxcam
                        absolute_head_X, absolute_head_Y = closest_detection["relative_head_X"] + left, closest_detection["relative_head_Y"] + top

                        crosshair_dist = math.dist((relative_head_X, relative_head_Y), (self.box_constant/2, self.box_constant/2))

                        x1, y1 = closest_detection["x1y1"]

                        if Aimbot.is_aimbot_enabled() and self.detection_box_size >= crosshair_dist:
                            #print(Aimbot.detection_box_size, crosshair_dist)
                            Aimbot.move_crosshair(self, absolute_head_X, absolute_head_Y)

                #cv2.putText(frame, f"FPS: {int(1/(time.perf_counter() - start_time))}", (5, 30), cv2.FONT_HERSHEY_DUPLEX, 1, (113, 116, 244), 2)
                #cv2.imshow("Lunar Vision", frame)
                
                if not Aimbot.is_aimbot_enabled:
                    print('Stopping DXcam..')
                    camera.stop()
                
                if cv2.waitKey(1) & 0xFF == ord('0'):
                    break
                
    def clean_up():
        print("\n[INFO] F2 WAS PRESSED. QUITTING...")
        os._exit(0)

if __name__ == "__main__": print("You are in the wrong directory and are running the wrong file; you must run lunar.py")
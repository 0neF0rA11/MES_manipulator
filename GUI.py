#TODO: Скеллинг интерфейса


import cv2
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import ttk, colorchooser
from PIL import Image, ImageTk
from ultralytics import YOLO
from server import Server
import platform
from calibration import CalibrationWindow
import os


class Application(tk.Tk):
    def __init__(self):
        super().__init__()
        self.set_calib = None
        self.min_area = 250

        self.image_size = (800, 560)
        self.video_cv2_running = False
        self.video_neuro_running = False
        self.video_arUco_running = False
        self.video_paused = False
        self.is_camera_on = False
        self.cap = None

        self.kernelOpen = np.ones((5, 5))
        self.kernelClose = np.ones((20, 20))

        self.color_dict = {
            'blue': (105, 219, 129),
            'orange': (11, 252, 176),
            'yellow': (19, 255, 169),
            'green': (77, 225, 77)
        }

        self.lowerBound = np.array([0, 0, 0], dtype=np.uint8)
        self.upperBound = np.array([179, 255, 255], dtype=np.uint8)
        self.font = cv2.FONT_HERSHEY_SIMPLEX

        self.objects_coord = []
        self.send_list = []

        self.title("Машинное зрение")

        self.tk.call('tk', 'scaling', 1.5)
        current_os = platform.system()

        if current_os == "Linux":
            self.attributes('-fullscreen', True)
        elif current_os == "Windows":
            self.state('zoomed')

        connection_frame = ttk.Frame(self, padding="20")
        connection_frame.grid(row=0, column=0, sticky=(tk.W, tk.E))
        connect_status_label = ttk.Label(connection_frame, text="Not connected", font=("Arial", 14, "bold"))
        connect_status_label.grid(column=0, row=0, sticky=tk.W)
        self.ser = Server()
        self.model = YOLO('yolov8s.pt')
        self.classes = self.model.names

        self.config()

        self.exposure = 0
        self.white_balance = 0
        self.color_components = 0

        self.sent_data_label = ttk.Label(self, text="Sent: None", font=("Arial", 14, "bold"))
        self.sent_data_label.place(relx=.05, rely=.85, anchor="sw")

        self.received_data_label = ttk.Label(self, text="Received: None", font=("Arial", 14, "bold"))
        self.received_data_label.place(relx=.05, rely=.9, anchor="sw")

        # self.draw_axis()
        self.create_port_widgets()
        self.create_camera_widgets()
        self.create_composition_settings()

    def config(self):
        if not os.path.exists("config.txt"):
            self.k_x = 83 / 70
            self.k_y = 83 / 70
            self.mm_to_pixel_x = int(250 * self.k_x)
            self.mm_to_pixel_y = int(250 * self.k_x)
            self.x_0, self.y_0 = self.mm_to_pixel_x, self.mm_to_pixel_y
            self.x_max = self.x_0 + self.mm_to_pixel_x
            self.y_max = self.y_0 + self.mm_to_pixel_y
            self.h = 70
            self.len_f_x, self.len_f_y = 500, 500
        else:
            with open("config.txt", "r") as file:
                config_data = {}
                for line in file:
                    key, value = line.strip().split()
                    config_data[key] = value
                self.k_x = float(config_data['k_x'])
                self.k_y = float(config_data['k_y'])
                self.len_f_x, self.len_f_y = int(config_data['len_f_x']), int(config_data['len_f_y'])
                self.x_0, self.y_0 = int(self.k_x * self.len_f_x // 2), int(self.k_y * self.len_f_y // 2)
                self.x_max = self.x_0 * 2
                self.y_max = self.y_0 * 2
                self.h = int(config_data['h'])

    def create_port_widgets(self):
        connection_frame = ttk.Frame(self, padding="20")
        connection_frame.grid(row=0, column=0, sticky=(tk.W, tk.E))

        ttk.Label(connection_frame, text="Port:").grid(column=0, row=0, sticky=tk.W)
        port_combobox = ttk.Combobox(connection_frame, values=self.ser.ports)
        port_combobox.grid(column=1, row=0)

        ttk.Label(connection_frame, text="Baud Rate:").grid(column=0, row=1, sticky=tk.W)
        baudrate_combobox = ttk.Combobox(connection_frame, values=[115200], state="readonly")
        baudrate_combobox.grid(column=1, row=1)
        baudrate_combobox.current(0)

        self.create_port_button(connection_frame, port_combobox, baudrate_combobox)

    def create_port_button(self, frame, port_box, baudrate_box):
        connect_status_label = ttk.Label(frame, text="Not connected", font=("Arial", 14, "bold"))
        connect_status_label.grid(column=3, row=0, sticky=tk.W)

        ttk.Button(frame,
                   text="Open Port",
                   command=lambda: self.ser.connect_port(port_box.get(),
                                                         baudrate_box.get(),
                                                         connect_status_label,
                                                         self,
                                                         self.received_data_label)
                   ).grid(column=2, row=0)

        ttk.Button(frame,
                   text="Close Port",
                   command=lambda: self.ser.close_port(connect_status_label)
                   ).grid(column=2, row=1)

        ttk.Button(frame, text="Завершить программу", command=self.destroy).grid(column=0, row=3)

        ttk.Button(frame,
                   text="Калибровка",
                   command=self.open_new_window
                   ).grid(column=2, row=3)

    def open_new_window(self):
        self.set_calib = CalibrationWindow(self, self.image_size, self.available_cameras)

    def response_to_request(self, received_list):
        if received_list[0].lower() not in self.color_dict:
            self.ser.send_command("Not cube\n", self.sent_data_label)
            return

        color, number = received_list

        tol_h = 2
        tol_s = 50
        tol_v = 50

        h_val, s_val, v_val = self.color_dict[color.lower()]

        self.lowerBound = np.array([
            max(h_val - tol_h, 0),
            max(s_val - tol_s, 0),
            max(v_val - tol_v, 0)
        ], dtype=np.uint8)
        self.upperBound = np.array([
            min(h_val + tol_h, 179),
            min(s_val + tol_s, 255),
            min(v_val + tol_v, 255)
        ], dtype=np.uint8)

        if self.camera_selection.get() != "Камеры не найдены":
            camera_index = int(self.camera_selection.get().split()[-1])
            if not self.cap:
                self.cap = cv2.VideoCapture(camera_index)

            ret, frame = self.cap.read()
            if ret:
                frame = cv2.resize(frame, self.image_size)
                imgHSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

                mask = cv2.inRange(imgHSV, self.lowerBound, self.upperBound)

                maskOpen = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernelOpen)
                maskClose = cv2.morphologyEx(maskOpen, cv2.MORPH_CLOSE, self.kernelClose)

                maskFinal = maskClose
                conts, _ = cv2.findContours(maskFinal.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

                self.objects_coord = []
                for i, contour in enumerate(conts):
                    x, y, w, h = cv2.boundingRect(contour)
                    center_x, center_y = x + w // 2, self.image_size[1] - (y + h) + h // 2
                    center_x_cam, center_y_cam = center_x - self.image_size[0] // 2, center_y - self.image_size[
                        1] // 2
                    center_x_coord, center_y_coord = center_x_cam + self.x_0, center_y_cam + self.y_0
                    if w * h > self.min_area and 0 <= center_x_coord <= self.x_max and 0 <= center_y_coord <= self.y_max:
                        self.objects_coord.append(
                            (int(center_x_coord * 1 / self.k_x),
                             int(center_y_coord * 1 / self.k_y)))
                self.send_coords(int(number))
            self.cap.release()
            self.cap = None

    def draw_axis(self):
        axis_frame = ttk.Frame(self)
        axis_frame.place(relx=0.43, rely=0.65, anchor='n')

        axis_canvas = tk.Canvas(axis_frame, width=130, height=110)
        axis_canvas.pack()

        padding = 25
        origin_x = padding
        origin_y = 90
        axis_length = 70

        axis_canvas.create_text(origin_x - 10, origin_y + 10, text="0", fill='black',
                                font=('Arial', 12, 'bold'))
        axis_canvas.create_line(origin_x, origin_y, origin_x, origin_y - axis_length, arrow=tk.LAST, fill='blue',
                                width=2)
        axis_canvas.create_text(origin_x - 15, origin_y - axis_length, text="Y", fill='blue',
                                font=('Arial', 12, 'bold'))
        axis_canvas.create_text(origin_x + 28, origin_y - axis_length, text=f"{self.len_f_y}, мм", fill='blue',
                                font=('Arial', 8, 'bold'))

        axis_canvas.create_line(origin_x, origin_y, origin_x + axis_length, origin_y, arrow=tk.LAST, fill='red',
                                width=2)
        axis_canvas.create_text(origin_x + axis_length, origin_y + 15, text="X", fill='red',
                                font=('Arial', 12, 'bold'))
        axis_canvas.create_text(origin_x + axis_length, origin_y - 15, text=f"f{self.len_f_x}, мм", fill='red',
                                font=('Arial', 8, 'bold'))

    def create_camera_widgets(self):
        camera_frame = ttk.Frame(self, padding='10')
        camera_frame.place(relx=0.7, rely=0.02, anchor='n')

        placeholder_img = Image.open("placeholder.jpg")
        placeholder_img = placeholder_img.resize(self.image_size)
        self.placeholder_photo = ImageTk.PhotoImage(placeholder_img)

        self.video_label = ttk.Label(self, image=self.placeholder_photo)
        self.video_label.place(relx=0.7, rely=0.02, anchor='n')
        self.video_label.bind("<Button-1>", self.pick_color)

        control_frame = ttk.Frame(self, padding='10')
        control_frame.place(relx=0.7, rely=0.75, anchor='n')
        # control_frame.place(relx=0.7, rely=0.77, anchor='n')
        # control_frame.lower()

        ttk.Label(control_frame, text="Выбор камеры:").grid(column=0, row=0, sticky=tk.W)

        self.camera_selection = ttk.Combobox(control_frame, values=self.detect_cameras(), state="readonly", width=10)
        self.camera_selection.grid(column=1, row=0)
        if self.camera_selection['values']:  # Если камеры найдены, установить первую камеру по умолчанию
            self.camera_selection.current(0)
        else:
            self.camera_selection.set("Камеры не найдены")

        self.option_var = tk.StringVar()
        self.combobox = ttk.Combobox(control_frame, textvariable=self.option_var, state='readonly')
        self.combobox['values'] = ("Манипулятор", "Нейросеть", "Распознавание ArUco")
        self.combobox.grid(column=2, row=0, padx=5)
        self.combobox.current(0)

        self.aruco_var = tk.StringVar()
        self.aruco_combobox = ttk.Combobox(control_frame, textvariable=self.aruco_var, state='readonly', width=4)
        self.aruco_combobox['values'] = ("4x4", "5x5", "6x6", "7x7")
        self.aruco_combobox.grid(column=7, row=0, padx=5)
        self.aruco_combobox.current(0)

        ttk.Button(control_frame, text="Запустить", command=self.start).grid(column=3, row=0, padx=5)
        self.pause_button = ttk.Button(control_frame, text="Пауза", command=self.pause_camera)
        self.pause_button.grid(column=4, row=0, padx=5)
        ttk.Button(control_frame, text="Остановить", command=self.stop_camera).grid(column=5, row=0, padx=5)

        self.class_selection = tk.StringVar()
        self.class_selection.set("All")
        self.class_labels = {1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane', 6: 'bs', 7: 'train',
                             8: 'trck', 9: 'boat', 10: 'traffic light', 11: 'fire hydrant', 12: 'stop sign',
                             13: 'parking meter', 14: 'bench', 15: 'bird', 16: 'cat', 17: 'dog', 18: 'horse',
                             19: 'sheep', 20: 'cow', 21: 'elephant', 22: 'bear', 23: 'zebra', 24: 'giraffe',
                             25: 'backpack', 26: 'mbrella', 27: 'handbag', 28: 'tie', 29: 'sitcase', 30: 'frisbee',
                             31: 'skis', 32: 'snowboard', 33: 'sports ball', 34: 'kite', 35: 'baseball bat',
                             36: 'baseball glove', 37: 'skateboard', 38: 'srfboard', 39: 'tennis racket', 40: 'bottle',
                             41: 'wine glass', 42: 'cp', 43: 'fork', 44: 'knife', 45: 'spoon', 46: 'bowl', 47: 'banana',
                             48: 'apple', 49: 'sandwich', 50: 'orange', 51: 'broccoli', 52: 'carrot', 53: 'hot dog',
                             54: 'pizza', 55: 'dont', 56: 'cake', 57: 'chair', 58: 'coch', 59: 'potted plant',
                             60: 'bed', 61: 'dining table', 62: 'toilet', 63: 'tv', 64: 'laptop', 65: 'mose',
                             66: 'remote', 67: 'keyboard', 68: 'cell phone', 69: 'microwave', 70: 'oven', 71: 'toaster',
                             72: 'sink', 73: 'refrigerator', 74: 'book', 75: 'clock', 76: 'vase', 77: 'scissors',
                             78: 'teddy bear', 79: 'hair drier', 80: 'toothbrsh'
                             }

        self.class_selection_entry = tk.OptionMenu(control_frame, self.class_selection, "All",
                                                   *self.class_labels.values())
        self.class_selection_entry.grid(column=6, row=0, padx=5)

    def create_composition_settings(self):
        settings_frame = ttk.Frame(self, padding='10')
        settings_frame.place(relx=0.2, rely=0.35, anchor='n')

        ttk.Label(settings_frame, text="Настройки композиции", font=("Arial", 14, "bold")).grid(column=0, row=0,
                                                                                                pady=10, columnspan=2)

        color_button = ttk.Button(settings_frame, text="Выбрать цвет", command=self.choose_color)
        color_button.grid(column=0, row=1, pady=5, columnspan=2)

        self.create_slider(settings_frame, "Настройка экспозиции", -180, 180, self.update_exposure, row=2)
        self.create_slider(settings_frame, "Баланс белого", -180, 180, self.update_white_balance, row=3)
        self.create_slider(settings_frame, "Цветоразностные составляющие", -180, 180, self.update_color_components,
                           row=4)

        send_frame = ttk.Frame(self, padding='10')
        send_frame.place(relx=0.2, rely=0.6, anchor='n')

        ttk.Button(send_frame, text="Отправить координаты", command=self.send_coords).grid(column=0,
                                                                                           row=0,
                                                                                           pady=10,
                                                                                           columnspan=2
                                                                                           )
        ttk.Button(send_frame, text="Обновить список", command=self.update_objects_list).grid(column=3,
                                                                                              row=0,
                                                                                              pady=10,
                                                                                              columnspan=2
                                                                                              )

    def update_objects_list(self):
        self.send_list = []

    def send_coords(self, number=1):
        if len(self.objects_coord) > 0 and 1 <= number <= len(self.objects_coord):
            object = sorted(self.objects_coord, key=lambda point: point[0] ** 2 + point[1] ** 2)[number-1]
            self.send_list.append(object)
            self.ser.send_command(
                f"G00 X {object[0]} Y {object[1]} Z {self.h}\n",
                self.sent_data_label
            )
        else:
            self.ser.send_command("Not cube\n", self.sent_data_label)

    def create_slider(self, frame, text, from_, to_, command, row, default=0):
        label = ttk.Label(frame, text=f"{text}")
        label.grid(column=0, row=row, pady=5, sticky=tk.W)

        slider = ttk.Scale(frame, from_=from_, to=to_, orient="horizontal", command=command)
        slider.set(default)
        slider.grid(column=1, row=row, pady=5)

    def update_exposure(self, value):
        self.exposure = int(float(value))

    def update_white_balance(self, value):
        self.white_balance = int(float(value))

    def update_color_components(self, value):
        self.color_components = int(float(value))

    def check_for_send(self, x, y):
        distance_list = [(x - p1) ** 2 + (y - p2) ** 2 for p1, p2 in self.send_list]
        distance_list = sorted(distance_list)
        if len(distance_list) != 0 and distance_list[0] < 20:
            return True
        else:
            return False

    def detect_cameras(self):
        index = 0
        self.available_cameras = []
        while True:
            cap = cv2.VideoCapture(index)
            if not cap.isOpened():
                break
            ret, _ = cap.read()
            if ret:
                self.available_cameras.append(f"Камера {index}")
            cap.release()
            index += 1
        return self.available_cameras

    def start(self):
        if self.camera_selection.get() != "Камеры не найдены":
            self.stop_camera()
            camera_index = int(self.camera_selection.get().split()[-1])
            self.cap = cv2.VideoCapture(camera_index)
            if not self.video_cv2_running and self.option_var.get() == "Манипулятор":
                self.video_cv2_running = True
                if self.set_calib and self.set_calib.flag:
                    self.config()
                self.show_frame()
            elif not self.video_neuro_running and self.option_var.get() == "Нейросеть":
                self.video_neuro_running = True
                self.show_neuro_frame()
            elif not self.video_arUco_running and self.option_var.get() == "Распознавание ArUco":
                self.video_arUco_running = True
                self.show_arUco_frame()

    def pick_color(self, event):
        if self.cap and self.video_cv2_running:
            self.update_objects_list()
            x, y = event.x, event.y
            image_width, image_height = self.image_size

            if 0 <= x < image_width and 0 <= y < image_height:
                frame_x = int((x / image_width) * self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                frame_y = int((y / image_height) * self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

                _, frame = self.cap.read()
                if frame is not None:
                    frame = self.apply_settings(frame)
                    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                    hsv_value = hsv_frame[frame_y, frame_x]

                    h_val = int(hsv_value[0])
                    s_val = int(hsv_value[1])
                    v_val = int(hsv_value[2])

                    tol_h = 2
                    tol_s = 50
                    tol_v = 50

                    self.lowerBound = np.array([
                        max(h_val - tol_h, 0),
                        max(s_val - tol_s, 0),
                        max(v_val - tol_v, 0)
                    ], dtype=np.uint8)
                    self.upperBound = np.array([
                        min(h_val + tol_h, 179),
                        min(s_val + tol_s, 255),
                        min(v_val + tol_v, 255)
                    ], dtype=np.uint8)

    def show_arUco_frame(self):
        if self.cap and self.video_arUco_running:
            if not self.video_paused:
                ret, frame = self.cap.read()
                if ret:
                    frame = self.apply_settings(frame)
                    frame = cv2.resize(frame, self.image_size)
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                    selected_dict = self.aruco_combobox.get()

                    if selected_dict == "4x4":
                        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
                    elif selected_dict == "5x5":
                        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)
                    elif selected_dict == "6x6":
                        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_50)
                    else:
                        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_7X7_50)

                    aruco_params = cv2.aruco.DetectorParameters()

                    corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=aruco_params)

                    if len(corners) > 0:
                        frame = cv2.aruco.drawDetectedMarkers(frame, corners, ids)

                    img = Image.fromarray(frame)
                    imgtk = ImageTk.PhotoImage(image=img)
                    self.video_label.imgtk = imgtk
                    self.video_label.configure(image=imgtk)

            self.video_label.after(10, self.show_arUco_frame)

    def show_neuro_frame(self):
        if self.cap and self.video_neuro_running:
            if not self.video_paused:
                ret, frame = self.cap.read()
                if ret:
                    frame = self.apply_settings(frame)
                    frame = cv2.resize(frame, self.image_size)
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    results = self.model.predict(frame, stream_buffer=True, verbose=False)

                    selected_class = self.class_selection.get()

                    a = results[0].boxes.data
                    px = pd.DataFrame(a).astype("float")

                    for index, row in px.iterrows():
                        confidence = row[4]

                        x1 = int(row[0])
                        y1 = int(row[1])
                        x2 = int(row[2])
                        y2 = int(row[3])
                        d = int(row[5])
                        c = self.classes[d]
                        if selected_class == "All" or c == selected_class:
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
                            cv2.putText(frame, f'{c} {confidence:.2f}', (x1, y1 - 25), self.font,
                                        0.5, (255, 255, 255), 2)

                    img = Image.fromarray(frame)
                    imgtk = ImageTk.PhotoImage(image=img)
                    self.video_label.imgtk = imgtk
                    self.video_label.configure(image=imgtk)
            self.video_label.after(10, self.show_neuro_frame)

    def show_frame(self):
        if self.cap and self.video_cv2_running:
            if not self.video_paused:
                ret, frame = self.cap.read()
                if ret:
                    frame = self.apply_settings(frame)
                    frame = cv2.resize(frame, self.image_size)
                    imgHSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

                    mask = cv2.inRange(imgHSV, self.lowerBound, self.upperBound)

                    maskOpen = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernelOpen)
                    maskClose = cv2.morphologyEx(maskOpen, cv2.MORPH_CLOSE, self.kernelClose)

                    maskFinal = maskClose
                    conts, _ = cv2.findContours(maskFinal.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    # cv2.drawContours(frame, conts, -1, (255, 0, 0), 3)
                    self.objects_coord = []
                    for i, contour in enumerate(conts):
                        x, y, w, h = cv2.boundingRect(contour)
                        center_x, center_y = x + w // 2, self.image_size[1] - (y + h) + h // 2
                        center_x_cam, center_y_cam = center_x - self.image_size[0] // 2, center_y - self.image_size[
                            1] // 2
                        center_x_coord, center_y_coord = center_x_cam + self.x_0, center_y_cam + self.y_0
                        if w * h > self.min_area and 0 <= center_x_coord <= self.x_max and 0 <= center_y_coord <= self.y_max:
                            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                            cv2.putText(frame, str(i + 1), (x, y - 10), self.font, 0.5, (0, 255, 255), 2)
                            if self.check_for_send(int(center_x_coord * 1 / self.k_x), int(center_y_coord * 1 / self.k_y)):
                                cv2.line(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                                cv2.line(frame, (x + w, y), (x, y + h), (0, 0, 255), 2)
                            else:
                                self.objects_coord.append(
                                    (int(center_x_coord * 1 / self.k_x),
                                     int(center_y_coord * 1 / self.k_y)))

                    center_x = self.image_size[0] // 2
                    center_y = self.image_size[1] // 2

                    line_length = 20
                    cv2.line(frame, (center_x - line_length // 2, center_y), (center_x + line_length // 2, center_y),
                             (255, 0, 0), 1)
                    cv2.line(frame, (center_x, center_y - line_length // 2), (center_x, center_y + line_length // 2),
                             (255, 0, 0), 1)

                    # Draw axis
                    image_width, image_height = self.image_size
                    padding = 35
                    axis_length = min(image_width, image_height) // 6

                    origin_x = padding
                    origin_y = image_height - padding
                    cv2.arrowedLine(
                        frame,
                        (origin_x, origin_y),  # Начало вектора
                        (origin_x, origin_y - axis_length),  # Конец вектора
                        (255, 0, 0),  # Цвет (синий)
                        3,  # Толщина линии
                        tipLength=0.15  # Длина наконечника стрелки
                    )
                    cv2.putText(frame, "Y", (origin_x - 15, origin_y - axis_length), self.font, 0.5,
                                (255, 0, 0), 2)
                    cv2.putText(frame, f"{self.len_f_y}, mm", (origin_x + 10, origin_y - axis_length),
                                self.font, 0.43, (255, 0, 0), 1)

                    cv2.arrowedLine(
                        frame,
                        (origin_x, origin_y),
                        (origin_x + axis_length, origin_y),
                        (0, 0, 255),
                        3,
                        tipLength=0.15
                    )
                    cv2.putText(frame, "X", (origin_x + axis_length - 5, origin_y + 20), self.font, 0.5,
                                (0, 0, 255), 2)
                    cv2.putText(frame, f"{self.len_f_x}, mm", (origin_x + axis_length - 30, origin_y - 20), self.font,
                                0.43, (0, 0, 255), 1)
                    # cv2.putText(frame, "0", (origin_x - 20, origin_y + 10), self.font, 0.5, (0, 255, 0), 1)

                    img = Image.fromarray(frame)
                    imgtk = ImageTk.PhotoImage(image=img)
                    self.video_label.imgtk = imgtk
                    self.video_label.configure(image=imgtk)

            self.video_label.after(10, self.show_frame)

    def apply_settings(self, frame):
        frame = cv2.convertScaleAbs(frame, alpha=1, beta=self.exposure)  # alpha отвечает за контраст, beta за яркость

        # Баланс белого (коррекция LAB)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(frame)
        l = cv2.add(l, self.white_balance)  # Управление уровнем освещения
        frame = cv2.merge([l, a, b])
        frame = cv2.cvtColor(frame, cv2.COLOR_LAB2BGR)

        # Цветоразностные составляющие (регулировка насыщенности)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        s = cv2.add(s, self.color_components)
        hsv = cv2.merge([h, s, v])
        frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        return frame

    def pause_camera(self):
        if self.video_cv2_running or self.video_neuro_running:
            self.video_paused = not self.video_paused
            if self.video_paused:
                self.pause_button.configure(text="Продолжить")
            else:
                self.pause_button.configure(text="Пауза")

    def stop_camera(self):
        if self.video_cv2_running or self.video_neuro_running or self.video_arUco_running:
            self.video_cv2_running = False
            self.video_neuro_running = False
            self.video_arUco_running = False
            self.video_paused = False
            if self.cap:
                self.cap.release()
                self.cap = None
            self.video_label.configure(image=self.placeholder_photo)
            self.pause_button.configure(text="Пауза")

    def choose_color(self):
        color = colorchooser.askcolor()[0]

        if color:
            color_bgr = np.uint8([[list(map(int, color))]])
            hsv_color = cv2.cvtColor(color_bgr, cv2.COLOR_RGB2HSV)

            h_val = int(hsv_color[0][0][0])
            s_val = int(hsv_color[0][0][1])
            v_val = int(hsv_color[0][0][2])

            tol_h = 2
            tol_s = 130
            tol_v = 130

            self.lowerBound = np.array([
                max(h_val - tol_h, 0),
                max(s_val - tol_s, 0),
                max(v_val - tol_v, 0)
            ], dtype=np.uint8)
            self.upperBound = np.array([
                min(h_val + tol_h, 179),
                min(s_val + tol_s, 255),
                min(v_val + tol_v, 255)
            ], dtype=np.uint8)


app = Application()
app.mainloop()

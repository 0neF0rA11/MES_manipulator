import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, colorchooser
from PIL import Image, ImageTk
from server import Server


class Application(tk.Tk):
    def __init__(self):
        super().__init__()
        self.min_area = 250

        self.image_size = (800, 560)
        self.video_running = False
        self.video_paused = False
        self.is_camera_on = False
        self.cap = None
        self.kernelOpen = np.ones((5, 5))
        self.kernelClose = np.ones((20, 20))

        self.lowerBound = np.array([0, 0, 0], dtype=np.uint8)
        self.upperBound = np.array([179, 255, 255], dtype=np.uint8)
        self.font = cv2.FONT_HERSHEY_SIMPLEX

        self.objects_coord = []

        self.title("Машинное зрение")

        self.tk.call('tk', 'scaling', 1.5)
        self.state('zoomed')

        self.ser = Server()

        self.k = 83 / 70
        self.cm_to_pixel = int(250 * self.k)
        self.x_0, self.y_0 = self.cm_to_pixel, self.cm_to_pixel
        self.x_max = self.x_0 + self.cm_to_pixel
        self.y_max = self.y_0 + self.cm_to_pixel

        self.exposure = 0
        self.white_balance = 0
        self.color_components = 0

        self.sent_data_label = ttk.Label(self, text="Sent: None", font=("Arial", 14, "bold"))
        self.sent_data_label.place(relx=.05, rely=.85, anchor="sw")

        self.received_data_label = ttk.Label(self, text="Received: None", font=("Arial", 14, "bold"))
        self.received_data_label.place(relx=.05, rely=.9, anchor="sw")

        self.create_port_widgets()
        self.create_camera_widgets()
        self.create_composition_settings()

    def create_port_widgets(self):
        connection_frame = ttk.Frame(self, padding="20")
        connection_frame.grid(row=0, column=0, sticky=(tk.W, tk.E))

        ttk.Label(connection_frame, text="Port:").grid(column=0, row=0, sticky=tk.W)
        port_combobox = ttk.Combobox(connection_frame, values=self.ser.ports)
        port_combobox.grid(column=1, row=0)

        ttk.Label(connection_frame, text="Baud Rate:").grid(column=0, row=1, sticky=tk.W)
        baudrate_combobox = ttk.Combobox(connection_frame, values=[9600, 19200, 38400, 57600, 115200], state="readonly")
        baudrate_combobox.grid(column=1, row=1)
        baudrate_combobox.current(4)

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

        ttk.Label(control_frame, text="Выбор камеры:").grid(column=0, row=0, sticky=tk.W)

        self.camera_selection = ttk.Combobox(control_frame, values=self.detect_cameras(), state="readonly")
        self.camera_selection.grid(column=1, row=0)
        if self.camera_selection['values']:  # Если камеры найдены, установить первую камеру по умолчанию
            self.camera_selection.current(0)
        else:
            self.camera_selection.set("Камеры не найдены")

        ttk.Button(control_frame, text="Запустить камеру", command=self.start_cv2_camera).grid(column=2, row=0, padx=5)
        ttk.Button(control_frame, text="Запустить нейросеть", command=self.start_neuro_camera).grid(column=3, row=0, padx=5)
        ttk.Button(control_frame, text="Остановить", command=self.stop_camera).grid(column=4, row=0, padx=5)

        self.pause_button = ttk.Button(control_frame, text="Пауза", command=self.pause_camera)
        self.pause_button.grid(column=5, row=0, padx=5)

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

    def send_coords(self):
        first_object = sorted(self.objects_coord, key=lambda point: point[0]**2 + point[1]**2)[0]
        self.ser.send_command(
            f"G00 X {first_object[0]} Y {first_object[1]} Z {70}\n",
            self.sent_data_label
        )

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

    def detect_cameras(self):
        index = 0
        available_cameras = []
        while True:
            cap = cv2.VideoCapture(index)
            if not cap.isOpened():
                break
            ret, _ = cap.read()
            if ret:
                available_cameras.append(f"Камера {index}")
            cap.release()
            index += 1
        return available_cameras

    def start_cv2_camera(self):
        if not self.video_running and self.camera_selection.get() != "Камеры не найдены":
            camera_index = int(self.camera_selection.get().split()[-1])
            self.cap = cv2.VideoCapture(camera_index)
            self.video_running = True
            self.video_paused = False
            self.show_frame()

    def start_neuro_camera(self):
        # TODO: ultralytics
        # camera_index = int(self.camera_selection.get().split()[-1])
        # self.cap = cv2.VideoCapture(camera_index)
        # self.video_running = True
        # self.video_paused = False
        # self.show_frame()
        pass

    def pick_color(self, event):
        if self.cap and self.video_running:
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

    def show_frame(self):
        if self.cap and self.video_running:
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
                        center_x_cam, center_y_cam = center_x - self.image_size[0] // 2, center_y - self.image_size[1] // 2
                        center_x_coord, center_y_coord = center_x_cam + self.x_0, center_y_cam + self.y_0
                        if w * h > self.min_area and 0 <= center_x_coord <= self.x_max and 0 <= center_y_coord <= self.y_max:
                            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                            cv2.putText(frame, str(i + 1), (x, y - 10), self.font, 0.5, (0, 255, 255), 2)
                            self.objects_coord.append((int(center_x_coord * 1 / self.k), int(center_y_coord * 1 / self.k)))

                    center_x = self.image_size[0] // 2
                    center_y = self.image_size[1] // 2

                    line_length = 20
                    cv2.line(frame, (center_x - line_length // 2, center_y), (center_x + line_length // 2, center_y),
                             (255, 0, 0), 1)
                    cv2.line(frame, (center_x, center_y - line_length // 2), (center_x, center_y + line_length // 2),
                             (255, 0, 0), 1)

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
        if self.video_running:
            self.video_paused = not self.video_paused
            if self.video_paused:
                self.pause_button.configure(text="Продолжить")
            else:
                self.pause_button.configure(text="Пауза")

    def stop_camera(self):
        if self.video_running:
            self.video_running = False
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

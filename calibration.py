import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import cv2
import numpy as np


class CalibrationWindow(tk.Toplevel):
    def __init__(self, parent, image_size, available_cameras):
        super().__init__(parent)
        self.flag = None
        self.h = None
        self.w = None
        self.min_area = 250
        self.image_size = image_size
        self.lowerBound = np.array([0, 0, 0], dtype=np.uint8)
        self.upperBound = np.array([179, 255, 255], dtype=np.uint8)
        self.kernelOpen = np.ones((5, 5))
        self.kernelClose = np.ones((20, 20))
        self.font = cv2.FONT_HERSHEY_SIMPLEX

        self.cap = None
        self.video_cv2_running = False
        self.title("Калибровочное окно")
        self.geometry("1280x720+100+100")

        ttk.Button(self, text="Закрыть", command=self.destroy).place(relx=0.01, rely=0.01)

        camera_frame = ttk.Frame(self, padding='10')
        camera_frame.place(relx=0.65, rely=0.02, anchor='n')

        placeholder_img = Image.open("placeholder.jpg")
        placeholder_img = placeholder_img.resize(image_size)
        self.placeholder_photo = ImageTk.PhotoImage(placeholder_img)

        self.video_label = ttk.Label(self, image=self.placeholder_photo)
        self.video_label.place(relx=0.65, rely=0.02, anchor='n')
        self.video_label.bind("<Button-1>", self.pick_color)

        control_frame = ttk.Frame(self, padding='10')
        control_frame.place(relx=0.15, rely=0.1, anchor='n')

        ttk.Label(control_frame, text="Выбор камеры:").grid(column=0, row=0, sticky=tk.W)
        ttk.Button(control_frame, text="Запустить", command=self.start).grid(column=3, row=0, sticky=tk.W)

        self.camera_selection = ttk.Combobox(control_frame, values=available_cameras, state="readonly", width=10)
        self.camera_selection.grid(column=1, row=0)
        if self.camera_selection['values']:
            self.camera_selection.current(0)
        else:
            self.camera_selection.set("Камеры не найдены")

        settings_frame = ttk.Frame(self, padding='10')
        settings_frame.place(relx=0.17, rely=0.2, anchor='n', width=350)

        ttk.Label(settings_frame, text="Инструкция", font=('Arial', 12)).grid(column=0, row=0, columnspan=2, pady=10)

        instructions = ("1. Поставьте объект в центр камеры.\n"
                        "2. Наведите на объект курсор мыши и нажмите.\n"
                        "Важно, чтобы выделялся лишь один объект!\n"
                        "3. Введите параметры:")
        ttk.Label(settings_frame, text=instructions, justify=tk.LEFT).grid(column=0, row=1, columnspan=2, sticky=tk.W)

        fields = ["Длина объекта, мм:", "Ширина объекта, мм:", "Высота объекта, мм:",
                  "Длина поля по оси X, мм:", "Длина поля по оси Y, мм:"]
        self.entries = {}
        for i, field in enumerate(fields):
            ttk.Label(settings_frame, text=field).grid(column=0, row=i + 2, sticky=tk.W, pady=5)
            entry = ttk.Entry(settings_frame, width=15)
            entry.grid(column=1, row=i + 2, pady=5)
            self.entries[field] = entry

        ttk.Button(settings_frame, text="Записать", command=self.save_config).grid(column=0, row=len(fields) + 2, columnspan=2, pady=10)

    def save_config(self):
        data = {label: entry.get() for label, entry in zip(['k_y', 'k_x', 'h', 'field_x', 'field_y'], self.entries.values())}
        data['k_x'] = self.w / int(data['k_x'])
        data['k_y'] = self.h / int(data['k_y'])
        with open("config.txt", "w") as file:
            file.write(f"k_x {data['k_x']}\n")
            file.write(f"k_y {data['k_y']}\n")
            file.write(f"h {int(data['h'])}\n")
            file.write(f"len_f_x {int(data['field_x'])}\n")
            file.write(f"len_f_y {int(data['field_y'])}\n")
        self.flag = True
        self.destroy()

    def start(self):
        if self.camera_selection.get() != "Камеры не найдены":
            camera_index = int(self.camera_selection.get().split()[-1])
            self.cap = cv2.VideoCapture(camera_index)
            if not self.video_cv2_running:
                self.video_cv2_running = True
                self.show_frame()

    def show_frame(self):
        if self.cap and self.video_cv2_running:
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.resize(frame, self.image_size)
                imgHSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

                mask = cv2.inRange(imgHSV, self.lowerBound, self.upperBound)

                maskOpen = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernelOpen)
                maskClose = cv2.morphologyEx(maskOpen, cv2.MORPH_CLOSE, self.kernelClose)

                maskFinal = maskClose
                conts, _ = cv2.findContours(maskFinal.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                for i, contour in enumerate(conts):
                    x, y, w, h = cv2.boundingRect(contour)
                    self.w, self.h = w, h
                    if w * h > self.min_area:
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                        cv2.putText(frame, str(i + 1), (x, y - 10), self.font, 0.5, (0, 255, 255), 2)

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

    def pick_color(self, event):
        if self.cap and self.video_cv2_running:
            x, y = event.x, event.y
            image_width, image_height = self.image_size

            if 0 <= x < image_width and 0 <= y < image_height:
                frame_x = int((x / image_width) * self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                frame_y = int((y / image_height) * self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

                _, frame = self.cap.read()
                if frame is not None:
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


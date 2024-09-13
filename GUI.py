import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, colorchooser
from PIL import Image, ImageTk

# Инициализация окна tkinter
root = tk.Tk()
root.title("Настройка параметров камеры и выделение области по цвету")
root.geometry("800x600")

# Захват видео с веб-камеры
cap = cv2.VideoCapture(0)

# Глобальные переменные для хранения цветовых границ
lower_bound = np.array([0, 0, 0])
upper_bound = np.array([0, 0, 0])

# Глобальные переменные для управления параметрами
exposure = 0
white_balance = 0
color_components = 0


# Функция для обновления изображения в tkinter
def update_image():
    ret, frame = cap.read()

    if ret:
        # Применяем настройки к изображению
        frame = apply_settings(frame)

        # Преобразуем изображение для tkinter
        processed_frame = process_frame(frame)
        img = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        imgtk = ImageTk.PhotoImage(image=img)
        camera_label.imgtk = imgtk
        camera_label.configure(image=imgtk)

    camera_label.after(10, update_image)


# Функция для применения настроек к изображению
def apply_settings(frame):
    global exposure, white_balance, color_components

    # Применение экспозиции (яркость)
    frame = cv2.convertScaleAbs(frame, alpha=1, beta=exposure)  # alpha отвечает за контраст, beta за яркость

    # Баланс белого (коррекция LAB)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(frame)
    l = cv2.add(l, white_balance)  # Управление уровнем освещения
    frame = cv2.merge([l, a, b])
    frame = cv2.cvtColor(frame, cv2.COLOR_LAB2BGR)

    # Цветоразностные составляющие (регулировка насыщенности)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    s = cv2.add(s, color_components)
    hsv = cv2.merge([h, s, v])
    frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return frame


# Функция для обработки кадра: поиск области по цвету
def process_frame(frame, min_width=50, min_height=50):
    # Преобразуем кадр в цветовое пространство HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Создаем маску для выделения пикселей в пределах цветовых границ
    mask = cv2.inRange(hsv, lower_bound, upper_bound)

    # Находим контуры областей
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Если найдены контуры, рисуем прямоугольник вокруг самой большой области
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        
        # Проверяем, больше ли ширина и высота минимальных значений
        if w >= min_width and h >= min_height:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return frame



# Функция для выбора цвета пользователем
def choose_color():
    global lower_bound, upper_bound
    # Открытие палитры для выбора цвета
    color = colorchooser.askcolor()[0]

    if color:
        # Преобразуем цвет из RGB в HSV
        color_bgr = np.uint8([[list(map(int, color))]])  # Преобразуем в формат для OpenCV
        hsv_color = cv2.cvtColor(color_bgr, cv2.COLOR_RGB2HSV)

        # Определяем диапазоны для поиска (допустимая погрешность ±20 для hue)
        hue = hsv_color[0][0][0]
        lower_bound = np.array([hue - 20, 50, 50])
        upper_bound = np.array([hue + 20, 255, 255])
        


# Функции для обновления значений ползунков
def update_exposure(value):
    global exposure
    exposure = int(float(value))  # Convert to float first, then to int

def update_white_balance(value):
    global white_balance
    white_balance = int(float(value))  # Convert to float first, then to int

def update_color_components(value):
    global color_components
    color_components = int(float(value))  # Convert to float first, then to int


# Метка для отображения изображения с веб-камеры
camera_label = ttk.Label(root)
camera_label.pack(pady=20)

# Кнопка для выбора цвета
color_button = ttk.Button(root, text="Выбрать цвет", command=choose_color)
color_button.pack(pady=10)


# Ползунки для управления параметрами
def create_slider(text, from_, to_, command, default=0):
    label = ttk.Label(root, text=f"{text}: {default}")
    label.pack(pady=5)

    slider = ttk.Scale(root, from_=from_, to=to_, orient="horizontal", command=command)
    slider.set(default)
    slider.pack()

    return slider


# Ползунки для управления экспозицией, балансом белого и цветоразностными составляющими
exposure_slider = create_slider("Настройка экспозиции", -100, 100, update_exposure, default=0)
white_balance_slider = create_slider("Баланс белого", -100, 100, update_white_balance, default=0)
color_components_slider = create_slider("Цветоразностные составляющие", -100, 100, update_color_components, default=0)

# Запуск обновления изображения
update_image()

# Запуск основного цикла tkinter
root.mainloop()

# Освобождение камеры и закрытие всех окон
cap.release()
cv2.destroyAllWindows()

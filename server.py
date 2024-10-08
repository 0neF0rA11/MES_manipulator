#!/usr/bin/python3
import time
import serial
import serial.tools.list_ports


class Server:
    def __init__(self):
        self.received_data = None
        self.ser = None
        self.ports = None
        self.get_ports()

    def get_ports(self):
        ports = serial.tools.list_ports.comports()
        self.ports = [port.device for port in ports]

    def connect_port(self, port, baudrate, label, page, received_label):
        try:
            self.ser = serial.Serial(
                port=port,
                baudrate=baudrate,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
            )
            time.sleep(1)

            # Обновление метки графического интерфейса
            label.config(text="Connected to " + port)

            # Отправка простого сообщения при подключении
            self.ser.write("UART Demonstration Program\r\n".encode())
            self.ser.write("NVIDIA Jetson Nano Developer Kit\r\n".encode())

            # Начало чтения данных с порта
            self.read_from_port(page, received_label)
        except Exception as e:
            label.config(text="Connection Failed")
            print(f"Error: {e}")

    def close_port(self, label):
        if self.ser and self.ser.is_open:
            self.ser.close()
            label.config(text="Port Closed")
        else:
            label.config(text="No Port is Open")

    def read_from_port(self, page, received_label):
        if self.ser and self.ser.is_open:
            if self.ser.in_waiting > 0:
                # Чтение данных из порта
                self.received_data = self.ser.read(self.ser.in_waiting).decode('utf-8', errors='replace').strip()

                # Обновление метки полученных данных
                received_label.config(text=f"Received: {self.received_data}")

                # Передача данных для обработки (например, в интерфейс)
                page.response_to_request(self.received_data.split())

                # Эхо-ответ (отправляем обратно то, что получили)
                self.ser.write(self.received_data.encode())

                # Добавление перевода строки для удобства работы с Windows системами
                if "\r" in self.received_data:
                    self.ser.write("\n".encode())

            # Повторный вызов функции для асинхронного чтения данных
            page.after(1, lambda: self.read_from_port(page, received_label))

    def send_command(self, command, label):
        if self.ser and self.ser.is_open:
            self.ser.write(command.encode())
            time.sleep(0.002)
        label.config(text=f"Sent: {command.strip()}")

# Пример использования:
# server = Server()
# server.connect_port("/dev/ttyTHS1", 115200, some_label, some_page, received_label)

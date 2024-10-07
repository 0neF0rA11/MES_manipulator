import serial
import serial.tools.list_ports
import time


class Server:
    def __init__(self):
        self.received_data = None
        self.ser = None
        self.ports = None
        self.get_ports()

    def connect_port(self, port, baudrate, label, page, received_label):
        try:
            self.ser = serial.Serial(port=port, baudrate=baudrate)
            label.config(text="Connected to " + port)
            self.read_from_port(page, received_label)
        except Exception as e:
            label.config(text="Connection Failed")

    def get_ports(self):
        ports = serial.tools.list_ports.comports()
        self.ports = [port.device for port in ports]

    def close_port(self, label):
        if self.ser and self.ser.is_open:
            self.ser.close()
            label.config(text="Port Closed")
        else:
            label.config(text="No Port is Open")

    def read_from_port(self, page, received_label):
        if self.ser and self.ser.is_open:
            if self.ser.in_waiting > 0:
                self.received_data = self.ser.read(self.ser.in_waiting).decode('utf-8', errors='replace').strip()
                received_label.config(text=f"Received: {self.received_data}")
                page.response_to_request(self.received_data.split())
            page.after(1, lambda: self.read_from_port(page, received_label))

    def send_command(self, command, label):
        if self.ser and self.ser.is_open:
            self.ser.write(command.encode())
            time.sleep(0.002)
        label.config(text=f"Sent: {command.strip()}")

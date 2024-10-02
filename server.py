import socket
import threading


class Server:
    def __init__(self, label, host='0.0.0.0', port=12345):
        self.host = host
        self.port = port
        self.server_socket = None
        self.client_socket = None
        self.address = None
        self.server_thread = threading.Thread(target=self.start_server, args=(label,), daemon=True)
        self.server_thread.start()

    def start_server(self, label):
        # Create a TCP/IP socket
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(1)

        # Wait for a client to connect
        self.client_socket, self.address = self.server_socket.accept()
        label.config(text=f"Connection from {self.address} established!")

    def close_port(self, label):
        if self.client_socket:
            self.client_socket.close()
            label.config(text="Connection Closed")
        else:
            label.config(text="No Connection is Open")

    def send_command(self, command, label):
        try:
            self.client_socket.sendall(command.encode('utf-8'))
            label.config(text=f"Sent: {command.strip()}")
        except Exception as e:
            label.config(text="Error sending command")


import socket
import threading

# Server setup
HOST = '127.0.0.1'  # Localhost
PORT = 12345         # Any open port number

# List to keep track of all connected clients
clients = []

def broadcast(message, sender_socket):
    """Send message to all clients except the sender."""
    for client in clients:
        if client != sender_socket:
            try:
                client.send(message)
            except:
                # If the send fails, remove the client
                clients.remove(client)

def handle_client(client_socket):
    """Handle individual client communication."""
    while True:
        try:
            message = client_socket.recv(1024)
            if message:
                broadcast(message, client_socket)
            else:
                break
        except:
            break

    # Remove client if disconnected
    clients.remove(client_socket)
    client_socket.close()

def start_server():
    """Initialize and run the server."""
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind((HOST, PORT))
    server.listen(5)
    print(f'Server started on {HOST}:{PORT}')

    while True:
        client_socket, client_address = server.accept()
        print(f'New connection: {client_address}')
        clients.append(client_socket)

        # Handle client communication in a new thread
        thread = threading.Thread(target=handle_client, args=(client_socket,))
        thread.start()

if __name__ == "__main__":
    start_server()

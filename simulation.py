import numpy as np

class Network:
    def __init__(self, n_servers, model):
        self.n_servers = n_servers  # List of Server instances
        self.model = model
    
    '''
    def broadcast_ping(self):
        for server in self.servers:
            for other_server in self.servers:
                if server != other_server:
                    server.ping(other_server)
    '''
                    
class Server:
    def __init__(self, server_id, model, buffer_size, arrival_rate, departure_rate):
        self.server_id = server_id
        self.model = model
        self.buffer_size = buffer_size
        self.n_packets = 0
        self.arrival_rate = arrival_rate
        self.departure_rate = departure_rate

    def update(self):
        # Update server state, e.g., process packets
        pass

    def ping(self, other_server):
        # Simulate pinging another server
        print(f"Pinging server {other_server.server_id} from server {self.server_id}")

    def can_accept_packet(self):
        return (self.n_packets + 1) <= self.buffer_size

class MemorylessServer(Server):
    def __init__(self, server_id, model, buffer_size, arrival_rate, departure_rate):
        super().__init__(server_id, model, buffer_size, arrival_rate, departure_rate)
        # Additional initialization for memoryless server

    def update(self):
        # Specific update logic for memoryless server
        pass

class DeterministicServer(Server):
    def __init__(self, server_id, model, buffer_size, arrival_rate, departure_rate):
        super().__init__(server_id, model, buffer_size, arrival_rate, departure_rate)
        # Additional initialization for deterministic server

    def update(self):
        # Specific update logic for deterministic server
        pass

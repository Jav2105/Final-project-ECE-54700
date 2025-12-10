
class Network:
    def __init__(self, n_servers, model):
        self.n_servers = n_servers  # List of Server instances
        self.model = model
    
    def broadcast_ping(self):
        for server in self.servers:
            for other_server in self.servers:
                if server != other_server:
                    server.ping(other_server)

class Server:
    def __init__(self, server_id, model, buffer_size):
        self.server_id = server_id
        self.model = model
        self.buffer_size = buffer_size
        self.n_packets = 0

    def ping(self, other_server):
        # Simulate pinging another server
        print(f"Pinging server {other_server.server_id} from server {self.server_id}")

    def can_accept_task(self, task_load):
        return (self.current_load + task_load) <= self.capacity

    def assign_task(self, task_load):
        if self.can_accept_task(task_load):
            self.current_load += task_load
            return True
        return False

    def release_task(self, task_load):
        self.current_load = max(0, self.current_load - task_load)

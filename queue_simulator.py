import numpy as np
import matplotlib.pyplot as plt

class Server:
    def __init__(self, arrival_rate, service_rate, buffer_size=np.inf):
        self.lam = arrival_rate
        self.mu = service_rate
        self.buffer_size = buffer_size
        self.queue = 0
        self.drops = 0
        self.history = []

    def log(self):
        self.history.append(self.queue)


class MM1N(Server):
    def update(self, dt):
        # Arrival
        if np.random.rand() < self.lam * dt:
            if self.queue < self.buffer_size:
                self.queue += 1
            else:
                self.drops += 1

        # Service
        if self.queue > 0 and np.random.rand() < self.mu * dt:
            self.queue -= 1

        self.log()


class DD1(Server):
    def __init__(self, arrival_rate, service_rate):
        super().__init__(arrival_rate, service_rate)
        self.arrival_timer = 0
        self.service_timer = 0

    def update(self, dt):
        self.arrival_timer += dt
        self.service_timer += dt

        if self.arrival_timer >= 1 / self.lam:
            self.queue += 1
            self.arrival_timer = 0

        if self.queue > 0 and self.service_timer >= 1 / self.mu:
            self.queue -= 1
            self.service_timer = 0

        self.log()


def run_sim(server, T=1000, dt=0.001):
    steps = int(T / dt)
    for _ in range(steps):
        server.update(dt)
    return np.array(server.history)


# ----------- EXPERIMENTS -----------

# M/M/1 example
lam = 0.8
mu = 1.0

mm1 = MM1N(lam, mu, buffer_size=np.inf)
q_mm1 = run_sim(mm1)

# M/M/1/N example
mm1n = MM1N(lam, mu, buffer_size=10)
q_mm1n = run_sim(mm1n)

# D/D/1 example
dd1 = DD1(lam, mu)
q_dd1 = run_sim(dd1)

# ----------- PLOTS -----------

plt.figure()
plt.plot(q_mm1[:5000])
plt.title("M/M/1 Queue Length Over Time")
plt.xlabel("Time Step")
plt.ylabel("Queue Length")

plt.figure()
plt.plot(q_mm1n[:5000])
plt.title("M/M/1/N Queue Length Over Time (N=10)")
plt.xlabel("Time Step")
plt.ylabel("Queue Length")

plt.figure()
plt.plot(q_dd1[:5000])
plt.title("D/D/1 Queue Length Over Time")
plt.xlabel("Time Step")
plt.ylabel("Queue Length")

print("M/M/1 avg queue:", q_mm1.mean())
print("M/M/1/N avg queue:", q_mm1n.mean())
print("M/M/1/N drop rate:", mm1n.drops / len(q_mm1n))

plt.show()

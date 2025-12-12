# queue_network_sim.py
# Discrete-event simulations for M/M/1, M/M/1/N, D/D/1 and a small multi-hop network.
# Standard-library only (no numpy required).

from __future__ import annotations
import argparse, heapq, math, random
from dataclasses import dataclass, field
from collections import deque, defaultdict
from typing import Callable, Deque, Dict, List, Optional, Tuple

# -------------------- Event engine --------------------

@dataclass(order=True)
class Event:
    time: float
    seq: int
    action: Callable[[], None] = field(compare=False)

class Sim:
    def __init__(self, seed: int = 0):
        self.t = 0.0
        self._pq: List[Event] = []
        self._seq = 0
        self.rng = random.Random(seed)

    def schedule(self, t: float, fn: Callable[[], None]) -> None:
        self._seq += 1
        heapq.heappush(self._pq, Event(t, self._seq, fn))

    def run(self, t_end: float) -> None:
        while self._pq and self._pq[0].time <= t_end:
            ev = heapq.heappop(self._pq)
            self.t = ev.time
            ev.action()
        self.t = t_end

def ia_time(rng: random.Random, rate: float, kind: str) -> float:
    if rate <= 0: return math.inf
    if kind == "poisson": return rng.expovariate(rate)
    if kind == "deterministic": return 1.0 / rate
    raise ValueError(kind)

def svc_time(rng: random.Random, mu: float, kind: str) -> float:
    if mu <= 0: return math.inf
    if kind == "exp": return rng.expovariate(mu)
    if kind == "det": return 1.0 / mu
    raise ValueError(kind)

# -------------------- Single queue --------------------

@dataclass
class SQStats:
    arrivals: int = 0
    accepted: int = 0
    dropped: int = 0
    departures: int = 0
    delay_sum: float = 0.0
    delay_cnt: int = 0
    area_n: float = 0.0
    last_t: float = 0.0

class SingleQueue:
    # FIFO, 1 server, optional finite capacity cap (includes job in service)
    def __init__(self, sim: Sim, mu: float, cap: Optional[int], service_kind: str):
        self.sim = sim
        self.mu = mu
        self.cap = cap
        self.service_kind = service_kind
        self.q: Deque[float] = deque()     # arrival times of waiting jobs
        self.busy = False
        self.stats = SQStats(last_t=sim.t)

    def _n(self) -> int:
        return len(self.q) + (1 if self.busy else 0)

    def _area_update(self) -> None:
        t = self.sim.t
        self.stats.area_n += self._n() * (t - self.stats.last_t)
        self.stats.last_t = t

    def arrival(self) -> None:
        self._area_update()
        self.stats.arrivals += 1

        if self.cap is not None and self._n() >= self.cap:
            self.stats.dropped += 1
            return

        self.stats.accepted += 1
        if not self.busy:
            self.busy = True
            at = self.sim.t
            self.sim.schedule(at + svc_time(self.sim.rng, self.mu, self.service_kind),
                              lambda at=at: self._depart(at))
        else:
            self.q.append(self.sim.t)

    def _depart(self, at: float) -> None:
        self._area_update()
        self.stats.departures += 1
        self.stats.delay_sum += (self.sim.t - at)
        self.stats.delay_cnt += 1

        if self.q:
            next_at = self.q.popleft()
            self.sim.schedule(self.sim.t + svc_time(self.sim.rng, self.mu, self.service_kind),
                              lambda at=next_at: self._depart(at))
        else:
            self.busy = False

    def reset_measurement(self) -> None:
        self.stats = SQStats(last_t=self.sim.t)

    def summary(self, meas_time: float) -> Dict[str, float]:
        self._area_update()
        avg_n = self.stats.area_n / max(meas_time, 1e-12)
        avg_T = self.stats.delay_sum / self.stats.delay_cnt if self.stats.delay_cnt else float("nan")
        drop = self.stats.dropped / self.stats.arrivals if self.stats.arrivals else 0.0
        thr = self.stats.departures / max(meas_time, 1e-12)
        return dict(avg_delay=avg_T, avg_n=avg_n, drop_prob=drop, throughput=thr,
                    arrivals=self.stats.arrivals, dropped=self.stats.dropped, departures=self.stats.departures)

def run_single_queue(lam: float, mu: float, cap: Optional[int],
                     arrival_kind: str, service_kind: str,
                     t_end: float = 4000.0, warmup: float = 400.0, seed: int = 0) -> Dict[str, float]:
    sim = Sim(seed=seed)
    q = SingleQueue(sim, mu=mu, cap=cap, service_kind=service_kind)

    def on_arrival():
        q.arrival()
        sim.schedule(sim.t + ia_time(sim.rng, lam, arrival_kind), on_arrival)

    sim.schedule(0.0, on_arrival)
    if warmup > 0:
        sim.schedule(warmup, q.reset_measurement)
    sim.run(t_end)
    return q.summary(meas_time=max(t_end - warmup, 1e-12))

# -------------------- Network --------------------

@dataclass
class Packet:
    pid: int
    flow_id: int
    dst: int
    birth_t: float
    hop: int = 0

@dataclass
class Flow:
    flow_id: int
    src: int
    dst: int
    lam: float
    arrival_kind: str = "poisson"

@dataclass
class FlowStats:
    injected: int = 0
    dropped: int = 0
    delivered: int = 0
    delay_sum: float = 0.0
    delay_cnt: int = 0

@dataclass
class LinkStats:
    arrivals_epoch: int = 0
    dropped: int = 0
    departures: int = 0
    area_n: float = 0.0
    last_t: float = 0.0

class Link:
    # FIFO per-link queue + single-server
    def __init__(self, sim: Sim, u: int, v: int, mu: float,
                 cap: Optional[int] = None, service_kind: str = "exp", name: Optional[str] = None):
        self.sim = sim
        self.u, self.v = u, v
        self.mu = mu
        self.cap = cap
        self.service_kind = service_kind
        self.name = name or f"{u}->{v}"

        self.q: Deque[Packet] = deque()
        self.busy = False
        self.in_service: Optional[Packet] = None
        self.stats = LinkStats(last_t=sim.t)

        self.on_depart: Callable[[Packet], None] = lambda pkt: None

    def _n(self) -> int:
        return len(self.q) + (1 if self.busy else 0)

    def _area_update(self) -> None:
        t = self.sim.t
        self.stats.area_n += self._n() * (t - self.stats.last_t)
        self.stats.last_t = t

    def enqueue(self, pkt: Packet) -> bool:
        self._area_update()
        self.stats.arrivals_epoch += 1

        if self.cap is not None and self._n() >= self.cap:
            self.stats.dropped += 1
            return False

        if not self.busy:
            self._start(pkt)
        else:
            self.q.append(pkt)
        return True

    def _start(self, pkt: Packet) -> None:
        self.busy = True
        self.in_service = pkt
        self.sim.schedule(self.sim.t + svc_time(self.sim.rng, self.mu, self.service_kind), self._depart)

    def _depart(self) -> None:
        self._area_update()
        self.stats.departures += 1
        pkt = self.in_service
        assert pkt is not None
        self.on_depart(pkt)

        if self.q:
            self._start(self.q.popleft())
        else:
            self.busy = False
            self.in_service = None

    def reset_measurement(self) -> None:
        self.stats = LinkStats(last_t=self.sim.t)

class Router:
    def on_epoch(self, t: float) -> None:
        pass
    def choose(self, node: int, dst: int, rng: random.Random) -> Optional[Link]:
        raise NotImplementedError

class StaticShortestHop(Router):
    # hop-distance BFS (reverse graph), uniform tie-break
    def __init__(self, nodes: List[int], out_links: Dict[int, List[Link]], destinations: List[int]):
        self.nodes = nodes
        self.out_links = out_links
        self.destinations = destinations
        self.dist: Dict[int, Dict[int, int]] = {d: {n: math.inf for n in nodes} for d in destinations}
        rev_adj: Dict[int, List[int]] = defaultdict(list)
        for u, links in out_links.items():
            for lk in links:
                rev_adj[lk.v].append(lk.u)

        for d in destinations:
            q = deque([d])
            self.dist[d][d] = 0
            while q:
                x = q.popleft()
                for prev in rev_adj.get(x, []):
                    if self.dist[d][prev] == math.inf:
                        self.dist[d][prev] = self.dist[d][x] + 1
                        q.append(prev)

    def choose(self, node: int, dst: int, rng: random.Random) -> Optional[Link]:
        links = self.out_links.get(node, [])
        if not links: return None
        dn = self.dist[dst].get(node, math.inf)
        if dn == math.inf: return None
        cand = [lk for lk in links if self.dist[dst].get(lk.v, math.inf) + 1 == dn]
        return rng.choice(cand) if cand else None

class QueueAware(Router):
    # epoch-based arrival-rate estimate -> delay proxy 1/(mu - lam_hat)
    def __init__(self, nodes: List[int], out_links: Dict[int, List[Link]], destinations: List[int],
                 epoch: float = 0.1, beta: float = 8.0, eps: float = 1e-6):
        self.nodes = nodes
        self.out_links = out_links
        self.destinations = destinations
        self.epoch = epoch
        self.beta = beta
        self.eps = eps

        self.w: Dict[Tuple[int, int], float] = {}
        self.dist: Dict[int, Dict[int, float]] = {d: {n: math.inf for n in nodes} for d in destinations}
        self.rev_links: Dict[int, List[Link]] = defaultdict(list)
        for u, links in out_links.items():
            for lk in links:
                self.rev_links[lk.v].append(lk)

        self._recompute()

    def _ekey(self, lk: Link) -> Tuple[int, int]:
        return (lk.u, lk.v)

    def _recompute(self) -> None:
        # weights
        for u, links in self.out_links.items():
            for lk in links:
                lam_hat = lk.stats.arrivals_epoch / max(self.epoch, self.eps)
                lk.stats.arrivals_epoch = 0
                lam_hat = min(lam_hat, lk.mu - 1e-6)
                self.w[self._ekey(lk)] = 1.0 / (lk.mu - lam_hat + self.eps)

        # Dijkstra on reversed graph per destination
        for d in self.destinations:
            dist = {n: math.inf for n in self.nodes}
            dist[d] = 0.0
            pq = [(0.0, d)]
            while pq:
                cd, x = heapq.heappop(pq)
                if cd != dist[x]: continue
                for lk in self.rev_links.get(x, []):
                    prev = lk.u
                    nd = cd + self.w.get(self._ekey(lk), 1.0)
                    if nd < dist[prev]:
                        dist[prev] = nd
                        heapq.heappush(pq, (nd, prev))
            self.dist[d] = dist

    def on_epoch(self, t: float) -> None:
        self._recompute()

    def choose(self, node: int, dst: int, rng: random.Random) -> Optional[Link]:
        links = self.out_links.get(node, [])
        if not links: return None
        if self.dist[dst].get(node, math.inf) == math.inf: return None

        scores: List[Tuple[Link, float]] = []
        for lk in links:
            tail = self.dist[dst].get(lk.v, math.inf)
            if tail == math.inf: continue
            cost = self.w.get(self._ekey(lk), 1.0) + tail
            scores.append((lk, cost))
        if not scores: return None

        minc = min(c for _, c in scores)
        ws = [math.exp(-self.beta * (c - minc)) for _, c in scores]
        s = sum(ws)
        r = rng.random() * s
        acc = 0.0
        for (lk, _), w in zip(scores, ws):
            acc += w
            if acc >= r:
                return lk
        return scores[-1][0]

class NetSim:
    def __init__(self, sim: Sim, nodes: List[int], links: List[Link], flows: List[Flow],
                 router: Router, t_end: float, warmup: float = 0.0, epoch: Optional[float] = None):
        self.sim = sim
        self.nodes = nodes
        self.links = links
        self.flows = flows
        self.router = router
        self.t_end = t_end
        self.warmup = warmup
        self.epoch = epoch

        self.pid = 0
        self.flow_stats: Dict[int, FlowStats] = {f.flow_id: FlowStats() for f in flows}
        self.out_links: Dict[int, List[Link]] = defaultdict(list)
        for lk in links:
            self.out_links[lk.u].append(lk)
            lk.on_depart = lambda pkt, lk=lk: self._on_depart(lk, pkt)

        if warmup > 0:
            self.sim.schedule(warmup, self._reset_measurement)
        if epoch is not None and epoch > 0:
            self.sim.schedule(0.0, self._epoch_tick)

    def _epoch_tick(self) -> None:
        self.router.on_epoch(self.sim.t)
        self.sim.schedule(self.sim.t + self.epoch, self._epoch_tick)

    def _reset_measurement(self) -> None:
        for lk in self.links:
            lk.reset_measurement()
        for fid in self.flow_stats:
            self.flow_stats[fid] = FlowStats()

    def _new_pkt(self, flow: Flow) -> Packet:
        self.pid += 1
        return Packet(pid=self.pid, flow_id=flow.flow_id, dst=flow.dst, birth_t=self.sim.t)

    def _inject(self, flow: Flow) -> None:
        st = self.flow_stats[flow.flow_id]
        st.injected += 1
        pkt = self._new_pkt(flow)

        lk = self.router.choose(flow.src, flow.dst, self.sim.rng)
        if lk is None:
            st.dropped += 1
        else:
            if not lk.enqueue(pkt):
                st.dropped += 1

        self.sim.schedule(self.sim.t + ia_time(self.sim.rng, flow.lam, flow.arrival_kind),
                          lambda flow=flow: self._inject(flow))

    def _on_depart(self, lk: Link, pkt: Packet) -> None:
        pkt.hop += 1
        node = lk.v
        if node == pkt.dst:
            st = self.flow_stats[pkt.flow_id]
            st.delivered += 1
            st.delay_sum += (self.sim.t - pkt.birth_t)
            st.delay_cnt += 1
            return

        nxt = self.router.choose(node, pkt.dst, self.sim.rng)
        if nxt is None:
            self.flow_stats[pkt.flow_id].dropped += 1
        else:
            if not nxt.enqueue(pkt):
                self.flow_stats[pkt.flow_id].dropped += 1

    def run(self) -> Dict[str, object]:
        for f in self.flows:
            self.sim.schedule(0.0, lambda f=f: self._inject(f))

        self.sim.run(self.t_end)
        meas_t = max(self.t_end - self.warmup, 1e-12)

        # aggregate flow stats
        tot_inj = sum(self.flow_stats[f.flow_id].injected for f in self.flows)
        tot_drop = sum(self.flow_stats[f.flow_id].dropped for f in self.flows)
        tot_del = sum(self.flow_stats[f.flow_id].delivered for f in self.flows)
        tot_delay = sum(self.flow_stats[f.flow_id].delay_sum for f in self.flows)
        tot_cnt = sum(self.flow_stats[f.flow_id].delay_cnt for f in self.flows)

        agg = dict(
            sim_time=meas_t,
            throughput=tot_del / meas_t,
            drop_prob=(tot_drop / tot_inj) if tot_inj else 0.0,
            mean_delay=(tot_delay / tot_cnt) if tot_cnt else float("nan"),
        )

        flows = {}
        for f in self.flows:
            st = self.flow_stats[f.flow_id]
            flows[f.flow_id] = dict(
                src=f.src, dst=f.dst,
                injected=st.injected, delivered=st.delivered, dropped=st.dropped,
                throughput=st.delivered / meas_t,
                avg_delay=(st.delay_sum / st.delay_cnt) if st.delay_cnt else float("nan"),
                drop_prob=(st.dropped / st.injected) if st.injected else 0.0,
            )

        links = {}
        for lk in self.links:
            lk._area_update()
            links[lk.name] = dict(mu=lk.mu, dropped=lk.stats.dropped,
                                  departures=lk.stats.departures,
                                  avg_n=lk.stats.area_n / meas_t)

        return dict(aggregate=agg, flows=flows, links=links)

# -------------------- Paper-friendly theory helpers --------------------

def mm1_ET(lam: float, mu: float) -> float: return 1.0 / (mu - lam)
def mm1_EN(lam: float, mu: float) -> float:
    rho = lam / mu
    return rho / (1 - rho)

def mm1n_p0(rho: float, N: int) -> float:
    if abs(rho - 1.0) < 1e-12: return 1.0 / (N + 1)
    return (1 - rho) / (1 - rho ** (N + 1))

def mm1n_pN(rho: float, N: int) -> float: return (rho ** N) * mm1n_p0(rho, N)

def mm1n_EN(lam: float, mu: float, N: int) -> float:
    rho = lam / mu
    if abs(rho - 1.0) < 1e-12: return N / 2.0
    num = rho * (1 - (N + 1) * rho ** N + N * rho ** (N + 1))
    den = (1 - rho) * (1 - rho ** (N + 1))
    return num / den

def mm1n_ET(lam: float, mu: float, N: int) -> float:
    rho = lam / mu
    pN = mm1n_pN(rho, N)
    lam_eff = lam * (1 - pN)
    return mm1n_EN(lam, mu, N) / lam_eff if lam_eff > 0 else float("inf")

# -------------------- Example experiments --------------------

def exp_small_network(policy: str, seed: int = 1) -> Dict[str, object]:
    """
    Topology (directed):
        0 -> 1 -> 4   (smaller capacity link 1->4)
        0 -> 2 -> 4   (higher capacity link 2->4)
    plus cross-traffic flow 1->4 to congest link 1->4.

    Queue-aware routing should shift more of Flow 0 onto the 0->2->4 path.
    """
    sim = Sim(seed=seed)
    nodes = [0, 1, 2, 4]
    cap = 25

    links = [
        Link(sim, 0, 1, mu=35.0, cap=cap, service_kind="exp", name="0->1"),
        Link(sim, 0, 2, mu=35.0, cap=cap, service_kind="exp", name="0->2"),
        Link(sim, 1, 4, mu=8.0,  cap=cap, service_kind="exp", name="1->4"),  # smaller capacity
        Link(sim, 2, 4, mu=16.0, cap=cap, service_kind="exp", name="2->4"),
    ]
    out_links = defaultdict(list)
    for lk in links:
        out_links[lk.u].append(lk)

    destinations = [4]
    if policy == "static":
        router = StaticShortestHop(nodes, out_links, destinations)
        epoch = None
    elif policy == "queue":
        router = QueueAware(nodes, out_links, destinations, epoch=0.1, beta=8.0)
        epoch = 0.1
    else:
        raise ValueError("policy must be 'static' or 'queue'")

    flows = [
        Flow(flow_id=0, src=0, dst=4, lam=18.0, arrival_kind="poisson"),
        Flow(flow_id=1, src=1, dst=4, lam=6.0,  arrival_kind="poisson"),  # cross traffic
    ]

    net = NetSim(sim, nodes, links, flows, router, t_end=3000, warmup=300, epoch=epoch)
    return net.run()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["mm1", "mm1n", "dd1", "net_static", "net_queue"], default="mm1")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    if args.mode == "mm1":
        lam, mu = 8.0, 10.0
        out = run_single_queue(lam, mu, None, "poisson", "exp", seed=args.seed)
        print("M/M/1 theory E[T] =", mm1_ET(lam, mu), "sim =", out["avg_delay"])
        print("M/M/1 theory E[N] =", mm1_EN(lam, mu), "sim =", out["avg_n"])

    if args.mode == "mm1n":
        lam, mu, N = 12.0, 10.0, 20
        out = run_single_queue(lam, mu, N, "poisson", "exp", seed=args.seed)
        print("M/M/1/N theory pN =", mm1n_pN(lam/mu, N), "sim =", out["drop_prob"])
        print("M/M/1/N theory E[T] =", mm1n_ET(lam, mu, N), "sim =", out["avg_delay"])
        print("M/M/1/N theory E[N] =", mm1n_EN(lam, mu, N), "sim =", out["avg_n"])

    if args.mode == "dd1":
        lam, mu = 8.0, 10.0
        out = run_single_queue(lam, mu, None, "deterministic", "det", t_end=2000, warmup=200, seed=args.seed)
        print("D/D/1 sojourn time should be ~1/mu =", 1/mu, "sim =", out["avg_delay"])
        print("D/D/1 E[N] should be rho =", lam/mu, "sim =", out["avg_n"])

    if args.mode == "net_static":
        res = exp_small_network("static", seed=args.seed)
        print("STATIC aggregate:", res["aggregate"])
        print("STATIC per-flow:", res["flows"])

    if args.mode == "net_queue":
        res = exp_small_network("queue", seed=args.seed)
        print("QUEUE aggregate:", res["aggregate"])
        print("QUEUE per-flow:", res["flows"])

if __name__ == "__main__":
    main()

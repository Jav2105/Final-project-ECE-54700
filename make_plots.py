# make_plots.py
import math
import matplotlib.pyplot as plt

from queue_network_sim import (
    run_single_queue, mm1_ET, mm1n_pN, exp_small_network
)

def avg_over_seeds(fn, seeds):
    vals = []
    for s in seeds:
        vals.append(fn(s))
    # average dict of scalars
    keys = vals[0].keys()
    out = {}
    for k in keys:
        out[k] = sum(v[k] for v in vals) / len(vals)
    return out

def plot_mm1_delay_vs_rho():
    mu = 10.0
    rhos = [0.1 * i for i in range(1, 10)] + [0.95]
    seeds = [1, 2, 3]

    sim_delays = []
    th_delays = []
    for rho in rhos:
        lam = rho * mu
        res = avg_over_seeds(
            lambda sd: run_single_queue(lam, mu, None, "poisson", "exp", seed=sd),
            seeds
        )
        sim_delays.append(res["avg_delay"])
        th_delays.append(mm1_ET(lam, mu))

    plt.figure()
    plt.plot(rhos, th_delays, label="Theory")
    plt.scatter(rhos, sim_delays, label="Simulation")
    plt.xlabel(r"Utilization $\rho=\lambda/\mu$")
    plt.ylabel(r"Mean sojourn time $E[T]$")
    plt.title("M/M/1: Delay vs Utilization")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("fig_mm1_delay_vs_rho.pdf")

def plot_mm1n_drop_vs_lambda():
    mu = 10.0
    N = 20
    lams = [2, 4, 6, 8, 10, 12, 14, 16]
    seeds = [1, 2, 3]

    sim_drop = []
    th_drop = []
    for lam in lams:
        res = avg_over_seeds(
            lambda sd: run_single_queue(lam, mu, N, "poisson", "exp", seed=sd),
            seeds
        )
        sim_drop.append(res["drop_prob"])
        th_drop.append(mm1n_pN(lam / mu, N))

    plt.figure()
    plt.plot(lams, th_drop, label="Theory")
    plt.scatter(lams, sim_drop, label="Simulation")
    plt.xlabel(r"Offered load $\lambda$")
    plt.ylabel(r"Drop probability $p_N$")
    plt.title(f"M/M/1/{N}: Drop vs Offered Load")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("fig_mm1n_drop_vs_lambda.pdf")

def plot_network_metrics_vs_load():
    from queue_network_sim import Sim, Link, Flow, NetSim, StaticShortestHop, QueueAware
    from collections import defaultdict

    def run_net(policy: str, lam0: float, seed: int):
        sim = Sim(seed=seed)
        nodes = [0, 1, 2, 4]
        cap = 25
        links = [
            Link(sim, 0, 1, mu=35.0, cap=cap, service_kind="exp", name="0->1"),
            Link(sim, 0, 2, mu=35.0, cap=cap, service_kind="exp", name="0->2"),
            Link(sim, 1, 4, mu=8.0,  cap=cap, service_kind="exp", name="1->4"),
            Link(sim, 2, 4, mu=16.0, cap=cap, service_kind="exp", name="2->4"),
        ]
        out_links = defaultdict(list)
        for lk in links:
            out_links[lk.u].append(lk)

        destinations = [4]
        if policy == "static":
            router = StaticShortestHop(nodes, out_links, destinations)
            epoch = None
        else:
            router = QueueAware(nodes, out_links, destinations, epoch=0.1, beta=8.0)
            epoch = 0.1

        flows = [
            Flow(flow_id=0, src=0, dst=4, lam=lam0, arrival_kind="poisson"),
            Flow(flow_id=1, src=1, dst=4, lam=6.0,  arrival_kind="poisson"),
        ]
        net = NetSim(sim, nodes, links, flows, router, t_end=3000, warmup=300, epoch=epoch)
        return net.run()["aggregate"]

    lams0 = [8, 12, 16, 18, 20, 22, 24]
    seeds = [1, 2, 3]

    def series(policy):
        thr, drop, delay = [], [], []
        for lam0 in lams0:
            agg = avg_over_seeds(lambda sd: run_net(policy, lam0, sd), seeds)
            thr.append(agg["throughput"])
            drop.append(agg["drop_prob"])
            delay.append(agg["mean_delay"])
        return thr, drop, delay

    thr_s, drop_s, delay_s = series("static")
    thr_q, drop_q, delay_q = series("queue")

    plt.figure()
    plt.plot(lams0, thr_s, label="Throughput (static)")
    plt.plot(lams0, thr_q, label="Throughput (queue-aware)")
    plt.xlabel(r"Flow-0 offered rate $\lambda_0$")
    plt.ylabel("Throughput (pkt/s)")
    plt.title("Network Throughput vs Load")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("fig_net_throughput_vs_load.pdf")

    plt.figure()
    plt.plot(lams0, delay_s, label="Delay (static)")
    plt.plot(lams0, delay_q, label="Delay (queue-aware)")
    plt.xlabel(r"Flow-0 offered rate $\lambda_0$")
    plt.ylabel("Mean delay (s)")
    plt.title("Network Mean Delay vs Load")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("fig_net_delay_vs_load.pdf")

    plt.figure()
    plt.plot(lams0, drop_s, label="Drop prob (static)")
    plt.plot(lams0, drop_q, label="Drop prob (queue-aware)")
    plt.xlabel(r"Flow-0 offered rate $\lambda_0$")
    plt.ylabel("Drop probability")
    plt.title("Network Drop Probability vs Load")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("fig_net_drop_vs_load.pdf")

def plot_bottleneck_occupancy_vs_load():
    import matplotlib.pyplot as plt
    from collections import defaultdict
    from queue_network_sim import Sim, Link, Flow, NetSim, StaticShortestHop, QueueAware

    def run_net(policy: str, lam0: float, seed: int):
        sim = Sim(seed=seed)
        nodes = [0, 1, 2, 4]
        cap = 25
        links = [
            Link(sim, 0, 1, mu=35.0, cap=cap, service_kind="exp", name="0->1"),
            Link(sim, 0, 2, mu=35.0, cap=cap, service_kind="exp", name="0->2"),
            Link(sim, 1, 4, mu=8.0,  cap=cap, service_kind="exp", name="1->4"),  # bottleneck
            Link(sim, 2, 4, mu=16.0, cap=cap, service_kind="exp", name="2->4"),
        ]
        out_links = defaultdict(list)
        for lk in links:
            out_links[lk.u].append(lk)

        destinations = [4]
        if policy == "static":
            router = StaticShortestHop(nodes, out_links, destinations)
            epoch = None
        else:
            router = QueueAware(nodes, out_links, destinations, epoch=0.1, beta=8.0)
            epoch = 0.1

        flows = [
            Flow(flow_id=0, src=0, dst=4, lam=lam0, arrival_kind="poisson"),
            Flow(flow_id=1, src=1, dst=4, lam=6.0,  arrival_kind="poisson"),
        ]
        net = NetSim(sim, nodes, links, flows, router, t_end=3000, warmup=300, epoch=epoch)
        res = net.run()
        return res["links"]["1->4"]["avg_n"]  # average queue occupancy at bottleneck

    lams0 = [8, 12, 16, 18, 20, 22, 24]
    seeds = [1, 2, 3]

    def avg(policy):
        ys = []
        for lam0 in lams0:
            vals = [run_net(policy, lam0, s) for s in seeds]
            ys.append(sum(vals)/len(vals))
        return ys

    y_static = avg("static")
    y_queue  = avg("queue")

    plt.figure()
    plt.plot(lams0, y_static, label="Static shortest-hop")
    plt.plot(lams0, y_queue, label="Queue-aware")
    plt.xlabel(r"Flow-0 offered rate $\lambda_0$")
    plt.ylabel(r"Avg. occupancy on link $1\to 4$ (packets)")
    plt.title("Bottleneck Link Queue Occupancy vs Load")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("fig_net_bottleneck_avgN_vs_load.pdf")

def plot_per_flow_throughput_vs_load():
    import matplotlib.pyplot as plt
    from collections import defaultdict
    from queue_network_sim import Sim, Link, Flow, NetSim, StaticShortestHop, QueueAware

    def run_net(policy: str, lam0: float, seed: int):
        sim = Sim(seed=seed)
        nodes = [0, 1, 2, 4]
        cap = 25
        links = [
            Link(sim, 0, 1, mu=35.0, cap=cap, service_kind="exp", name="0->1"),
            Link(sim, 0, 2, mu=35.0, cap=cap, service_kind="exp", name="0->2"),
            Link(sim, 1, 4, mu=8.0,  cap=cap, service_kind="exp", name="1->4"),
            Link(sim, 2, 4, mu=16.0, cap=cap, service_kind="exp", name="2->4"),
        ]
        out_links = defaultdict(list)
        for lk in links:
            out_links[lk.u].append(lk)

        destinations = [4]
        if policy == "static":
            router = StaticShortestHop(nodes, out_links, destinations)
            epoch = None
        else:
            router = QueueAware(nodes, out_links, destinations, epoch=0.1, beta=8.0)
            epoch = 0.1

        flows = [
            Flow(flow_id=0, src=0, dst=4, lam=lam0, arrival_kind="poisson"),
            Flow(flow_id=1, src=1, dst=4, lam=6.0,  arrival_kind="poisson"),
        ]
        net = NetSim(sim, nodes, links, flows, router, t_end=3000, warmup=300, epoch=epoch)
        res = net.run()
        f0 = res["flows"][0]["throughput"]
        f1 = res["flows"][1]["throughput"]
        return f0, f1

    lams0 = [8, 12, 16, 18, 20, 22, 24]
    seeds = [1, 2, 3]

    def avg(policy):
        f0s, f1s = [], []
        for lam0 in lams0:
            vals = [run_net(policy, lam0, s) for s in seeds]
            f0s.append(sum(v[0] for v in vals)/len(vals))
            f1s.append(sum(v[1] for v in vals)/len(vals))
        return f0s, f1s

    f0_static, f1_static = avg("static")
    f0_queue,  f1_queue  = avg("queue")

    plt.figure()
    plt.plot(lams0, f0_static, label="Flow0 thr (static)")
    plt.plot(lams0, f0_queue,  label="Flow0 thr (queue-aware)")
    plt.plot(lams0, f1_static, label="Flow1 thr (static)")
    plt.plot(lams0, f1_queue,  label="Flow1 thr (queue-aware)")
    plt.xlabel(r"Flow-0 offered rate $\lambda_0$")
    plt.ylabel("Throughput (pkt/s)")
    plt.title("Per-flow Throughput vs Load")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("fig_net_perflow_throughput_vs_load.pdf")



def main():
    plot_mm1_delay_vs_rho()
    plot_mm1n_drop_vs_lambda()
    plot_network_metrics_vs_load()
    plot_bottleneck_occupancy_vs_load()
    plot_per_flow_throughput_vs_load()

    print("Saved PDFs:")
    print("  fig_mm1_delay_vs_rho.pdf")
    print("  fig_mm1n_drop_vs_lambda.pdf")
    print("  fig_net_throughput_vs_load.pdf")
    print("  fig_net_delay_vs_load.pdf")
    print("  fig_net_drop_vs_load.pdf")
    print("  bottleneck, throughput")

if __name__ == "__main__":
    main()

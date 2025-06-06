import math
from collections import defaultdict
import numpy as np
import pandas as pd
from tqdm import tqdm
from itertools import repeat
import heapq

# ---------- Agent definition ----------
class Agent:
    def __init__(self, agent_id, agent_type, baseline_emissions, eta, gamma):
        self.id = agent_id
        self.type = agent_type  # 'individual', 'company', or 'country'
        self.baseline_emissions = baseline_emissions
        self.eta = eta
        self.gamma = gamma
        self.credits_received = 0
        self.net_cost = 0


    def wtp(self, x, w=1):
        y = self.baseline_emissions
        x = np.asarray(x, dtype=np.float64)
        return self.eta * (1 - np.power(x / y, self.gamma))

    def marginal_bids(self, resolution=100, w=1):
        y = self.baseline_emissions
        # vector of upper bounds: res, 2·res, …, y
        xs  = np.arange(resolution, y + 1, resolution, dtype=np.float64)
        x1  = xs - resolution
        price = self.wtp(x1, w=w) - self.wtp(xs, w=w)
        keep = price > 1e-6
        # zip NumPy array with a repeated reference to self
        return list(zip(price[keep], repeat(self), repeat(resolution, keep.sum())))


# ---------- helper used by several auctions ----------
def _allocate_from_sorted(bids_sorted, total_credits):
    winners = []
    idx = 0
    while total_credits > 0 and idx < len(bids_sorted):
        price, ag, q = bids_sorted[idx]
        take = q if q <= total_credits else total_credits
        winners.append((price, ag, take))
        total_credits -= take
        idx += 1
    # everything that was never touched is unused
    return winners, bids_sorted[idx:]

# ---------- uniform‑price sealed bid ----------
def _uniform_price(agents, total_credits, w=1):

    bids = [b for a in agents for b in a.marginal_bids(w=w)]
    bids.sort(reverse=True, key=lambda x: x[0])

    winners, unused = _allocate_from_sorted(bids, total_credits)
    clearing_price = winners[-1][0] if winners else 0.0

    for price, ag, q in winners:
        ag.credits_received += q
    for ag in agents:
        ag.net_cost += ag.credits_received * clearing_price

    return unused

# ---------- ascending clock auction ----------
def _ascending_clock(agents, total_credits, start_price=0.01, step=1.0, w=1):


    # Collect and sort bids
    bids = [b for a in agents for b in a.marginal_bids(w=w)]
    bids.sort(reverse=True, key=lambda x: x[0])


    # Extract sorted price list
    prices = [p for p, _, _ in bids]  # descending

    # Binary search for price that clears cap
    lo, hi = 0, len(prices) - 1
    clearing_idx = len(prices)

    while lo <= hi:
        mid = (lo + hi) // 2
        demand = mid + 1  # number of bids ≥ price[mid]
        if demand <= total_credits:
            clearing_idx = mid
            hi = mid - 1
        else:
            lo = mid + 1

    clearing_price = prices[clearing_idx] if clearing_idx < len(prices) else 0.0


    # Select winners
    winners, unused = _allocate_from_sorted(bids[:clearing_idx + 1], total_credits)


    for _, ag, q in winners:
        ag.credits_received += q
    for ag in agents:
        ag.net_cost += ag.credits_received * clearing_price


    return unused


# ---------- VCG pivotal mechanism ----------
def _vcg(agents, total_credits, w=1):
    print("DEBUG: Running VCG auction...")
    all_bids = [b for a in agents for b in a.marginal_bids(w=w)]
    all_bids.sort(reverse=True, key=lambda x: x[0])
    winners, unused = _allocate_from_sorted(all_bids, total_credits)

    for price, ag, q in winners:
        ag.credits_received += q

    value_others_full = {}
    total_value_full = {}
    for ag in agents:
        value_others_full[ag] = sum(p * q for p, a, q in winners if a is not ag)
    for ag in agents:
        total_value_full[ag] = sum(p * q for p, a, q in winners if a is ag)

    for pivot in agents:
        bids_without = [b for b in all_bids if b[1] is not pivot]
        bids_without.sort(reverse=True, key=lambda x: x[0])
        winners_without, _ = _allocate_from_sorted(bids_without, total_credits)
        value_without = sum(p * q for p, _, q in winners_without)

        payment = value_without - value_others_full[pivot]
        payment = max(payment, 0.0)
        if pivot.credits_received > 0:
            pivot.net_cost += payment

    return unused

# ---------- master wrapper ----------
def run_primary_auction(agents, total_credits, auction_type="uniform_price", w=1):

    if auction_type == "uniform_price":
        return _uniform_price(agents, total_credits, w=w)
    if auction_type == "clock":
        return _ascending_clock(agents, total_credits, w=w)
    if auction_type == "vcg":
        return _vcg(agents, total_credits, w=w)
    raise ValueError("auction_type must be 'uniform_price', 'clock', or 'vcg'")

# ---------- fairness and utility ----------
def compute_group_fairness_penalty(agents):

    groups = defaultdict(list)
    for ag in agents:
        groups[ag.type].append(ag)

    penalty = 0.0
    for members in groups.values():
        total = sum(a.credits_received for a in members)
        if total == 0:
            continue
        shares = [a.credits_received / total for a in members]
        entropy = -sum(s * math.log(s + 1e-12) for s in shares)
        penalty += 1 - entropy / math.log(len(shares))

    return penalty

def compute_global_utility(agents, alpha=0.05):

    cost = sum(a.net_cost for a in agents)
    penalty = compute_group_fairness_penalty(agents)
    util = -cost - alpha * cost * penalty

    return util, cost, penalty

def compute_value_of_free_credits(unused_bids, k):
    if k == 0 or not unused_bids:
        return 0.0
    top_prices = heapq.nlargest(k, (p for p, _, _ in unused_bids))
    return float(sum(top_prices))


# Agent.marginal_bids = marginal_bids
_uniform_price.__globals__['_allocate_from_sorted'] = _allocate_from_sorted
_ascending_clock.__globals__['_allocate_from_sorted'] = _allocate_from_sorted
_vcg.__globals__['_allocate_from_sorted'] = _allocate_from_sorted

# ---------- example scenario ----------
if __name__ == "__main__":

    df = pd.read_csv("sampled_agents/combined.csv").iloc[:2]

    agents = []
    for _, row in df.iterrows():
        name = row["name"]
        agent_type = row["type"]
        y = int(row["y"])
        revenue = float(row["revenue"])
        g = float(row["g"])

        if agent_type == "individual": #to make the data easier, let's group people into batches of 100. The rest of the comparisons (equality) will stay the same
            y *= 1000
            revenue *= 1000

        agent = Agent(name, agent_type, y, revenue, g)
        agents.append(agent)



    auction_types = ['clock', "vcg", 'uniform_price']
    # auction_types = []
    k = 3  # decay parameter for wtp as we get further from baseline

    for type in auction_types:

        for ag in agents:
            ag.credits_received = 0
            ag.net_cost = 0

        caps = [10000, 9000, 8000, 7000, 6000, 5000]
        giveaway_proportions = {}
        total_util = 0
        total_cost = 0
        total_free_val = 0
        total_fair = 0

        for i in tqdm(range(len(caps))):
            cap = caps[i]

            giveaways = {k: int(v * cap) for k, v in giveaway_proportions.items()}

            for ag in agents:
                if ag.id in giveaways:
                    g = giveaways[ag.id]
                    ag.credits_received += g
                    cap -= g

            weight = np.exp(-i / k)
            leftovers = run_primary_auction(agents, cap, auction_type=type, w=weight)

            free_val = compute_value_of_free_credits(leftovers, sum(giveaways.values()))
            true_cost = sum(a.net_cost for a in agents) + free_val
            util, cost, fair = compute_global_utility(agents)


            total_util += util
            total_cost += cost + free_val
            total_free_val += free_val
            total_fair += fair

        print("\n--- Totals --- for type: ", type)
        print(f"Total utility: {total_util:,.0f}")
        print(f"Total true economic cost: {total_cost:,.0f}")
        print(f"Total value of free credits: {total_free_val:,.0f}")
        print(f"Total fairness penalty (sum): {total_fair:.3f}")

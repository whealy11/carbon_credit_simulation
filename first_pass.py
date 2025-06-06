# =============================================================
#  Cap‑and‑Trade Auction Simulator
#  - Primary goal: minimise abatement cost (unused marginal bids)
#  - New metric:  division_score  =  Σ utilities − β·std(utilities)
#                 (utility = value of credits − amount paid)
# =============================================================

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
        self.id   = agent_id
        self.type = agent_type            # 'individual', 'company', or 'country'
        self.baseline_emissions = baseline_emissions
        self.eta   = eta
        self.gamma = gamma
        self.reset()                      # <<< CHANGED >>>

    def reset(self):                      # <<< CHANGED >>>
        self.credits_received = 0
        self.net_cost         = 0.0       # money paid
        self.value_gained     = 0.0       # Σ (marginal value) of credits won

    def wtp(self, x, w=1):
        y = self.baseline_emissions
        return self.eta * (1 - (x / y) ** self.gamma)

    def marginal_bids(self, resolution=100, w=1):
        y  = self.baseline_emissions
        xs = np.arange(resolution, y + 1, resolution, dtype=np.float64)
        x1 = xs - resolution
        price = self.wtp(x1, w=w) - self.wtp(xs, w=w)
        keep  = price > 1e-6
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
    return winners, bids_sorted[idx:]

# ---------- uniform‑price sealed bid ----------
def _uniform_price(agents, total_credits, w=1):
    bids = [b for a in agents for b in a.marginal_bids(w=w)]
    bids.sort(reverse=True, key=lambda x: x[0])

    winners, unused = _allocate_from_sorted(bids, total_credits)
    clearing_price  = winners[-1][0] if winners else 0.0

    # credits + value
    for price, ag, q in winners:
        ag.credits_received += q
        ag.value_gained    += price * q      # utility side

    # payments (uniform clearing price)
    for ag in agents:
        if ag.credits_received:
            ag.net_cost += ag.credits_received * clearing_price

    return winners, unused                  # <<< CHANGED >>>

# ---------- ascending clock (pay‑as‑bid) ----------
def _ascending_clock(agents, total_credits, w=1):
    
    bids = [b for a in agents for b in a.marginal_bids(w=w)]


    if not bids:
        return [], []

    bids.sort(reverse=True, key=lambda x: x[0])

    cum_qty, cutoff_idx, clearing_price = 0, None, 0.0
    for idx, (price, _, q) in enumerate(bids):
        cum_qty += q
        if cum_qty >= total_credits:
            cutoff_idx = idx
            clearing_price = price
            break

    if cutoff_idx is None:
        cutoff_idx = len(bids) - 1
        clearing_price = bids[-1][0]

    winners, unused = _allocate_from_sorted(bids[:cutoff_idx + 1], total_credits)

    total_alloc = 0
    for price, ag, q in winners:
        start = ag.credits_received
        ag.credits_received += q
        ag.net_cost += price * q
        total_alloc += q

        # Value gained from each tranche
        xs = np.arange(start + 1, start + q + 1, dtype=np.float64)
        marginal_values = ag.wtp(xs - 1) - ag.wtp(xs)
        total_value = np.sum(marginal_values)
        ag.value_gained += total_value

        print(f"[DEBUG clock] Agent {ag.id}: got {q}, paid {price*q:,.2f}, value {total_value:,.2f}")

    print(f"[DEBUG clock] Total allocated = {total_alloc}, clearing price ≈ {clearing_price:,.4f}")
    print(f"[DEBUG clock] Remaining bids: {len(unused)}")
    return winners, unused

# ---------- VCG pivotal mechanism ----------
def _vcg(agents, total_credits, w=1):
    all_bids = [b for a in agents for b in a.marginal_bids(w=w)]
    all_bids.sort(reverse=True, key=lambda x: x[0])
    winners, unused = _allocate_from_sorted(all_bids, total_credits)

    for price, ag, q in winners:
        ag.credits_received += q
        ag.value_gained    += price * q          # <<< CHANGED >>>

    value_others_full = {}
    for ag in agents:
        value_others_full[ag] = sum(p * q for p, a, q in winners if a is not ag)

    for pivot in agents:
        bids_without = [b for b in all_bids if b[1] is not pivot]
        bids_without.sort(reverse=True, key=lambda x: x[0])
        winners_without, _ = _allocate_from_sorted(bids_without, total_credits)
        value_without = sum(p * q for p, _, q in winners_without)

        payment = max(value_without - value_others_full[pivot], 0.0)
        if pivot.credits_received:
            pivot.net_cost += payment

    return winners, unused

# ---------- master wrapper ----------
def run_primary_auction(agents, total_credits, auction_type="uniform_price", w=1):
    if auction_type == "uniform_price":
        return _uniform_price(agents, total_credits, w=w)
    if auction_type == "clock":
        return _ascending_clock(agents, total_credits, w=w)
    if auction_type == "vcg":
        return _vcg(agents, total_credits, w=w)
    raise ValueError("auction_type must be 'uniform_price', 'clock', or 'vcg'")

# ----------  cost of the unavoidable abatement ----------
def abatement_cost(unused_bids):
    return sum(p * q for p, _, q in unused_bids)

# ----------  new division‑score metric  --------------------  <<< CHANGED >>>
def division_score(agents, beta=0.1):
    """Utility fairness score: total utilities minus β·std(utilities)."""
    utils = np.array([ag.value_gained - ag.net_cost for ag in agents], dtype=float)
    if len(utils) < 2:
        return float(utils.sum()), float(utils.sum()), 0.0
    return float(utils.sum() - beta * utils.std()), float(utils.sum()), float(utils.std())

# ---------- fairness penalty (still based on credits) -------
def compute_group_fairness_penalty(agents):
    groups = defaultdict(list)
    for ag in agents:
        groups[ag.type].append(ag)

    penalty = 0.0
    for members in groups.values():
        total = sum(a.credits_received for a in members)
        if total == 0:
            continue
        shares  = [a.credits_received / total for a in members]
        entropy = -sum(s * math.log(s + 1e-12) for s in shares)
        penalty += 1 - entropy / math.log(len(shares))
    return penalty

# -------------------  driver  -------------------------------
if __name__ == "__main__":
    print("Reading CSV and initialising agents...")
    df = pd.read_csv("sampled_agents/combined.csv").iloc[:6]

    agents = []
    for _, row in df.iterrows():
        name = row["name"]
        a_typ = row["type"]
        y     = int(row["y"])
        eta   = float(row["revenue"]) * 1e-10
        g     = float(row["g"])

        if a_typ == "individual":          # bundle 1k individuals into one
            y   *= 1000
            eta *= 1000

        agents.append(Agent(name, a_typ, y, eta, g))

    auction_types = ['clock', 'vcg', 'uniform_price']
    k = 3                                   # bid‑decay parameter

    for atype in auction_types:
        # reset cumulative tallies
        total_abate_cost = 0.0
        total_div_score  = 0.0
        total_util_sum   = 0.0
        total_util_std   = 0.0

        caps = [10000, 9000, 8000, 7000, 6000, 5000]

        for year_idx, cap in enumerate(tqdm(caps, desc=atype)):
            # clear per‑year agent state
            for ag in agents:
                ag.reset()                 # <<< CHANGED >>>

            weight = np.exp(-year_idx / k)

            winners, leftovers = run_primary_auction(
                agents, cap, auction_type=atype, w=weight)

            year_cost = abatement_cost(leftovers)
            score, util_sum, util_std = division_score(agents)

            total_abate_cost += year_cost
            total_div_score  += score
            total_util_sum   += util_sum
            total_util_std   += util_std

        print(f"\n=== {atype.upper()} ===")
        print(f"Total abatement cost : {total_abate_cost:,.0f}")
        print(f"Division score       : {total_div_score:,.0f}")
        print(f"(Σ utilities = {total_util_sum:,.0f},  "
              f"Σ stds = {total_util_std:,.1f})")

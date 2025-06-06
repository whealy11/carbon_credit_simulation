import math
from collections import defaultdict
import numpy as np
import pandas as pd
from tqdm import tqdm
from itertools import repeat
import heapq
import matplotlib.pyplot as plt

# Agent definition 
class Agent:
    def __init__(self, agent_id, agent_type, baseline_emissions, eta, gamma):
        self.id   = agent_id
        self.type = agent_type            # person country or company
        self.initial_baseline = baseline_emissions
        self.current_y        = baseline_emissions   
        self.eta   = eta
        self.gamma = gamma
        self.credits_received = 0
        self.net_cost = 0

    #  dynamic‑y helpers 
    def prepare_next_year(self):
        """
        After the auction finishes, carry forward this year’s allocation
        so that next year’s y equals the credits just received.
        """

        self.current_y = max(float(self.credits_received), 1.0)
        self.credits_received = 0
        self.net_cost = 0

    def reset_type_run(self):
        """Reset to starting conditions at the beginning of a new auction type."""
        self.current_y = self.initial_baseline
        self.credits_received = 0
        self.net_cost = 0

    # WTP and bids 
    def wtp(self, x, w: float = 1):
        y = self.current_y
        return w * self.eta * y * (1 - (x / y) ** self.gamma)

    def lost_utility(self) -> float:
        """
        Utility lost by being capped at k = credits_received.

        ∫_k^y η(1 − (x/y)^γ) dx
        Analytical form:
            η * [ γy/(γ+1) − k + k^{γ+1} / ((γ+1) y^γ) ]
        """
        k, y, n, γ = self.credits_received, self.current_y, self.eta, self.gamma
        if k >= y:
            return 0.0
        return n * (γ * y / (γ + 1) - k + (k ** (γ + 1)) / ((γ + 1) * (y ** γ)))

    def marginal_bids(
        self,
        base_resolution: int = 1_000,
        w: float = 1,
        growth_factor: float = 2.0,
    ):
        base_resolution = 10
        growth_factor = 1
        if growth_factor < 1:
            raise ValueError("growth_factor must be at least 1")

        y = self.current_y                         
        bids, x_prev, lot = [], 0, base_resolution

        while x_prev < y:
            x_next = min(x_prev + lot, y)
            price  = self.wtp(x_prev, w=w) - self.wtp(x_next, w=w)
            if price > 1e-6:
                bids.append((price, self, x_next - x_prev))
            x_prev = x_next
            lot = min(int(lot * growth_factor), max(1, y - x_prev))

        return bids


# plotting helper 
def plot_all_wtp(agents, points: int = 200, cols: int = 4):
    n    = len(agents)
    rows = (n + cols - 1) // cols
    fig, ax = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 2), squeeze=False)

    for idx, ag in enumerate(agents):
        r, c = divmod(idx, cols)
        axis = ax[r][c]
        xs   = np.linspace(0, ag.current_y, points)
        axis.plot(xs, ag.wtp(xs))
        axis.set_title(f"{ag.id} ({ag.type})", fontsize=8)
        axis.set_xlabel("credits, x")
        axis.set_ylabel("total WTP")
        axis.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
        axis.grid(True, linewidth=0.3)

    for extra in range(n, rows * cols):
        fig.delaxes(ax[extra // cols][extra % cols])

    fig.tight_layout()
    plt.show()


# helper used by several auctions 
def _allocate_from_sorted(bids_sorted, total_credits):
    winners, idx = [], 0
    while total_credits > 0 and idx < len(bids_sorted):
        price, ag, q = bids_sorted[idx]
        take = q if q <= total_credits else total_credits
        winners.append((price, ag, take))
        total_credits -= take
        idx += 1
    return winners, bids_sorted[idx:]


# uniform‑price sealed bid 
def _uniform_price(agents, total_credits, w=1):
    bids = [b for a in agents for b in a.marginal_bids(w=w)]
    bids.sort(reverse=True, key=lambda x: x[0])

    winners, unused = _allocate_from_sorted(bids, total_credits)
    clearing_price = winners[-1][0] if winners else 0.0

    for _, ag, q in winners:
        ag.credits_received += q
    for ag in agents:
        ag.net_cost += ag.credits_received * clearing_price
    return unused


# ascending clock auction 
def _ascending_clock(agents, total_credits, w=1):
    bids = [b for a in agents for b in a.marginal_bids(w=w)]
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
        clearing_price = bids[-1][0] if bids else 0.0

    winners, unused = _allocate_from_sorted(bids[:cutoff_idx + 1], total_credits)

    for _, ag, q in winners:
        ag.credits_received += q
    for ag in agents:
        ag.net_cost += ag.credits_received * clearing_price
    return unused


# VCG pivotal mechanism 
def _vcg(agents, total_credits, w=1):
    all_bids = [b for a in agents for b in a.marginal_bids(w=w)]
    all_bids.sort(reverse=True, key=lambda x: x[0])
    winners, unused = _allocate_from_sorted(all_bids, total_credits)

    for _, ag, q in winners:
        ag.credits_received += q

    value_others_full = {ag: sum(p * q for p, a, q in winners if a is not ag)
                         for ag in agents}

    for pivot in agents:
        bids_without = [b for b in all_bids if b[1] is not pivot]
        bids_without.sort(reverse=True, key=lambda x: x[0])
        winners_without, _ = _allocate_from_sorted(bids_without, total_credits)
        value_without = sum(p * q for p, _, q in winners_without)
        payment = max(value_without - value_others_full[pivot], 0.0)
        if pivot.credits_received > 0:
            pivot.net_cost += payment

    return unused


# master wrapper 
def run_primary_auction(agents, total_credits, auction_type="uniform_price", w=1):

    if auction_type == "uniform_price":
        return _uniform_price(agents, total_credits, w=w)
    if auction_type == "clock":
        return _ascending_clock(agents, total_credits, w=w)
    if auction_type == "vcg":
        return _vcg(agents, total_credits, w=w)
    raise ValueError("auction_type must be 'uniform_price', 'clock', or 'vcg'")


#  fairness and utility 
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


def compute_global_utility(agents, alpha: float = 0.05):
    cost = sum(a.lost_utility() for a in agents)      
    penalty = compute_group_fairness_penalty(agents)
    util = -cost - alpha * cost * penalty
    return util, cost, penalty

def get_initial_allocation(agents):

    persons = [ag for ag in agents if ag.type == "person"]
    num_persons = len(persons)

    # Amount each person receives
    proportions = {}
    for elem in persons:
        proportions[elem.id] = 0/ num_persons
    return proportions
# ---------- example scenario ----------
if __name__ == "__main__":
    df = pd.read_csv("sampled_agents/combined.csv").iloc[0:113]
    x_1, x_2, x_3 = 0.01, 0.69, 0.30  

    # Group by type and sum y
    #total_y = df['y'].sum()
    total_y = 37410
    current_y = df.groupby('type')['y'].sum().to_dict()

    # Target total y for each type
    target_y = {
        'person': total_y * x_1,
        'company': total_y * x_2,
        'country': total_y * x_3,
    }

    # Scale factor for each type
    scaling_factors = {
        t: target_y[t] / current_y[t]
        for t in current_y if current_y[t] > 0
    }

    # Apply scaling
    df[['y', 'revenue']] = df.apply(
        lambda row: [row['y'] * scaling_factors.get(row['type'], 1.0),
                    row['revenue'] * scaling_factors.get(row['type'], 1.0)],
        axis=1, result_type='expand'
    )
    agents = []
    total_y = 0
    people_y = 0
    company_y = 0
    country_y = 0
    for _, row in df.iterrows():
        name = row["name"]
        agent_type = row["type"]
        y = int(row["y"])
        revenue = float(row["revenue"])
        g = float(row["g"])


        total_y += y
        if agent_type == 'person':
            people_y += y
        if agent_type == 'country':
            country_y += y
        if agent_type == 'company':
            company_y += y
        agents.append(Agent(name, agent_type, y, 1, g))
    print("TOTAL Y: ", total_y)
    print("company Y: ", company_y / total_y)
    print("country Y: ", country_y / total_y)
    print("person Y: ", people_y / total_y)



    auction_types = ["clock", "vcg", "uniform_price"]
    auction_types = ['vcg']
    decay_k = 3
    base = total_y
    decay = .905 
    caps = [base * (1 - 0.45 * i / 6) for i in range(1, 7)] #linear
    caps = [decay * base, decay ** 2 * base, decay ** 3 * base, decay ** 4 * base, decay ** 5 * base, decay ** 6 * base] #exponential decay
    caps = [base, base, base, base, base, .55 * base] #cliff

    for a_type in auction_types:


        for ag in agents:
            ag.reset_type_run()                       

        total_util = total_cost = total_fair = 0.0
        giveaway_proportions = get_initial_allocation(agents)

        for i, cap in enumerate(tqdm(caps, desc=f"{a_type} years")):

            giveaways = {k: int(v * cap) for k, v in giveaway_proportions.items()}

            for ag in agents:
                if ag.id in giveaways:
                    g = giveaways[ag.id]
                    ag.credits_received += g
                    cap -= g
            weight = math.exp(-i / decay_k)
            run_primary_auction(agents, cap, auction_type=a_type, w=weight)

            util, cost, fair = compute_global_utility(agents)
            total_util += util
            total_cost += cost
            total_fair += fair

            # prepare y for next year
            # if i == 0:
            #     for ag in agents:
            #         print(ag.id)
            #         print(ag.credits_received)
            for ag in agents:
                ag.prepare_next_year()               #


        print("\n--- Totals --- for type:", a_type)
        print(f"Total utility: {total_util:,.0f}")
        print(f"Total economic cost (lost utility): {total_cost:,.0f}")
        print(f"Total fairness penalty (sum): {total_fair:.3f}")

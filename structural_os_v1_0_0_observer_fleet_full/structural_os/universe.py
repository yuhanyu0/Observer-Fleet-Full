from __future__ import annotations

CORE_GROUPS = {
    "Equity_US": ["SPY", "QQQ", "IWM"],
    "Rates": ["TLT", "IEF"],
    "Credit": ["LQD", "HYG"],
    "USD": ["UUP"],
    "Commodities": ["DBC", "GLD", "SLV"],
    "MegaCap_AI": ["MSFT", "NVDA", "AMZN"],
    "Semis": ["SMH", "TSM", "ASML"],
}

FULL_GROUPS = {
    **CORE_GROUPS,
    "DataCenter_Infra": ["SRVR", "EQIX", "DLR"],
    "Defense": ["ITA"],
    "Energy": ["XLE"],
    "Utilities": ["XLU"],
    "Nuclear_Uranium": ["URA", "CCJ"],
    "Critical_Minerals": ["LIT", "COPX"],
    "Quantum": ["QTUM", "IONQ", "RGTI"],
    "Crypto": ["BTC-USD", "ETH-USD", "COIN", "MSTR"],
    "Cyber": ["CIBR"],
}

CONSTRAINT_MAP = {
    "Equity_US": "Demand_Risk",
    "Rates": "Policy_Rates",
    "Credit": "Funding_Stress",
    "USD": "Dollar_Liquidity",
    "Commodities": "Physical_Scarcity",
    "MegaCap_AI": "Compute_Scale",
    "Semis": "Compute_Scale",
    "DataCenter_Infra": "Compute_Scale",
    "Defense": "Geopolitical",
    "Energy": "Physical_Scarcity",
    "Utilities": "Grid_Infra",
    "Nuclear_Uranium": "Energy_Transition",
    "Critical_Minerals": "Physical_Scarcity",
    "Quantum": "Frontier_Speculation",
    "Crypto": "Speculative_Liquidity",
    "Cyber": "Security",
}

def groups_for_universe(universe_id: str):
    if universe_id == "core":
        return CORE_GROUPS
    if universe_id == "full":
        return FULL_GROUPS
    raise KeyError(f"Unknown universe_id: {universe_id}")

def flatten_groups(groups: dict[str, list[str]]) -> list[str]:
    seen, out = set(), []
    for g, xs in groups.items():
        for x in xs:
            if x not in seen:
                out.append(x)
                seen.add(x)
    return out

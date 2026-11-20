"""共享测试夹具：第 2.3 节实例与两个预算计数案例。"""

from __future__ import annotations

from symplan.env import Rule, World


def example_world() -> World:
    """第 2.3 节实例。"""
    return World(
        symbols=("A", "B", "C", "D", "G"),
        initial=(1, 1, 1, 0, 0),
        goal=(0, 0, 0, 0, 1),
        rules=(
            Rule("r0", (1, 1, 0, 0, 0), (0, 0, 0, 1, 0)),
            Rule("r1", (0, 0, 1, 1, 0), (0, 0, 0, 0, 1)),
        ),
        world_id="w_example",
        group_id="g_example",
        meta={"structure": "chain", "witness_plan": ["r0", "r1"], "witness_steps": 2},
    )


FORWARD_TRACE = [
    "EXPAND n0 STATE 1 1 1 0 0",
    "NEW n1 VIA r0 STATE 0 0 1 1 0",
    "CLOSE n0",
    "EXPAND n1 STATE 0 0 1 1 0",
    "NEW n2 VIA r1 STATE 0 0 0 0 1",
    "CLOSE n1",
    "EXPAND n2 STATE 0 0 0 0 1",
    "GOAL n2",
    "PLAN r0 r1",
    "END",
]

BACKWARD_TRACE = [
    "EXPAND n0 NEED 0 0 0 0 1",
    "NEW n1 VIA r0 NEED 1 1 0 0 1",
    "NEW n2 VIA r1 NEED 0 0 1 1 0",
    "CLOSE n0",
    "EXPAND n1 NEED 1 1 0 0 1",
    "CLOSE n1",
    "EXPAND n2 NEED 0 0 1 1 0",
    "NEW n3 VIA r0 NEED 1 1 1 0 0",
    "CLOSE n2",
    "EXPAND n3 NEED 1 1 1 0 0",
    "GOAL n3",
    "PLAN r0 r1",
    "END",
]


def deadend_world(chain: int = 100) -> World:
    """第 7 节案例一：长死路加直接目标，BFS=3、DFS=chain+2。"""
    symbols = ["S", "G"] + [f"X{i}" for i in range(chain)]
    index = {s: i for i, s in enumerate(symbols)}

    def vec(**kwargs) -> tuple[int, ...]:
        out = [0] * len(symbols)
        for key, count in kwargs.items():
            out[index[key]] = count
        return tuple(out)

    rules = [
        Rule("r0", vec(S=1), vec(X0=1)),
        Rule("r1", vec(S=1), vec(G=1)),
    ]
    for i in range(chain - 1):
        rules.append(Rule(f"r{i + 2}", vec(**{f"X{i}": 1}), vec(**{f"X{i + 1}": 1})))
    return World(
        symbols=tuple(symbols),
        initial=vec(S=1),
        goal=vec(G=1),
        rules=tuple(rules),
        world_id="w_deadend",
        group_id="g_deadend",
        meta={"structure": "trap"},
    )


def binary_tree_world(depth: int = 5) -> World:
    """第 7 节案例二：完整二叉树，首条深入路径末端为唯一目标，DFS=depth+1、BFS=2^depth。"""
    symbols = [f"N{d}_{k}" for d in range(depth + 1) for k in range(2**d)]
    index = {s: i for i, s in enumerate(symbols)}

    def vec(**kwargs) -> tuple[int, ...]:
        out = [0] * len(symbols)
        for key, count in kwargs.items():
            out[index[key]] = count
        return tuple(out)

    rules = []
    counter = 0
    for d in range(depth):
        for k in range(2**d):
            for side in (0, 1):
                rules.append(
                    Rule(
                        f"r{counter}",
                        vec(**{f"N{d}_{k}": 1}),
                        vec(**{f"N{d + 1}_{2 * k + side}": 1}),
                    )
                )
                counter += 1
    return World(
        symbols=tuple(symbols),
        initial=vec(N0_0=1),
        goal=vec(**{f"N{depth}_0": 1}),
        rules=tuple(rules),
        world_id="w_tree",
        group_id="g_tree",
        meta={"structure": "trap"},
    )


def cycle_world() -> World:
    """含循环的小世界：A->B->C->A，另有一条到目标的支路。"""
    symbols = ("A", "B", "C", "G")
    return World(
        symbols=symbols,
        initial=(1, 0, 0, 0),
        goal=(0, 0, 0, 1),
        rules=(
            Rule("r0", (1, 0, 0, 0), (0, 1, 0, 0)),
            Rule("r1", (0, 1, 0, 0), (0, 0, 1, 0)),
            Rule("r2", (0, 0, 1, 0), (1, 0, 0, 0)),
            Rule("r3", (0, 0, 1, 0), (0, 0, 0, 1)),
        ),
        world_id="w_cycle",
        group_id="g_cycle",
        meta={"structure": "trap"},
    )

"""Attribute-aware evaluation metrics for bubble diagram generation.

Standard MMD on graph statistics (degree, clustering, spectral) ignores
node attributes, so it cannot detect failure modes like "Living bloat"
or "Bedroom always adjacent to Bedroom." These helpers compute KL-style
metrics over attr-conditioned distributions:

* :func:`endpoint_attr_pair_counts` — for each edge, count the unordered
  ``(attr_u, attr_v)`` pair across all graphs. Tells us which attribute
  combinations tend to be connected.
* :func:`kl_divergence` — KL(p || q) over sparse count dicts with
  additive smoothing so disjoint supports don't blow up to infinity.
* :func:`living_count_per_graph` — histogram of "how many nodes of a
  given class are in each generated graph" (e.g. living rooms per
  floorplan).
* :func:`per_class_degree_stats` — mean/std/count of node degrees
  per attribute class. Tells us whether Living is a hub, Bathroom a
  leaf, etc.

All functions accept ``attrs_list`` as a list of per-graph attr
sequences (``List[List[int]]`` or ``List[None]``). A ``None`` entry or
a too-short list silently skips the offending nodes/edges so a few
pathological graphs never crash a whole evaluation run.
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict


def endpoint_attr_pair_counts(graphs, attrs_list, num_classes):
    """Count (attr_u, attr_v) unordered pairs across all edges.

    Returns ``dict[(int, int), int]`` where the key is ``(min(a_u, a_v),
    max(a_u, a_v))`` so (Living, Kitchen) and (Kitchen, Living) collapse
    into the same bin.
    """
    del num_classes  # unused; kept for API symmetry
    counts = defaultdict(int)
    for idx, G in enumerate(graphs):
        if idx >= len(attrs_list):
            continue
        node_attrs = attrs_list[idx]
        if node_attrs is None:
            continue
        n_attrs = len(node_attrs)
        for u, v in G.edges():
            if u >= n_attrs or v >= n_attrs:
                continue
            a_u = int(node_attrs[u])
            a_v = int(node_attrs[v])
            pair = (min(a_u, a_v), max(a_u, a_v))
            counts[pair] += 1
    return dict(counts)


def kl_divergence(p_counts, q_counts, eps=1e-8):
    """KL(p || q) over sparse count dicts with additive smoothing.

    If ``p_counts`` is empty returns 0 (no gen signal, nothing to test).
    Else both distributions are normalized to sum to 1, then every key
    present in either gets an ``eps`` added before the division to keep
    the result finite when supports are disjoint.
    """
    total_p = sum(p_counts.values())
    if total_p == 0:
        return 0.0
    total_q = sum(q_counts.values())
    all_keys = set(p_counts) | set(q_counts)
    kl = 0.0
    for k in all_keys:
        p = p_counts.get(k, 0) / total_p + eps
        q = q_counts.get(k, 0) / total_q + eps if total_q > 0 else eps
        kl += p * math.log(p / q)
    return kl


def living_count_per_graph(graphs, attrs_list, target_class=0):
    """For each graph, count nodes with ``attr == target_class``.

    Returns ``Counter`` mapping count-value -> number of graphs with that
    count (e.g. ``{0: 50, 1: 800, 2: 120, 3: 30}`` for living rooms).
    """
    hist = Counter()
    for idx, G in enumerate(graphs):
        del G  # unused; only count attrs
        if idx >= len(attrs_list):
            continue
        node_attrs = attrs_list[idx]
        if node_attrs is None:
            continue
        n_target = sum(1 for a in node_attrs if int(a) == target_class)
        hist[n_target] += 1
    return hist


def per_class_degree_stats(graphs, attrs_list, num_classes):
    """For each attr class, aggregate node degrees across graphs.

    Returns ``dict[int, dict]``::

        {
          0: {'mean': 3.2, 'std': 0.7, 'count': 842},
          1: {'mean': 1.9, 'std': 0.9, 'count': 2350},
          ...
          6: {'mean': None, 'std': None, 'count': 0},
        }

    ``mean`` / ``std`` are ``None`` when the class never appears (to
    avoid ``np.nan`` which doesn't JSON-serialize cleanly).
    """
    degrees_by_class = defaultdict(list)
    for idx, G in enumerate(graphs):
        if idx >= len(attrs_list):
            continue
        node_attrs = attrs_list[idx]
        if node_attrs is None:
            continue
        n_attrs = len(node_attrs)
        for u in G.nodes():
            if u >= n_attrs:
                continue
            cls = int(node_attrs[u])
            degrees_by_class[cls].append(G.degree(u))
    stats = {}
    for c in range(num_classes):
        degs = degrees_by_class.get(c, [])
        if degs:
            mean = sum(degs) / len(degs)
            var = sum((d - mean) ** 2 for d in degs) / len(degs)
            std = math.sqrt(var)
            stats[c] = {
                'mean': float(mean),
                'std': float(std),
                'count': len(degs),
            }
        else:
            stats[c] = {'mean': None, 'std': None, 'count': 0}
    return stats

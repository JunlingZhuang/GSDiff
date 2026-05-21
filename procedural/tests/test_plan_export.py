from procedural.plan_export import derive_walls


def _square(x0, y0, x1, y1):
    """Closed CCW polygon as list of (x, y), first point repeated at end."""
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)]


def test_derive_walls_two_adjacent_squares_dedup_shared_edge():
    # Two unit squares sharing the edge x=1, y in [0,1].
    rooms = {
        "rA": _square(0, 0, 1, 1),
        "rB": _square(1, 0, 2, 1),
    }
    walls = derive_walls(rooms, thickness=0.2)
    # 4 + 4 edges, the shared (1,0)-(1,1) edge deduped to one → 7 unique walls.
    assert len(walls) == 7
    # every wall has the requested thickness and non-zero length
    for w in walls:
        assert w["thickness"] == 0.2
        ax, ay = w["a"]
        bx, by = w["b"]
        assert (ax, ay) != (bx, by)
    # the shared segment appears exactly once
    def key(w):
        return tuple(sorted([tuple(w["a"]), tuple(w["b"])]))
    keys = [key(w) for w in walls]
    shared = tuple(sorted([(1.0, 0.0), (1.0, 1.0)]))
    assert keys.count(shared) == 1

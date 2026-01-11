# src/models/quantum/relational/freezing.py

def load_param_dict(path):
    """Load parameter file: one 'name: value' per line."""
    d = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or ":" not in line:
                continue
            k, v = line.split(":", 1)
            d[k.strip()] = float(v.strip())
    return d


def freeze_circuit_from_dict(circ, name_to_value):
    """
    Replace selected symbols in a lambeq circuit by constants,
    keeping all others trainable.
    """
    import torch
    from sympy import default_sort_key

    symbols = sorted(circ.free_symbols, key=default_sort_key)
    args = []
    for s in symbols:
        key = getattr(s, "name", str(s))
        if key in name_to_value:
            args.append(torch.tensor(name_to_value[key]))  # frozen
        else:
            args.append(s)  # trainable
    return circ.lambdify(*symbols)(*args)


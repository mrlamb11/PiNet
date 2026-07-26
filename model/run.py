from safetensors import safe_open
from collections import defaultdict

with safe_open("PiNet.safetensors", framework="numpy") as f:
    buf = f.get_tensor("buffer").tobytes()

    def unpack(name):
        sign, off, ln = f.get_tensor(name+".sign"), f.get_tensor(name+".offset"), f.get_tensor(name+".length")
        return [int(s) * int.from_bytes(buf[int(o):int(o)+int(l)], "big") for s, o, l in zip(sign, off, ln)]

    w1 = defaultdict(list)
    for r, c, w in zip(f.get_tensor("w1.row"), f.get_tensor("w1.col"), unpack("w1.val")):
        w1[r].append((c, w))
    w2 = defaultdict(list)
    for r, c, w in zip(f.get_tensor("w2.row"), f.get_tensor("w2.col"), unpack("w2.val")):
        w2[r].append((c, w))
    b1, b2 = unpack("b1"), f.get_tensor("b2")
    IDX = {k: int(f.get_tensor(f"idx.{k}")[0]) for k in ("init", "done", "kcnt", "Jreg", "PI")}

H, S = len(b1), len(b2)

def step(h):
    a = [max(0, b1[u] + sum(w * h[c] for c, w in w1[u])) for u in range(H)]
    return [int(b2[r]) + sum(w * a[c] for c, w in w2[r]) for r in range(S)]

def pi(digits):
    dp = digits + 12
    h = [0] * S
    h[IDX["init"]] = 1
    h[IDX["kcnt"]] = int(dp / 14.1816) + 2
    h[IDX["Jreg"]] = dp + 10
    while not h[IDX["done"]]:
        h = step(h)
    h = step(h)
    s = str(h[IDX["PI"]])
    return s[0] + "." + s[1:digits + 1]

print(pi(1000))

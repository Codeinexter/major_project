"""One simulated edge device. Each round: get global model -> train locally -> send update back."""
import os, time, math
import numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader
from common import *

CID    = int(os.environ.get("CLIENT_ID", 0))
N      = int(os.environ.get("NUM_CLIENTS", 3))
ROUNDS = int(os.environ.get("ROUNDS", 5))
EPOCHS = int(os.environ.get("LOCAL_EPOCHS", 1))
PART   = os.environ.get("PARTITION", "iid")          # iid | noniid
ALPHA  = float(os.environ.get("ALPHA", 0.5))         # Dirichlet; smaller = more skewed
LR     = float(os.environ.get("LR", 5e-4))
COSINE = int(os.environ.get("COSINE_LR", 1))         # 1 = cosine decay over the rounds (as in the centralized run)
PERSIST_OPT = int(os.environ.get("PERSIST_OPT", 1))  # 1 = keep AdamW state across rounds (client process is persistent)
torch.set_num_threads(int(os.environ.get("CPU_THREADS", 1)))
torch.manual_seed(SEED)

# ---- this client's shard (split by lesion so no lesion is shared between clients)
df  = load_frame("split_train.csv")
rng = np.random.default_rng(SEED)                    # same seed everywhere => clients agree on the partition
les = df.drop_duplicates(GROUP_COL)[[GROUP_COL, "y"]].reset_index(drop=True)
if PART == "iid":
    shards = np.array_split(rng.permutation(len(les)), N)
else:
    shards = [[] for _ in range(N)]
    for c in les["y"].unique():
        idx = rng.permutation(les.index[les["y"] == c].values)
        cuts = (np.cumsum(rng.dirichlet([ALPHA] * N))[:-1] * len(idx)).astype(int)
        for k, part in enumerate(np.split(idx, cuts)): shards[k] += part.tolist()
mine = df[df[GROUP_COL].isin(les.loc[shards[CID], GROUP_COL])].reset_index(drop=True)
counts = mine["y"].value_counts().reindex(range(len(CLASSES)), fill_value=0).values
print(f"[client {CID}] {len(mine)} images | class counts {counts.tolist()}", flush=True)
w = torch.tensor(counts.sum() / (len(CLASSES) * np.maximum(counts, 1)), dtype=torch.float32)
loss_fn = nn.CrossEntropyLoss(weight=w)
dl = DataLoader(DS(mine, TRAIN_TF), BS, shuffle=True, num_workers=0)
model = build_model()
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

for r in range(1, ROUNDS + 1):
    wait_for(SHARED / f"global_round{r}.pth")
    model.load_state_dict(torch.load(SHARED / f"global_round{r}.pth", map_location="cpu"))
    if not PERSIST_OPT:
        opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    # centralized: CosineAnnealingLR(T_max=EPOCHS), stepped once per epoch -> epoch e uses 0.5*LR*(1+cos(pi*(e-1)/T)).
    # here one round = one local epoch, so round r uses the same value.
    lr_r = LR * 0.5 * (1 + math.cos(math.pi * (r - 1) / ROUNDS)) if COSINE else LR
    for g in opt.param_groups: g["lr"] = lr_r
    t0 = time.time()
    for ep in range(EPOCHS):
        model.train(); tot = correct = n = 0
        for x, y in dl:
            opt.zero_grad(); out = model(x); loss = loss_fn(out, y); loss.backward(); opt.step()
            tot += loss.item() * len(y); correct += (out.argmax(1) == y).sum().item(); n += len(y)
    atomic_save({"state": model.state_dict(), "n": len(mine)}, SHARED / f"client{CID}_round{r}.pt")
    print(f"[client {CID}] round {r}/{ROUNDS} loss {tot/n:.3f} acc {correct/n:.3f} | lr {lr_r:.2e} | {time.time()-t0:.0f}s", flush=True)
print(f"[client {CID}] done", flush=True)
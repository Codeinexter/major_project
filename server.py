"""FedAvg server: waits for all client updates each round, averages them, evaluates, sends the new model back."""
import os, time, json, copy
import pandas as pd, torch
from sklearn.metrics import classification_report
from common import *

N      = int(os.environ.get("NUM_CLIENTS", 3))
ROUNDS = int(os.environ.get("ROUNDS", 5))
torch.set_num_threads(int(os.environ.get("CPU_THREADS", 2)))

SHARED.mkdir(exist_ok=True); OUT.mkdir(exist_ok=True)
for f in SHARED.glob("*"):                      # clear leftovers from earlier runs
    if f.is_file(): f.unlink()
val, test = load_frame("split_val.csv"), load_frame("split_test.csv")
model = build_model()
_, a, f1, _, _ = evaluate(model, val)
print(f"[server] round 0 (initial model) val acc {a:.4f} macroF1 {f1:.4f}", flush=True)
atomic_save(model.state_dict(), SHARED / "global_round1.pth")
(SHARED / "ready").write_text("1")              # clients start only after this exists (compose healthcheck)

history = []
best_f1, best_state, best_round = -1.0, None, 0   # same selection rule as the centralized run: best validation macro-F1
for r in range(1, ROUNDS + 1):
    t0 = time.time()
    updates = []
    for cid in range(N):
        wait_for(SHARED / f"client{cid}_round{r}.pt")
        updates.append(torch.load(SHARED / f"client{cid}_round{r}.pt", map_location="cpu"))
        print(f"[server] round {r}: client {cid} arrived after {time.time()-t0:.0f}s", flush=True)
    total = sum(u["n"] for u in updates)
    avg = {}
    for k in updates[0]["state"]:               # FedAvg: weighted by number of local samples
        if updates[0]["state"][k].is_floating_point():
            avg[k] = sum(u["state"][k] * (u["n"] / total) for u in updates)
        else:
            avg[k] = updates[0]["state"][k]     # e.g. BatchNorm counters
    model.load_state_dict(avg)
    vl, va, vf, _, _ = evaluate(model, val)
    history.append({"round": r, "val_loss": vl, "val_acc": va, "val_macro_f1": vf, "round_seconds": time.time() - t0})
    print(f"[server] round {r}/{ROUNDS} | val loss {vl:.3f} acc {va:.4f} macroF1 {vf:.4f} | {time.time()-t0:.0f}s", flush=True)
    pd.DataFrame(history).to_csv(OUT / "fedavg_history.csv", index=False)
    if vf > best_f1: best_f1, best_state, best_round = vf, copy.deepcopy(model.state_dict()), r
    if r < ROUNDS: atomic_save(model.state_dict(), SHARED / f"global_round{r+1}.pth")

torch.save(model.state_dict(), OUT / "global_final.pth")      # last round
torch.save(best_state, OUT / "global_best_val.pth")           # best validation macro-F1 round
results = {}
for tag, state in [("last_round", copy.deepcopy(model.state_dict())), ("best_val", best_state)]:
    model.load_state_dict(state)
    _, ta, tf1, preds, labels = evaluate(model, test)
    rnd = ROUNDS if tag == "last_round" else best_round
    print(f"[server] FINAL ({tag}, round {rnd}) test accuracy {ta:.4f} | macro-F1 {tf1:.4f}", flush=True)
    print(classification_report(labels, preds, target_names=CLASSES, zero_division=0), flush=True)
    results[tag] = {"round": rnd, "test_acc": ta, "test_macro_f1": tf1,
                    "report": classification_report(labels, preds, target_names=CLASSES, zero_division=0, output_dict=True)}
    pd.DataFrame({"label": labels, "pred": preds}).to_csv(OUT / f"test_predictions_{tag}.csv", index=False)
(OUT / "final_metrics.json").write_text(json.dumps(results, indent=2))
# Federated Learning for Edge Devices

Centralized vs. vanilla federated (FedAvg) **MobileNetV3-Small** for 7-class skin-lesion classification on
**HAM10000**. The federated clients are simulated as resource-limited Docker containers (different CPU and memory
limits), and everything runs on CPU only.

**Authors:** Roshaid Atique and Mohammad Anas
**Guide:** Prof. Rashid Ali
Department of Computer Engineering, Zakir Husain College of Engineering and Technology, Aligarh Muslim University

## What this project does

- Trains a **centralized baseline** (30 epochs) and a **vanilla FedAvg baseline** (30 rounds, 3 IID clients) with the same
  recipe: same split, model, augmentation, cosine learning-rate schedule and checkpoint rule (best validation macro-F1).
- Simulates three edge clients in Docker with different limits and measures each client's local training time.
- Compares both baselines on the same 2,014 test images: per-client and per-class metrics, confusion matrices and
  bootstrap confidence intervals.

## Results (single run, 2,014 test images)

| Model | Accuracy | Macro-F1 |
|---|---|---|
| Centralized (best validation epoch 29) | 0.7920 | 0.6225 |
| FedAvg global, best-validation round 21 | 0.7453 | 0.5680 |
| FedAvg global, last round 30 | 0.7542 | 0.5744 |
| FedAvg client 0 / 1 / 2 (round 30) | 0.7562 / 0.7532 / 0.7532 | 0.5766 / 0.5745 / 0.5798 |

- FedAvg reaches 94.1% of the centralized accuracy and 91.2% of its macro-F1 without sharing any image.
- All seven classes lose F1 (0.03 to 0.10); the largest drop is akiec.
- The three clients end level in test accuracy (within 0.003).

| Client | Limits | Local training time per round (mean ± std) |
|---|---|---|
| 0 | 1.0 CPU, 1 GB | 56.9 ± 4.8 s |
| 1 | 0.5 CPU, 768 MB | 75.3 ± 5.4 s |
| 2 | 2.0 CPU, 1.5 GB | 49.2 ± 4.2 s |

Client 1 is 1.53× slower than Client 2, so a synchronous round waits for the slowest device.

![Validation curves](docs/cmp_curves.png)
![Confusion matrices](docs/cmp_confusion.png)
![Per-class F1](docs/cmp_per_class_f1.png)
![Client training time](docs/client_training_time.png)

## Repository layout

| Path | Purpose |
|---|---|
| `notebooks/centralized_baseline.ipynb` | Centralized baseline; creates `split_train.csv`, `split_val.csv`, `split_test.csv` |
| `notebooks/make_imagenet_init.ipynb` | Saves the ImageNet-initialised model `imagenet_init.pth` |
| `common.py` | Shared data loading, model, augmentation and evaluation code |
| `server.py` | FedAvg server: aggregation, per-round validation, best-round selection, final test |
| `client.py` | One simulated edge client (local training on its shard) |
| `Dockerfile`, `docker-compose.yml` | Container setup: 1 server + 3 clients with different CPU/RAM limits |
| `analysis/plot_fl_clients_30.py` | Per-client loss, accuracy, macro-F1 and confusion matrices |
| `analysis/compare_central_vs_fl.py` | Centralized vs. FedAvg comparison with bootstrap intervals |
| `analysis/client_training_time.py` | Mean local training time per client from the container logs |
| `docs/` | Figures used in this README |

## Data (not included)

HAM10000 (Tschandl et al., *Scientific Data*, 2018) is not stored in this repository. Download it from its original
source and respect its license. The expected layout next to `docker-compose.yml` is:

```
dataset/                 # all HAM10000 images (any sub-folders)
split_train.csv          # image_id, dx, lesion_id (lesion-grouped split)
split_val.csv
split_test.csv
imagenet_init.pth        # created by notebooks/make_imagenet_init.ipynb
```

## How to run

```bash
pip install -r requirements.txt

# 1) create the ImageNet-initialised model:  notebooks/make_imagenet_init.ipynb  -> imagenet_init.pth
# 2) run the centralized baseline (also writes the split CSVs):  notebooks/centralized_baseline.ipynb

# 3) federated run: 1 server + 3 clients (needs Docker)
docker compose up --build                 # results in outputs_aligned_seed42/
SEED=43 docker compose up --build         # another seed, separate output folder

# 4) analysis (run BEFORE starting another FL run; the server clears shared/ at start)
python analysis/plot_fl_clients_30.py
python analysis/compare_central_vs_fl.py
docker logs edge0 > edge0.log; docker logs edge1 > edge1.log; docker logs edge2 > edge2.log
python analysis/client_training_time.py
```

### Settings (environment variables in `docker-compose.yml`)

| Variable | Default | Meaning |
|---|---|---|
| `NUM_CLIENTS` | 3 | Number of clients |
| `ROUNDS` | 30 | Communication rounds |
| `LOCAL_EPOCHS` | 1 | Local epochs per round |
| `PARTITION` | `iid` | `iid` or `noniid` (Dirichlet) |
| `ALPHA` | 0.5 | Dirichlet concentration (smaller = more skewed) |
| `SEED` | 42 | Random seed (partition and training) |
| `LR` | 5e-4 | Peak learning rate |
| `COSINE_LR` | 1 | Cosine decay over the rounds |
| `PERSIST_OPT` | 1 | Keep each client's AdamW state across rounds |

The server writes `fedavg_history.csv`, `global_final.pth` (last round), `global_best_val.pth` (best validation round),
`final_metrics.json` and `test_predictions_*.csv` to the output folder.

## Experimental platform

Intel Core i5-12500H (12 cores: 4 performance + 8 efficient, 16 threads), 16 GB DDR4-3200, 512 GB NVMe SSD.
Integrated graphics were not used. Container limits: server 2.0 CPU / 2 GB; clients 1.0 CPU / 1 GB,
0.5 CPU / 768 MB and 2.0 CPU / 1.5 GB.

## Limitations

Single seed, IID partition over three clients, and very small minority classes (22 df and 34 vasc test images), so
single-class differences should be read with caution.

## References

1. P. Tschandl, C. Rosendahl, H. Kittler, "The HAM10000 dataset...", *Sci. Data*, vol. 5, 180161, 2018.
2. A. Howard et al., "Searching for MobileNetV3", *ICCV*, 2019.
3. B. McMahan et al., "Communication-efficient learning of deep networks from decentralized data", *AISTATS*, 2017.
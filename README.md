# GraphAISMNet Web Server

## Files needed in this folder
- `app.py` — Main Streamlit application
- `requirements.txt` — Python dependencies
- `GRAPH.png` — Your methodology flowchart image (copy from your figures folder)
- `model_random_gt.pt` — GT model weights for random split
- `model_scaffold_gat.pt` — GAT model weights for scaffold split

## Step 1 — Save model weights from notebook

In your v15 notebook, run EV3 twice:

**For Random Split (GT model):**
```python
# Change best_model_cls_class to SimpleGT temporarily
torch.manual_seed(42)
gt_final = SimpleGT(NODE_DIM, DESC_DIM, hidden=128, heads=4, dropout=0.2).to(device)
opt_gt = AdamW(gt_final.parameters(), lr=0.001, weight_decay=1e-4)
sched_gt = build_scheduler(opt_gt, 0.001)
# Train on all random aug data
for ep in range(30):
    train_epoch(gt_final, full_ldr, opt_gt, sched_gt, base_hparams, device, ep, verbose=True, use_aug=True)
torch.save(gt_final.state_dict(), "model_random_gt.pt")
print("Saved model_random_gt.pt")
```

**For Scaffold Split (GAT model):**
```python
torch.manual_seed(42)
gat_final = SimpleGAT(NODE_DIM, DESC_DIM, hidden=128, heads=4, dropout=0.2).to(device)
opt_gat = AdamW(gat_final.parameters(), lr=0.001, weight_decay=1e-4)
sched_gat = build_scheduler(opt_gat, 0.001)
for ep in range(30):
    train_epoch(gat_final, full_ldr, opt_gat, sched_gat, base_hparams, device, ep, verbose=True, use_aug=True)
torch.save(gat_final.state_dict(), "model_scaffold_gat.pt")
print("Saved model_scaffold_gat.pt")
```

## Step 2 — Update normalization stats in app.py

Find these lines near the top and set to your Cell 4 values:
```python
Y_REG_MEAN = 6.45   # your actual mean
Y_REG_STD  = 1.12   # your actual std
```

## Step 3 — Copy GRAPH.png

Copy your methodology flowchart (GRAPH.png) into this folder.

## Step 4 — Test locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Step 5 — Deploy FREE on Streamlit Cloud
1. Push to GitHub (include model .pt files if <100MB, or use Git LFS)
2. Go to share.streamlit.io
3. New App → select repo → Deploy

## Note on model file size
If .pt files are >100MB, use Hugging Face Spaces instead:
1. huggingface.co/spaces → New Space → Streamlit
2. Upload all files including .pt models

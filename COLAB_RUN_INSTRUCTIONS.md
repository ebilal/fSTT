# Colab GPU Run Instructions

This covers:
1) Predict the **other speaker's next utterance** (conversation completion)
2) Predict **keyterms/keywords** for the other speaker (for Deepgram STT injection)

Both use the public Colab notebooks in `notebooks/`.

---

## 0) Create a Colab GPU runtime
1. Open Google Colab: https://colab.research.google.com
2. Click **Runtime → Change runtime type**.
3. Set **Hardware accelerator** to **GPU**.
4. Click **Save**.

---

## 1) Run the conversation completion notebook
Notebook: `notebooks/colab_train_gpu_public_dual_dataset.ipynb`

1. **Upload the notebook** to Colab (File → Upload notebook) or open it from GitHub.
2. The notebook is already configured with `REPO_URL = "https://github.com/ebilal/fSTT.git"` (no changes needed).
3. (Optional) In Colab: **Tools → Secrets**, add a secret:
   - `HF_TOKEN` (Hugging Face token, recommended for faster/less rate-limited downloads)
4. Run cells from top to bottom:
   - **Clone repo**
   - **Install dependencies**
   - **Mount Google Drive**
   - **Load HF token**
   - **Train** (`train_dual_epoch_test.py`) with `--target_role SYSTEM`
5. Outputs are written to:
   - `/content/drive/MyDrive/listener_prior_runs/<run_id>/`
6. The last cell runs a quick offline demo using the best encoder.

Notes:
- `--target_role SYSTEM` means: predict the **other person** in the dialog.
- `--max_dialogs_multiwoz 0` and `--max_dialogs_dailydialog 0` mean **no cap**.
- If `datasets` installs as 4.x, **restart runtime** and rerun.

---

## 2) Run the keyterms/keywords notebook
Notebook: `notebooks/colab_train_gpu_public_dual_dataset_keyterms.ipynb`

1. **Upload the notebook** to Colab (File → Upload notebook) or open it from GitHub.
2. The notebook is already configured with `REPO_URL = "https://github.com/ebilal/fSTT.git"` (no changes needed).
3. (Optional) In Colab: **Tools → Secrets**, add:
   - `HF_TOKEN`
4. Run cells from top to bottom:
   - **Clone repo**
   - **Install dependencies**
   - **Mount Google Drive**
   - **Load HF token**
   - **Train** (same script, same target role)
   - **Keyterms demo**: runs `src.demo_offline` to print extracted keyterms/keywords
5. Outputs are written to:
   - `/content/drive/MyDrive/listener_prior_runs/<run_id>/`

Notes:
- The demo uses retrieved candidates and `src.prior.extract_priors` to extract keyterms.
- For Deepgram injection, use `src.prior.build_deepgram_extra_kwargs`.

---

## 3) Optional: adjust size/epochs for speed
In either notebook, edit the training cell parameters:
- `--epochs 2`
- `--max_dialogs_multiwoz 500`
- `--max_dialogs_dailydialog 500`
- `--batch_size 16`

---

## 4) Troubleshooting
- **CUDA not available**: Confirm runtime GPU is enabled and rerun.
- **datasets too new**: If you see the assert failure, restart runtime and rerun.
- **Download rate limits**: Add `HF_TOKEN` in Colab Secrets.

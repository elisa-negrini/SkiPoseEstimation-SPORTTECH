# SkiPose
Una repository per l'**adattamento di pose 2D di sciatori** utilizzando deep learning e transformer.

---

## 📋 Panoramica
Questo progetto implementa un sistema di **completamento e adattamento di pose 2D per sciatori**, con particolare focus sul salto con gli sci. Il sistema utilizza un'architettura basata su **transformer** per prevedere i keypoint mancanti (principalmente gli sci) partendo da pose parziali estratte da immagini.

---

## 🏗️ Architettura
Il progetto utilizza due approcci principali:

### 1. SkiDat Network (`model.py`)
* Architettura basata su **Vision Transformer**.
* Supporto per embedding di patch di immagini.
* Backbone opzionale (**SwAV pretrained**) per feature extraction.
* Mascheramento adattivo dei joint per il training.

### 2. Adaptation Network (`model_fdp.py`)
* Network più leggero per **domain adaptation**.
* Focus sul completamento degli sci mancanti.
* Valutazione con metriche **MPJPE** (Mean Per Joint Position Error).

---

## 📁 Struttura dei File

### Training & Modelli
* `main.py`: Entry point principale per training con **SkiDat Network**.
* `main_fdp.py`: Entry point alternativo per **Adaptation Network**.
* `model.py`: Implementazione **SkiDat Network** con transformer.
* `model_fdp.py`: Implementazione **Adaptation Network**.
* `transformer.py`: Componenti transformer (Attention, FeedForward, PatchEmbed).
* `discriminator.py`: Discriminatore per approccio GAN (se utilizzato).

### Data Management
* `datamodule_ski.py`: DataModule PyTorch Lightning per dataset sci (**25 joint**).
* `datamodule_body_25.py`: DataModule per pose **BODY_25** (test/demo).
* `dataset/ski_datamodule.py`: DataModule alternativo con flip e normalizzazione.
* `dataset/mask_generator.py`: Generatore di maschere per training.

### Preprocessing
* `preprocess_ski_2d.py`: Preprocessing **Ski 2DPose Dataset**.
* `preprocess_ski_jump.py`: Preprocessing dataset **salto con sci annotato**.
* `preprocess_body_25.py`: Preprocessing pose **OpenPose BODY_25**.
* `preprocess_synthetic.py`: Preprocessing dati sintetici.
* `k_fold_split.py`: Split del dataset in fold per cross-validation.

### Utilities
* `utils.py`: Funzioni di utilità (normalizzazione, plotting, metriche).
* `utils_fdp.py`: Utilities specifiche (rotazioni, allineamento sci).
* `plot.py`: Funzioni per visualizzazione risultati.
* `domainadapt_flags.py`: Configurazione flags tramite `absl`.

---
## 🎯 Skeleton Format
Il progetto lavora con pose a **25 keypoint**:

(0-head, 1-neck, 2-r_shoulder, 3-r_elbow, 4-r_wrist, 5-r_pole, 6-l_shoulder, 7-l_elbow, 8-l_wrist, 9-l_pole, 10-r_hip, 11-r_knee, 12-r_ankle, 13-l_hip, 14-l_knee, 15-l_ankle, 16-r_ski_tip, 17-r_toes, 18-r_heel, 19-r_ski_tail, 20-l_ski_tip, 21-l_toes, 22-l_heel, 23-l_ski_tail, 24-pelvis)
---

## 📊 Dataset
Il progetto supporta multipli dataset:

* **Ski 2DPose Dataset (S2D)**: Dataset pubblico di sciatori.
* **Ski Jump Annotated (SJ)**: Dataset annotato manualmente di salti.
* **Synthetic Ski Jump (S)**: Dati sintetici generati.
* **YouTube Ski Jump**: Dataset estratto da video YouTube.
* **BODY_25**: Pose OpenPose per testing.

### Modalità di Training
| Modalità | Descrizione |
| :--- | :--- |
| **S2D** | Solo Ski 2DPose Dataset |
| **SJ** | Solo Ski Jump Dataset |
| **S2D_SJ** | Combinazione dei due |
| **S2D_SJ+S** | Tutti i dataset inclusi i dati sintetici (**default**) |

---

## 🔧 Preprocessing
### Pipeline di Preprocessing:
* **Ski 2D Dataset**: Rimozione *pole baskets*, aggiunta *pelvis* (centro anche). Formato finale: **23 joint**.
* **Ski Jump Dataset**: Split in fold per cross-validation (4 jump), normalizzazione e traslazione, creazione train/test split.
* **BODY_25**: Rimozione multipose, filtraggio pose incomplete, mappatura a formato target.
* **Synthetic Data**: Parsing JSON, correzione errori di annotazione (swap tail/hip), normalizzazione coordinate.

---

## 🚀 Training
### Configurazione Base (via flags):
| Categoria | Flag | Descrizione |
| :--- | :--- | :--- |
| **Directories** | `--data_path` | path principale dataset |
| | `--load_checkpoint` | path per salvare/caricare modelli |
| **Architettura** | `--n_joints` | 25 (numero joint target) |
| | `--use_backbone` | True/False (usa ResNet+SwAV) |
| | `--use_image_patches` | True/False (usa patch di immagini) |
| | `--freeze_fe` | True/False (congela feature extractor) |
| **Training** | `--lr` | 2e-4 |
| | `--batch_size` | 8 |
| | `--n_epochs` | 25 |
| | `--masked_joints` | numero joint da mascherare |
| | `--data_augmentation` | True/False |
| | `--fine_tuning` | True/False |
| **Dataset** | `--train_set_mode` | 'S2D', 'SJ', 'S2D_SJ', 'S2D_SJ+S' |
| | `--testing_jump` | 1-4 (fold per cross-validation) |
| | `--filter_openpose` | 'delete_pose' o 'no_filter' |

### Avvio Training:
```bash
# Training standard con tutti i dataset
python main.py --mode=train --train_set_mode=S2D_SJ+S

# Fine-tuning con Adaptation Network
python main_fdp.py --mode=train --fine_tuning=True \
  --load_pretrained_model=path/to/checkpoint.ckpt
```

# Testing/Demo

## Test con Adaptation Network
python main_fdp.py --mode=demo \
  --dataset=body_25 \
  --testing_jump=1 \
  --load_checkpoint=path/to/checkpoint.ckpt

Il testing:

- Carica pose BODY_25 (senza sci).

- Predice i keypoint degli sci mancanti.

- Calcola metriche MPJPE (totale e solo sci).

- Salva visualizzazioni in test_result_dir.


## 📈 Metriche di Valutazione

* **MPJPE (Mean Per Joint Position Error)**: Errore quadratico medio per giunto, misurato in **pixel**.
* **Normalized MPJPE**: MPJPE normalizzato per **altezza figura**.
* **Skis MPJPE**: MPJPE calcolato **solo sui joint degli sci**.
* **MSE Loss (Mean Squared Error)**: Loss principale durante training.

---

## 🎨 Features Principali

### Data Augmentation
* Simmetria assiale (**flip orizzontale**).
* **Rotazioni casuali** (**0-20°**).
* Flip dei joint durante loading.

### Normalizzazione
* Centramento su **pelvis/neck**.
* Scaling basato su distanza **head-pelvis**.
* Denormalizzazione per visualizzazione.

### Masking Strategy
* **Training mode**: Maschera sci + joint casuali.
* **Demo mode**: Maschera solo sci (**14-24**).
* Supporto per maschere specifiche per dataset.

### Ski Alignment
* **Allineamento sci** tramite regressione lineare.
* **Proiezione joint** sulla linea dello sci.
* Applicabile sia a destra che sinistra.

---

## 🔍 Note Tecniche Avanzate

### Formati dei Dati
* **Train**: Lista di `[source_id, pose]` dove `source_id` indica il dataset.
* **Test**: Dizionario `{image_name: pose}` per tracking.
* **Pose**: Array numpy/tensor di shape `[n_joints, 2]`.

### Weighted Loss
Quando `weighted_loss=True` e `train_set_mode='total'`, il dataset viene bilanciato:

* Ski Jump samples: $weight = 1.0$
* Ski 2D samples: $weight = \frac{len\_ski\_jump}{len\_ski\_2d}$

### Cross-Validation
4 fold basati su sequenze temporali di salti:

| Fold (Testing Jump) | Range di Frames |
| :--- | :--- |
| **Jump 1** | frames 0-63 |
| **Jump 2** | frames 121-163 |
| **Jump 3** | frames 342-414 |
| **Jump 4** | frames 273-334 |

---

## 📦 Dipendenze Principali
* **PyTorch + PyTorch Lightning**
* **Transformers** (`timm`, `einops`)
* **OpenCV** per elaborazione immagini
* **Wandb** per logging
* **NumPy, Matplotlib** per utilities

---

## 📝 TODO / Note Mancanti
* Checkpoint dei modelli (`*.pt` files)
* Dataset completi
* Risultati di training salvati
* File di configurazione specifici per esperimenti

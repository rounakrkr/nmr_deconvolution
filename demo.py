"""
MixNet Demo — Interactive Presentation
Har technical term ke saath uska meaning likha hai.
Run: python demo.py
"""

import os, sys, time
import numpy as np
import torch

# Fix Windows encoding
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

def pause():
    input(f"\n{'='*60}\n  >> Press Enter for next step...\n{'='*60}")

def header(step, title, subtitle=""):
    print(f"\n\n{'#'*60}")
    print(f"  STEP {step}: {title}")
    if subtitle:
        print(f"  {subtitle}")
    print(f"{'#'*60}\n")


# ====================================================
header(1, "KYA HAI MIXNET?", "Poora project 1 minute mein")
# ====================================================

print("""
  PROBLEM:
  --------
  Jab kisi sample (blood, plant, drug) ka NMR karte hain,
  toh ek MIXED spectrum milta hai -- saare compounds ke signals
  overlap ho rahe hain. Kiska signal hai ye pata nahi chalta.
  
  Abhi tak iske liye multiple experiments karne padte hain = time + cost.
  
  SOLUTION: MixNet
  -----------------
  Ek AI model jo kisi bhi mixed NMR spectrum ko ALAG karta hai:

    Input:  M mixed spectra (jisme sab overlap hai)
              |
         [ MixNet ]
              |
    Output: N alag alag compound spectra (separated!)

  KEY POINTS:
  - Kisi bhi sample pe kaam karega (blood, plant, drug)
  - Compounds ko NAAM nahi deta -- sirf ALAG karta hai
  - Agar alag kiya spectrum kisi database mein match nahi karta
    --> Possibly NEW compound discover ho sakta hai!
""")

pause()


# ====================================================
header(2, "MODEL KA DESIGN", "U-Net = Compress karo, phir wapas bada karo")
# ====================================================

print("""
  MixNet ka design "U-Net" se inspired hai.
  
  Ye 3 parts mein kaam karta hai:
  
  PART 1: ENCODER (chhota karo)
  ----
  Spectrum ko step by step chhota karta hai.
  Jaise photo ka resolution kam karo -- 1080p --> 720p --> 480p --> 144p
  Har step pe size kam hota hai, but model "kya important hai" wo seekhta hai.
  Har step ka result SAVE karta hai (baad mein kaam aayega).
  
  PART 2: BOTTLENECK (sabse chhota point)
  ----
  16384 points ka spectrum 16 points mein compress ho gaya.
  Ye 16 points = poore spectrum ka "summary" hai.
  
  PART 3: DECODER (wapas bada karo)
  ----
  Summary se wapas full-size spectrum banata hai.
  Encoder se saved details USE karta hai (SKIP CONNECTIONS).
  Skip connections = "cheat sheet" -- decoder ko yaad dilate hain ki 
  original mein kahan kya tha.
  
  Visual:
  
  16384 --> 4096 --> 1024 --> 256 --> 64 --> [16] --> 64 --> 256 --> 1024 --> 4096 --> 16384
  |chhota karo                        | summary |                        bada karo|
  |           ENCODER                 |BOTTLENECK|           DECODER               |
  |----skip connection 1----------->>-|---------|->>-use skip 1-------------------|
  |--------skip connection 2------>>--|---------|-->>-use skip 2--------------|
  (saved details wapas laaye -- fine details preserve hote hain)
""")

pause()


# ====================================================
header(3, "BUILDING BLOCKS", "Model 3 blocks se bana hai")
# ====================================================

from src.models.blocks import ConvBlock, DownBlock, UpBlock

print("""  BLOCK 1: ConvBlock (Pattern Finder)
  ====================================
  Kya karta hai:
    Spectrum pe ek chhoti WINDOW (3 points wide) slide karta hai.
    Har position pe check karta hai: "yahan kya pattern hai?"
    Jaise magnifying glass slide karo page pe.
  
  Andar kya hota hai:
    Conv1D     = window slide karke pattern dhundho
    BatchNorm  = numbers ko ek range mein laao (normalize)
    ReLU       = negative values hatao (NMR mein negative nahi hota)
    (ye 2 baar hota hai for better pattern detection)
""")
cb = ConvBlock(64, 128)
x = torch.randn(1, 64, 1024)
out = cb(x)
print(f"  Live test:")
print(f"    Input:  {x.shape[1]} patterns dhundh rahe the, {x.shape[2]} points pe")
print(f"    Output: {out.shape[1]} patterns mil gaye, {out.shape[2]} points pe")
print(f"    Size same rahi (1024 --> 1024), sirf patterns zyada ho gaye\n")

print("""  BLOCK 2: DownBlock (Compress karo -- Encoder mein use hota hai)
  ================================================================
  Kya karta hai:
    Step 1: ConvBlock lagao (patterns dhundho)
    Step 2: MaxPool lagao = 4 mein se SABSE BADA number rakho, baaki hatao
            Size 4x CHHOTA ho jaata hai!
  
  2 outputs deta hai:
    - Full-size result --> SAVE karo (skip connection ke liye)
    - Chhota result --> next stage mein jaayega
""")
db = DownBlock(64, 128, pool_kernel=4, pool_stride=4)
skip, pooled = db(x)
print(f"  Live test:")
print(f"    Input:   {x.shape[2]} points")
print(f"    Skip:    {skip.shape[2]} points (full-size SAVED for later)")
print(f"    Pooled:  {pooled.shape[2]} points (4x CHHOTA! next stage mein jaayega)\n")

print("""  BLOCK 3: UpBlock (Wapas bada karo -- Decoder mein use hota hai)
  ================================================================
  Kya karta hai:
    Step 1: Chhoti version ko 4x STRETCH karo (bada banao)
    Step 2: Encoder se saved "skip" chipkao (details wapas laao)
    Step 3: ConvBlock lagao (combine + refine karo)
""")
ub = UpBlock(128, 128, 64, scale_factor=4)
up_out = ub(pooled, skip)
print(f"  Live test:")
print(f"    Input:  {pooled.shape[2]} points (chhota)")
print(f"    + Skip: {skip.shape[2]} points (saved details)")
print(f"    Output: {up_out.shape[2]} points (WAPAS original size!)")

pause()


# ====================================================
header(4, "POORA MODEL", "MixNet = 5 DownBlocks + Bottleneck + 5 UpBlocks")
# ====================================================

from src.models.unet1d import MixNet

M = 8     # mixtures
N = 5     # compounds
L = 16384 # spectral points

model = MixNet(in_channels=M, out_channels=N)
total_params = sum(p.numel() for p in model.parameters())

print(f"  Model: MixNet (1D U-Net)")
print(f"  Input:  {M} mixture spectra, har ek {L} points ka")
print(f"  Output: {N} compound spectra, har ek {L} points ka")
print(f"""
  PARAMETERS: {total_params:,} (71 lakh numbers)
  ---------
  Kya hain ye?
    Model ke andar adjustable numbers hain -- jaise knobs.
    Training mein ye numbers adjust hote hain taaki output sahi aaye.
  
  71 lakh KYUN?
    Har Conv1D layer mein filters hain. Ek example:
    Conv1D(64 channels, 128 channels, window=3)
    = 64 x 128 x 3 = 24,576 numbers SIRF EK layer mein!
    5 encoder + 5 decoder + bottleneck = bahut layers
    Sab milake = 71,11,429
  
  ZYADA KARE TOH? (jaise 50 lakh se 1 crore)
    + Complex patterns seekh sakta hai
    - Zyada data chahiye warna ratta maarlega (overfitting)
    - Training slow hogi
  
  KAM KARE TOH? (jaise 71 lakh se 10 lakh)
    + Fast training
    - Simple patterns hi seekhega, complex spectra pe galat hoga
  
  71 lakh = BALANCE hai 16384 points ke problem ke liye.
""")

print(f"  Encoder (compress):")
print(f"    16384 --> 4096 --> 1024 --> 256 --> 64 --> 16 (bottleneck)")
print(f"  Decoder (expand):")
print(f"    16 --> 64 --> 256 --> 1024 --> 4096 --> 16384")

print(f"\n  Forward Pass (data model se guzaarna -- test ki chain kaam kar rahi hai):")
x = torch.randn(1, M, L)
with torch.no_grad():
    y = model(x)
print(f"    Input diya:  {list(x.shape)}")
print(f"    Output aaya: {list(y.shape)}")
print(f"    Output min: {y.min().item():.4f} (positive! Softplus kaam kar raha hai)")
print(f"    (Abhi output random hai kyunki model ne SEEKHA nahi hai abhi)")

pause()


# ====================================================
header(5, "TRAINING DATA", "Model ko kaise sikhaate hain?")
# ====================================================

from src.data.dummy_generator import generate_mixture_set

ppm = np.linspace(0, 10, 4096)

print("""
  Model ko sikhaane ke liye PAIRED data chahiye:
  
    Input (question paper):  M mixture spectra (mixed)
    Target (answer key):     N compound spectra (individual)
  
  Model ko dikhate hain: "ye mixed hai, ye answer hai"
  Baar baar dikhao --> model seekh jaata hai pattern.
  
  DATA KAHAN SE AAYEGA?
    Option 1: Didi/team se realistic synthetic data milega
    Option 2: Testing ke liye hamare paas dummy generator hai
              (random Gaussian peaks se fake spectra banata hai)
  
  Dummy Generator kaise kaam karta hai:
""")

data = generate_mixture_set(ppm, num_compounds=3, num_mixtures=5, noise_std=0.005)

print(f"    Step 1: {data['compounds'].shape[0]} random compounds banaye")
print(f"    Step 2: Random concentrations assign ki (kitna kiska hai mix mein):")
for i in range(min(3, len(data['concentrations']))):
    c = np.round(data['concentrations'][i], 3)
    print(f"            Mixture {i+1}: Compound1={c[0]:.1%}, Compound2={c[1]:.1%}, Compound3={c[2]:.1%}")
print(f"    Step 3: mixture = concentration1 x compound1 + concentration2 x compound2 + ...")
print(f"    Step 4: Thoda noise add kiya (real data jaisa lagein)")
print(f"\n    Result: {data['mixtures'].shape[0]} mixtures + {data['compounds'].shape[0]} compounds generated!")

pause()


# ====================================================
header(6, "LOSS FUNCTION", "Model ko kaise batate hain ki GALAT kiya?")
# ====================================================

from src.training.losses import CombinedLoss

print("""
  Loss = ek number jo batata hai "model kitna galat hai"
  
  Jaise exam mein marks = kitna sahi kiya
  Loss = kitna GALAT kiya (kam = better)
  
  Hamaare model mein 2 CHECKS hain:
  
  CHECK 1: SPECTRAL LOSS (weight: 1.0) -- MAIN CHECK
  -------
    "Kya predicted compounds sahi dikh rahe hain?"
    
    Har point pe: (prediction - actual)^2 ka average
    Jaise: predicted=5, actual=3 --> (5-3)^2 = 4 (galti)
    Ye number kam hona chahiye.
  
  CHECK 2: RECONSTRUCTION LOSS (weight: 0.5) -- PHYSICS CHECK
  -------
    "Kya predicted compounds ko add karne pe original mixture banta hai?"
    
    Kyun zaruri hai?
    Imagine: model ne 3 compounds predict kiye, sab individually acche dikhte hain
    BUT agar unko add karein toh original mixture NAHI banta!
    Matlab physics galat hai. Ye loss ye pakadta hai.
  
  TOTAL LOSS = 1.0 x Spectral + 0.5 x Reconstruction
  
  1.0 aur 0.5 kyun?
    Spectral loss MAIN hai (1.0 = full importance)
    Reconstruction SECONDARY hai (0.5 = half importance)
    Ye ratio adjust kar sakte hain agar zarurat pade.
""")

B, N_max, M_max, L_demo = 2, 5, 8, 1024
pred = torch.randn(B, N_max, L_demo).abs()
true = torch.randn(B, N_max, L_demo).abs()
conc = torch.rand(B, M_max, N_max)
mixes = torch.bmm(conc, true) + 0.01 * torch.randn(B, M_max, L_demo)
criterion = CombinedLoss(1.0, 0.5)
loss, ld = criterion(pred, true, conc, mixes)
print(f"  Live test (random data, untrained model):")
print(f"    Spectral loss:       {ld['spectral_loss']:.4f} (high = model abhi seekha nahi)")
print(f"    Reconstruction loss: {ld['reconstruction_loss']:.4f}")
print(f"    Total loss:          {ld['loss']:.4f}")
print(f"    Training mein ye numbers DECREASE honge!")

pause()


# ====================================================
header(7, "LIVE TRAINING!", "Dekhiye model seekh raha hai!")
# ====================================================

print("""
  TRAINING kya hota hai?
  ----------------------
  1. Data dikhao model ko (forward pass)
  2. Check karo kitna galat kiya (loss calculate karo)
  3. Parameters thoda adjust karo (backpropagation)
  4. Repeat!
  
  Har baar jab poora data 1 baar dekh liya = 1 EPOCH
  Jaise exam prep: 1 baar poora syllabus padha = 1 epoch
  5 baar padha = 5 epochs
""")

try:
    epochs_input = input("  Kitni baar sikhana hai? (number daalo, default 5): ").strip()
    num_epochs = int(epochs_input) if epochs_input else 5
except:
    num_epochs = 5

print(f"\n  {num_epochs} epochs training shuru!")
print(f"  (Pehle 10 test samples generate kar raha hai...)\n")

from src.data.dummy_generator import generate_dataset
from src.data.dataset import NMRMixtureDataset, collate_fn
from torch.utils.data import DataLoader

demo_dir = os.path.join(PROJECT_ROOT, "data", "demo_temp")
generate_dataset(num_samples=10, save_dir=demo_dir, spectral_length=4096, seed=42)

dataset = NMRMixtureDataset(demo_dir)
loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)

small_model = MixNet(
    in_channels=dataset.global_max_m,
    out_channels=dataset.global_max_n,
    encoder_channels=[32, 64, 128, 128, 256]
)
small_params = sum(p.numel() for p in small_model.parameters())

optimizer = torch.optim.AdamW(small_model.parameters(), lr=1e-3)
# AdamW = optimizer = "coach" jo parameters adjust karta hai
# lr = learning rate = kitna bada step lo (0.001 = chhota safe step)

criterion = CombinedLoss(1.0, 0.5)

losses_history = []
print(f"  Demo model: {small_params:,} parameters (chhota model for speed)")
print(f"  Learning rate: 0.001 (har step mein kitna adjust karo)")
print()
print(f"  {'Baar':<8} {'Galti (Loss)':<15} {'Status'}")
print(f"  {'-'*45}")

for epoch in range(num_epochs):
    epoch_loss, batches = 0, 0
    for batch in loader:
        optimizer.zero_grad()
        output = small_model(batch['mixtures'])
        loss, losses = criterion(
            output, batch['compounds'],
            batch['concentrations'], batch['mixtures'],
            batch.get('compound_mask'), batch.get('mixture_mask')
        )
        loss.backward()   # backpropagation = galti kahan hui wo trace karo
        optimizer.step()  # parameters adjust karo
        epoch_loss += losses['loss']
        batches += 1

    avg_loss = epoch_loss / batches
    losses_history.append(avg_loss)
    
    status = ""
    if len(losses_history) > 1:
        if avg_loss < losses_history[-2]:
            status = "  << SEEKH RAHA HAI! (loss kam hua)"
        else:
            status = "  (thoda upar gaya -- normal hai)"
    
    print(f"  {epoch+1:<8} {avg_loss:<15.4f}{status}")

drop = (losses_history[0] - losses_history[-1]) / losses_history[0] * 100
print(f"\n  Galti: {losses_history[0]:.4f} --> {losses_history[-1]:.4f}")
if drop > 0:
    print(f"  {drop:.1f}% kam hui galti! Model SEEKH RAHA HAI!")
    print(f"  Zyada data + zyada epochs = aur better results")

import shutil
if os.path.exists(demo_dir):
    shutil.rmtree(demo_dir)

pause()


# ====================================================
header(8, "SETTINGS KAISE CHANGE KARE?", "config.yaml file mein sab settings hain")
# ====================================================

print("""
  File: src/configs/config.yaml
  Ye ek settings file hai -- notepad mein khol ke edit kar sakte ho.
  
  MAIN SETTINGS:
  
  epochs: 200
    = Kitni baar poora data dekhna hai
    = Zyada = better seekhega (but ek point ke baad ratta maarega)
    = Sir bole "10 baar" toh 10 likho, "500 baar" toh 500
  
  batch_size: 16
    = Ek baar mein kitne samples dikhao
    = Jaise class mein ek baar mein 16 students ka test lo
    = Zyada = fast but zyada memory chahiye
  
  learning_rate: 0.0001
    = Har baar kitna adjust karo (step size)
    = Bada = fast but galat jagah ja sakta hai
    = Chhota = slow but safe
    = 0.0001 = bahut chhota, safe starting point
  
  patience: 20
    = Agar 20 baar check kiya aur koi improvement nahi = training roko
    = Jaise student 20 din se marks same hain = "bas ab aur nahi hoga"
  
  spectral_length: 16384
    = Ek spectrum mein kitne points
    = 16384 = sir ne confirm kiya tha
  
  encoder_channels: [64, 128, 256, 256, 512]
    = Har stage mein kitne patterns dhundhe
    = Zyada numbers = zyada parameters = zyada seekhega but slow
  
  KAISE CHANGE KARE?
    1. src/configs/config.yaml file kholo (notepad ya VS Code se)
    2. Number badlo
    3. Save karo
    4. python main.py --mode train
  
  FULL PIPELINE:
    python main.py --mode generate   = Training data banao
    python main.py --mode train      = Model train karo
    python main.py --mode evaluate   = Results dekho
    python main.py --mode all        = Sab ek saath karo
""")

print(f"  PROJECT STATUS:")
print(f"  +---------------------------------------------------+")
print(f"  |  Model architecture        -- DONE                |")
print(f"  |  Data pipeline             -- DONE                |")
print(f"  |  Loss functions            -- DONE                |")
print(f"  |  Training loop             -- DONE                |")
print(f"  |  Realistic training data   -- WAITING (from team) |")
print(f"  |  Full training + results   -- NEXT STEP           |")
print(f"  +---------------------------------------------------+")

print(f"\n\n  Demo complete!")

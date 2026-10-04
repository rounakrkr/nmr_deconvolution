# Complete Study Guide — Everything You Need to Know

> Har concept: **Kya hai | Kyun use kiya | Kaise kaam karta hai | Kya nahi hai**

---
---

# Section A: NMR Basics (Sir assume karenge ye pata hai)

---

### NMR Spectroscopy
- **Kya:** Nuclear Magnetic Resonance — ek analytical technique jo molecules ki structure aur identity batati hai
- **Kaise:** Sample ko strong magnetic field mein rakho, radio waves maaro → nuclei (mainly ¹H, hydrogen) respond karte hain → signal milta hai
- **Output:** Spectrum — x-axis pe chemical shift (ppm), y-axis pe intensity (signal strength)
- **Kyun important:** Non-destructive (sample kharab nahi hota), quantitative (kitna hai wo bhi pata chalta hai)

### Chemical Shift (ppm)
- **Kya:** Har compound ka peak spectrum mein specific position pe aata hai, measured in ppm (parts per million)
- **Kyun ppm:** Magnetic field strength se independent unit — 600 MHz ya 700 MHz spectrometer pe same ppm value
- **Example:** Glycine ka peak ~3.55 ppm pe aata hai, Alanine ka ~1.48 ppm pe
- **Hamare config mein:** 0 to 10 ppm range use kar rahe hain

### Peaks aur Overlap
- **Kya:** Har compound spectrum mein ek ya zyada peaks produce karta hai
- **Problem:** Jab multiple compounds mix hote hain → peaks ek doosre pe overlap karte hain → dekhna mushkil ki kiska peak hai
- **Kyun mushkil:** Complex mixtures (blood, plant extracts, drug formulations) mein dozens of compounds hote hain — sab ke peaks 0–10 ppm mein squeeze hain

### Linewidth
- **Kya:** Peak kitni chaudi hai (Hz mein measure hote hain)
- **Small molecules:** Narrow/sharp peaks (< 3.66 Hz) — kyunki chhote hain, freely move karte hain
- **Macromolecules** (proteins, lipids): Broad/wide peaks (≥ 3.66 Hz) — kyunki bade hain, slowly tumble karte hain
- **SENNet mein:** Yahi threshold (3.66 Hz) use kiya classify karne ke liye

### Peak Shape
- **Lorentzian:** Ideal NMR peak shape — pure exponential decay se aata hai
- **Gaussian:** Bell curve shape — hamara model ye use karta hai (sir ne confirm kiya)
- **Pseudo-Voigt:** Gaussian + Lorentzian ka mixture — real peaks often ye hote hain
- **Kya nahi:** Real peaks perfectly Lorentzian ya Gaussian nahi hote — shimming errors, temperature variations se shape distort hota hai

### FID (Free Induction Decay)
- **Kya:** Raw signal jo NMR instrument se aata hai — time-domain mein hota hai
- **Kaise spectrum banta hai:** FID pe Fourier Transform lagao → frequency-domain spectrum milta hai
- **SENNet mein:** Synthetic data FID equation se generate kiya gaya tha
- **Hamare mein:** Hum directly frequency-domain spectra ke saath kaam kar rahe hain (FID se nahi)

---

## NMR Experiments (Sir zaroor poochenge)

### NOESY (Nuclear Overhauser Effect SpectroscopY)
- **Kya:** Standard 1D NMR experiment — **sab dikhata hai** (small molecules + macromolecules dono)
- **Kyun standard:** Sabse basic, comprehensive spectrum deta hai
- **Problem:** Sab overlap hota hai — kisi ek cheez ko isolate karna mushkil

### CPMG (Carr-Purcell-Meiboom-Gill)
- **Kya:** Special experiment jo **macromolecule signals suppress** karta hai
- **Kaise:** T2 relaxation use karta hai — macromolecules ka T2 chhota hota hai, toh unke signals pehle decay ho jaate hain
- **Output:** Sirf small molecule peaks dikhte hain — clean spectrum
- **Kyun separately karna padta hai:** Extra spectrometer time lagta hai

### Diffusion-edited
- **Kya:** Experiment jo **small molecule signals suppress** karta hai
- **Kaise:** Diffusion coefficient use karta hai — chhote molecules fast diffuse karte hain, gradient pulse se unke signals cancel ho jaate hain
- **Output:** Sirf macromolecule (lipids, proteins) peaks dikhte hain

### pJRES (Projected J-Resolved)
- **Kya:** 2D experiment ka 1D projection — **J-coupling information** separate karta hai
- **Kaise:** Chemical shift aur J-coupling ko alag dimensions mein resolve karta hai
- **Kyun useful:** Overlapping multiplets resolve ho jaate hain
- **Paper 4 mein:** Cascade approach se better predict hota hai (NOESY → CPMG → pJRES)

### Summary Table — Experiments

| Experiment | Kya dikhata hai | Kyun use karte hain |
|-----------|----------------|-------------------|
| NOESY | Sab kuch | Complete picture |
| CPMG | Sirf small molecules | Clean metabolite spectrum |
| Diffusion-edited | Sirf macromolecules | Lipid/protein analysis |
| pJRES | Decoupled peaks | Overlap resolve karna |

**Key point for sir:** Currently 4 experiments per sample karne padte hain. Papers (2 & 4) aur hamaara model yahi shortcut dhundh rahe hain — ek experiment se baaki sab nikalo.

---
---

# Section B: Papers — In Depth

---

## Paper 1: Review — Deep Learning in NMR

### Kya hai
Survey paper — 2025 tak ke saare DL applications in NMR ka overview. Khud koi naya method nahi propose kiya, existing literature organize ki.

### Kyun padha
Big picture samajhne ke liye — ki field mein kya ho raha hai, kahan gaps hain, aur hamaara kaam kahaan fit baithta hai.

### Key DL applications in NMR (jo review mein cover hain):

**1. NUS Reconstruction**
- **Kya:** Non-Uniform Sampling — pura data collect karne ki jagah sirf kuch points lo, baaki DL se reconstruct karo
- **Kyun:** Multi-dimensional NMR mein acquisition time exponentially badhta hai
- **Result:** 75-90% time save
- **Hamare liye:** Directly relevant nahi — hum 1D spectra ke saath kaam kar rahe hain

**2. Denoising**
- **Kya:** Neural network se noise hatao, signal preserve karo
- **Kyun:** Low concentration samples mein SNR poor hota hai
- **Hamare liye:** Indirectly relevant — hamaara model bhi noisy input handle karega

**3. Peak Picking / Deconvolution**
- **Kya:** Automated peak detection + overlapping peaks ko separate karna
- **Kyun:** Manual peak picking subjective hai — expert bias
- **Hamare liye:** Directly related — hamaara kaam bhi deconvolution hi hai, but DL-based

**4. Chemical Shift Prediction**
- **Kya:** Molecular structure → NMR spectrum predict karna (forward problem)
- **Kyun:** DFT calculations bahut slow hain, DL fast alternative
- **Hamare liye:** Not our problem — hum inverse kaam kar rahe hain (spectrum → compounds)

**5. Structure Elucidation**
- **Kya:** NMR spectrum → molecular structure guess karna (inverse problem)
- **Kyun:** Drug discovery, natural product identification
- **Hamare liye:** Related concept but different scope

### Architectures jo review mein mentioned hain:

| Architecture | Kya karta hai | Kahaan use hota hai | Hum use kar rahe? |
|-------------|--------------|-------------------|-----------------|
| **CNN / U-Net** | Spatial patterns extract | Spectral decomposition, denoising | **Haan ✅** |
| **RNN / LSTM** | Sequential data process | FID time series | Nahi |
| **Transformer** | Long-range dependencies | Sequence-to-sequence | Nahi |
| **GNN** | Graph structure process | Molecular structure prediction | Nahi |
| **MLP** | Simple regression | Parameter estimation | Nahi |

---

## Paper 2: SENNet — Step by Step

### Problem
Blood NOESY spectrum = small molecules + macromolecules overlapped. Alag karne ke liye separately CPMG + Diffusion experiments karne padte hain → time waste.

### Data Generation (sabse important step — yahi seekhna hai)

**Step 1: Real spectrum analyse karo**
- Ek real plasma NOESY spectrum liya
- `scipy.signal.find_peaks` se peaks detect kiye
- `scipy.signal.peak_widths` se linewidth measure ki
- Har peak ke 3 parameters extract kiye: **frequency (ν), linewidth, amplitude**

**Step 2: Classify karo**
- Linewidth < 3.66 Hz → **small molecule** peak
- Linewidth ≥ 3.66 Hz → **macromolecule** peak
- **3.66 Hz kyun?** Supplementary material mein statistically determine kiya — yahi optimal threshold tha

**Step 3: Synthetic spectra banao**
- Extracted parameters ko **randomly perturb** karo (range mein)
- FID equation use karke time-domain signal generate karo:
  ```
  FID = Σ A_k × exp(2πiν_k × t) × exp(-α_k × t)
  ```
- Fourier Transform → frequency domain spectrum
- Random noise add karo (real jaisa lagein)

**Step 4: Paired data**
- Input X = total spectrum (macro + small dono)
- Target Y = sirf macromolecule spectrum
- Ye pairs se model train hota hai

### SENNet Architecture

1D U-Net — basically same concept jo hamaara MixNet use karta hai:
- Encoder: Downsample (features extract)
- Bottleneck: Compressed representation
- Decoder: Upsample (reconstruct)
- Skip connections: Fine details preserve

### Loss Function

```
Loss = w × TVE + NMSE
```

**NMSE (Normalized Mean Squared Error):**
- Standard prediction error
- Predicted aur target mein kitna fark hai

**TVE (Total Variation Error):**
- Output kitna smooth hai ye measure karta hai
- `TVE = Σ|x[n] - x[n-1]|` — consecutive points ka difference
- **Kyun:** Macromolecule spectrum **smooth** hona chahiye (broad peaks) — agar sharp spikes aa rahe hain toh penalty do
- **Kya nahi:** Ye hamare model mein nahi hai — kyunki hamaare output mein sharp peaks bhi expected hain (individual compounds ke peaks sharp hote hain)

**w = 22:** TVE ko kitna weight dena hai — 22 best nikla experiments se

### Inference
```
Input: Real NOESY spectrum
         ↓
    Trained SENNet
         ↓
Output: Predicted macromolecule spectrum
         ↓
Subtraction: Small molecule = Total - Macro
```

### Results
- 92.6% Pearson correlation with experimental CPMG
- 600 MHz + 700 MHz dono pe kaam kiya
- 113 serum samples pe validated (MetaboLights MTBLS37411)

### SENNet vs MixNet

| | SENNet | MixNet |
|--|--------|--------|
| Input | 1 spectrum | M spectra |
| Output | 1 macro spectrum (+ subtraction) | N individual compound spectra |
| Classification | 2 broad categories | N named compounds |
| Data | Synthetic (FID-based) | Synthetic (Gaussian peaks) |
| Loss | NMSE + TVE | Spectral MSE + Reconstruction MSE |
| TVE use kiya? | Haan (smooth output chahiye) | Nahi (sharp peaks expected) |

---

## Paper 3: NMR-Onion — Classical Approach

### Kya hai
Mathematical deconvolution — **deep learning nahi hai.** Traditional curve fitting approach.

### "Onion" naam kyun?
Jaise onion ki layers peel karte hain — spectrum ke overlapping peaks ko layer by layer fit karke alag karte hain.

### 5-Step Pipeline Detail:

**Step 1: Band-Pass Filtering**
- **Kya:** Spectrum ka sirf ek region select karo (Region of Interest)
- **Kyun:** Poora spectrum ek saath fit karna computationally expensive + unnecessary
- **Kaise:** Digital filter (time-domain mein multiply karo)

**Step 2: Peak Detection**
- **Kya:** ROI mein peaks kahan hain identify karo
- **Kaise:** Savitzky-Golay filter — smoothing + derivative calculation
  - 1st derivative: peak position (zero crossing)
  - 2nd derivative: peak ki sharpness
- **Kyun ye method:** Fast, well-established, robust to noise

**Step 3: Multi-Model Fitting**
- 3 mathematical models simultaneously fit karo:
  - **Model 1 — Lorentzian:** `exp(-αt)` — ideal NMR peak, simplest
  - **Model 2 — Pseudo-Voigt:** `(1-η)×exp(-αt) + η×exp(-αt²)` — Gaussian-Lorentzian mix, more flexible
  - **Model 3 — Power Law:** `exp(-αt^β)` — most flexible, handles distorted peaks
- + **Asymmetric skew factor:** `exp(exp(jγ)×t)` — shimming errors se peaks asymmetric ho jaate hain

**Step 4: BIC Model Selection**
- **BIC = Bayesian Information Criterion**
- **Kya:** Statistical score — model fit kitna accha hai AND model kitna simple hai
- **Formula:** `BIC = -2×log(likelihood) + k×log(n)`
  - pehla term: fit accha → kam
  - doosra term: zyada parameters → penalty
- **Kyun BIC:** Automatically best model pick karta hai — na underfitting na overfitting

**Step 5: Wild Bootstrap**
- **Kya:** Statistical technique jo confidence intervals deta hai
- **Kaise:** Residuals ko randomly flip karke (+/-) bahut baar fit karo → distribution milta hai
- **Kyun important:** Sirf "ye peak yahan hai" nahi, balki "ye peak yahan hai ± 0.02 ppm" — uncertainty quantify hoti hai

### Kyun padha ye paper?
- Classical baseline comparison ke liye
- **Limitations samajhne ke liye:**
  - Manually ROI define karna padta hai
  - Ek baar mein ek region — scalable nahi
  - Complex mixtures mein bahut slow
  - DL approach zyada generalizable hai

---

## Paper 4: Nature Communications — PLS Approach

### PLS Regression kya hai?
- **Partial Least Squares** — linear regression ka advanced version
- **Kya karta hai:** High-dimensional input (NOESY spectrum ke thousands of points) se high-dimensional output (CPMG spectrum) predict karta hai
- **Kaise:** Latent variables find karta hai jo input-output relationship best explain karte hain
- **Kya nahi:** Deep learning nahi — koi neural network nahi, koi training loop nahi. Sirf mathematical optimization.

### Dataset
- **1,753 serum spectra** — real clinical data
- **18 centers** — different hospitals/labs se (generalization ensure karne ke liye)
- **3 MetaboLights databases:** MTBLS242 (obesity), MTBLS395 (heart attack), MTBLS424 (breast cancer)
- **Split:** Training + Validation + Independent test set (alag center ka data)

### Preprocessing
1. **Alignment:** Anomeric glucose signal (δ 5.24 ppm) ko reference banao — sab spectra align karo
2. **Normalization:** Total area normalize — concentration differences remove karo
3. **Binning:** 0.001 ppm bins — matrix size reduce + small shift variations absorb

### Models
```
Model 1: NOESY → CPMG (small molecules)
Model 2: NOESY → Diffusion-edited (macromolecules)
Model 3: NOESY → pJRES (direct)
Model 4: predicted CPMG → pJRES (cascade — better!)
```

### Cascade Strategy ka Logic
- **Direct:** NOESY → pJRES → R² = 0.960
- **Cascade:** NOESY → CPMG → pJRES → R² = 0.968
- **Kyun better?** pJRES basically CPMG jaisa hai but J-coupling removed. Toh pehle CPMG predict karo (macro hatao), phir CPMG se pJRES predict karo (J-coupling hatao). Step by step easier hai ek sath se.

### Results (Table 1 se exact numbers)

**Independent Test Set (n=232):**

| Spectrum | R² | MRE% | RPD |
|---------|-----|------|-----|
| CPMG | 0.995 | 5.97% | 12.7 |
| Diffusion-edited | 0.998 | 3.80% | 17.7 |
| pJRES (cascade) | 0.968 | 12.6% | 5.03 |

**R² kya hai:** 1.0 = perfect prediction, 0.995 = 99.5% variance explained  
**MRE% kya hai:** Median Relative Error — har point pe kitna % error hai, uska median  
**RPD kya hai:** Ratio of Performance to Deviation — >3 = excellent prediction

### Key Takeaway for Us
Paper 4 ke authors ne khud likha hai (page mein):
> *"even in the age of complex AI models, simpler methods can be remarkably effective"*

**BUT** — PLS tab kaam karta hai jab:
- Mapping roughly linear ho ✅ (NOESY → CPMG linear-ish hai)
- Bahut saara real training data ho ✅ (1753 samples)

**Hamaare case mein:**
- Mixture → N individual compounds = **non-linear** problem ❌
- Real paired data abhi nahi hai ❌
- Isliye DL approach better hai hamare liye

---
---

# Section C: MixNet Architecture — Har Component

---

### Neural Network Basics (agar sir poochein)

**Neuron:** Input × Weight + Bias → Activation → Output  
**Layer:** Bahut saare neurons ek saath  
**Deep Learning:** Bahut saari layers stack karo → complex patterns seekh sakte hain  
**Training:** Data dikhao → prediction karo → error measure karo (loss) → weights adjust karo (backpropagation)

---

### Conv1D (1D Convolution)

- **Kya:** Ek small filter (kernel) jo spectrum pe slide karta hai
- **Kaise:** Har position pe filter × signal ka dot product → ek number niklta hai
- **Kyun:** Local patterns detect karta hai — peaks, shoulders, valleys
- **Kernel size = 3:** 3 adjacent points dekhta hai ek baar mein
- **2D nahi kyun:** Hamaara data 1D hai (spectrum = 1D signal), images 2D hote hain

```
Spectrum: [0.1, 0.5, 0.9, 0.5, 0.1]
Filter:   [1, 2, 1]

Position 1: 0.1×1 + 0.5×2 + 0.9×1 = 2.0
Position 2: 0.5×1 + 0.9×2 + 0.5×1 = 2.8
Position 3: 0.9×1 + 0.5×2 + 0.1×1 = 2.0
```

---

### BatchNorm (Batch Normalization)

- **Kya:** Har layer ke output ko normalize karo (mean=0, std=1)
- **Kyun:** Training stable aur fast hoti hai
- **Kaise:** Batch ke across mean aur variance calculate karo → normalize karo → learnable scale + shift lagao
- **Kya nahi:** Ye data preprocessing nahi hai — ye network ke andar hota hai, har layer ke baad

---

### ReLU (Rectified Linear Unit)

- **Kya:** Activation function — `f(x) = max(0, x)`
- **Kyun:** Non-linearity add karta hai — bina iske saari layers milke ek linear function ban jaayengi
- **Kaise:** Negative values → 0 kar do, positive values → jaisi hain
- **Kya nahi:** Output layer pe ReLU nahi lagaya — wahan Softplus hai (see below)

---

### Pooling (MaxPool1D, kernel=4, stride=4)

- **Kya:** Signal ko chhota karta hai — 4 points mein se max le lo → 1 point
- **Kyun:** 
  1. Computation reduce (chhota signal, faster processing)
  2. Receptive field badha (model zyada bada area ek saath dekh sake)
  3. Important features preserve (max value = dominant feature)
- **Stride=4:** 4 points skip karo — overlap nahi
- **Effect:** Har stage pe size 4× chhota: 16384 → 4096 → 1024 → 256 → 64 → 16

---

### Upsampling (scale_factor=4)

- **Kya:** Signal ko wapas bada karta hai
- **Kaise:** Nearest-neighbor interpolation — har value ko 4 baar repeat karo
- **Kyun:** Decoder mein compressed representation se wapas full-size spectrum banana hai
- **Baad mein:** Conv1D lagta hai jo repeated values ko smooth/refine karta hai

---

### Skip Connections

- **Kya:** Encoder ke output ko directly decoder ke corresponding stage ko bhejo
- **Kaise:** Concatenation — encoder features + decoder features join karo along channel dimension
- **Kyun zaroori:** 
  - Encoder compress karte waqt fine details (sharp peaks) kho jaate hain
  - Skip connections se decoder ko original details wapas milte hain
  - Without skip: output blurry/smooth hoga — peaks accurately reconstruct nahi honge
- **Example:**
  ```
  Encoder Stage 1 output (64 channels, 16384 points) ──→ Decoder Stage 1 ko directly milta hai
  ```

---

### Bottleneck

- **Kya:** Encoder aur Decoder ke beech ka narrowest point
- **Hamaare mein:** 512 channels × 16 points
- **Kyun important:** Yahan spectrum ka **most compressed representation** hota hai
- **Analogy:** Poora spectrum ka "summary" — sabse important patterns yahan encoded hain

---

### Softplus Output Activation

- **Kya:** `f(x) = log(1 + exp(x))` — smooth, always positive function
- **Kyun:** NMR spectrum ki intensity kabhi negative nahi hoti — Softplus ye guarantee karta hai
- **ReLU kyun nahi:** ReLU mein x=0 pe kink hai → gradient exactly 0 ho jata hai → dead neurons. Softplus smooth hai, gradient kabhi 0 nahi hota.
- **Sigmoid kyun nahi:** Sigmoid output 0 se 1 ke beech limit karta hai — spectra ki intensity 1 se zyada bhi ho sakti hai

---

### 1×1 Convolution (Final Layer)

- **Kya:** Kernel size = 1 ka convolution — har point independently process hota hai
- **Kyun:** Channel dimension change karna — decoder ke 64 channels → N output channels (N compounds)
- **Kaise:** Har spectral point pe 64 features ka weighted sum → N outputs
- **Kya nahi:** Ye spatial/spectral features extract nahi karta — sirf channel mixing karta hai

---

### Parameters Count: 7.1 Million

- **Kya matlab:** Model ke 7.1 million adjustable weights hain
- **Perspective:** SENNet bhi similar order — ye reasonable size hai 1D U-Net ke liye
- **Zyada kyun nahi:** Overfitting risk — zyada parameters = zyada memorization chance
- **Kam kyun nahi:** Complex overlapping patterns seekhne ke liye sufficient parameters chahiye

---

### Channels kya hain?

- **Kya:** Ek layer mein kitne different filters hain
- **Analogy:** Ek channel = ek "lens" — alag alag patterns detect karta hai
- **Encoder mein badhte hain (64→512):** Deeper = more complex patterns, zyada channels chahiye
- **Decoder mein ghatte hain (512→64):** Wapas simple representation pe aao

---
---

# Section D: Loss Function — Full Detail

---

### MSE (Mean Squared Error)

- **Kya:** Predicted aur true value ka squared difference ka average
- **Formula:** `MSE = (1/n) × Σ(predicted - true)²`
- **Kyun squared:** Bade errors ko zyada penalize karta hai (error=2 → penalty=4, error=10 → penalty=100)
- **Kyun MSE:** Simple, differentiable, widely used — training ke liye gradient easily milta hai

---

### Hamaara Dual Loss System

**Loss 1 — Spectral Loss (weight = 1.0)**
```
L_spectral = MSE(predicted_compound_spectra, true_compound_spectra)
```
- **Kya measure karta hai:** Kya har predicted compound sahi dikh raha hai?
- **Kyun primary (weight=1.0):** Ye hamaara main objective hai

**Loss 2 — Reconstruction Loss (weight = 0.5)**
```
reconstructed_mixture = Σ(predicted_compounds)  [sab compounds add karo]
L_recon = MSE(reconstructed_mixture, original_input_mixture)
```
- **Kya measure karta hai:** Kya predicted compounds ko add karne se original mixture wapas banta hai?
- **Kyun secondary (weight=0.5):** Ye physics constraint hai, direct objective nahi
- **Kyun zaroori:** Bina iske model aisi compounds predict kar sakta hai jo individually sahi dikhen but physics violate karein (sum ≠ mixture)

**Total Loss:**
```
L = 1.0 × L_spectral + 0.5 × L_recon
```

---

### SENNet ki Loss se Comparison

| | SENNet | MixNet |
|--|--------|--------|
| Primary | NMSE | Spectral MSE |
| Regularizer | TVE (smoothness) | Reconstruction MSE (physics) |
| TVE kyun nahi? | Unka output smooth macro spectrum | Hamaara output sharp compound peaks — TVE penalize karega |

---

### Masking

- **Problem:** Batch mein sab samples ka M aur N alag hai. Batching ke liye same size chahiye.
- **Solution:** Padding — chhote samples mein zero-filled extra channels add karo
- **New problem:** Padded (zero) channels pe loss calculate karna galat hai
- **Solution:** Masks — binary tensor jo batata hai kaunse channels real hain (1) aur kaunse padding (0)
- **compound_mask:** [1, 1, 1, 0, 0] → pehle 3 compounds real, baaki 2 padding
- **mixture_mask:** [1, 1, 1, 1, 0] → pehle 4 mixtures real, last 1 padding

---
---

# Section E: Data Pipeline — Full Detail

---

### Synthetic Data Generation

**Kyun synthetic?**
- Real paired data (mixture + individual compounds) abhi available nahi
- Synthetic data se controlled experiments ho sakte hain — N, M, concentrations sab set kar sakte hain
- SENNet ne bhi synthetic data use kiya aur kaam kar gaya

**Process:**
1. Compound library se N compounds select karo
2. Har compound ke liye 1–8 Gaussian peaks generate karo (random amplitude, width)
3. M mixtures banao — har mixture mein alag random concentrations
4. `mixture[j] = Σ(concentration[j][i] × compound[i])` for all i
5. Gaussian noise add karo (σ = 0.005)

**Gaussian Peak Formula:**
```
peak(x) = amplitude × exp(-((x - center)² / (2 × width²)))
```
- **center:** ppm position (e.g., 3.55 for glycine)
- **amplitude:** signal strength
- **width:** peak ki chaudai (σ)

---

### 16384 Points

- **Kyun 16k:** Sir ne 16k confirm kiya
- **Kyun 16384 specifically:** 2¹⁴ — power of 2
- **Kyun power of 2 zaroori:**
  - 5 pooling stages (÷4 each): 16384 → 4096 → 1024 → 256 → 64 → 16
  - Sab whole numbers — koi fractional size nahi
  - Agar 16000 hota: 16000 ÷ 4 = 4000, 4000 ÷ 4 = 1000, 1000 ÷ 4 = 250, 250 ÷ 4 = 62.5 ❌

---

### Real Data Loader (Sir ka CSV)

- **Kya:** Sir ne jo blood mixture CSV diya, uske liye special loader banaya
- **Process:**
  1. pandas se CSV load karo
  2. ppm column identify karo
  3. -1.0 to 10.0 range crop karo (raw data mein wider range hoti hai)
  4. 16384 points pe resample karo (scipy.interpolate.interp1d)
  5. 5 replicates handle karo (multiple measurements average)
  6. PyTorch tensor mein convert karo

---

### DataLoader

- **Kya:** PyTorch ka built-in class jo data batches mein serve karta hai
- **batch_size = 16:** Ek baar mein 16 samples process karo
- **shuffle = True:** Har epoch mein data ka order random karo (overfitting reduce)
- **Kyun batching:** Poora dataset ek saath GPU memory mein nahi aata — batches mein karo

---
---

# Section F: Training Process — Full Detail

---

### AdamW Optimizer

- **Adam kya hai:** Adaptive Moment Estimation — har parameter ke liye alag learning rate
- **W kya hai:** Weight Decay — weights ko slightly 0 ki taraf push karo (regularization)
- **lr = 0.0001:** Learning rate — kitna bada step lo har update mein (bahut chhota = slow but stable)
- **weight_decay = 0.0001:** Kitna regularize karo (bahut zyada = underfitting)

---

### Cosine Annealing LR + Warmup

- **Warmup (5 epochs):** Learning rate 0 se 0.0001 tak slowly badhao
  - **Kyun:** Shuruat mein random weights hain — bada lr = unstable
- **Cosine Annealing:** Warmup ke baad lr cosine curve pe slowly decrease hota hai
  - **Kyun:** Training ke end mein chhota lr = fine-tuning, precise convergence

```
Epoch:  1    5    50   100   200
LR:    0 → 0.0001 → slowly decreasing → ~0
```

---

### Early Stopping

- **Kya:** Validation loss improve nahi ho raha toh training band karo
- **patience = 20:** 20 epochs tak no improvement → stop
- **min_delta = 0.0001:** Improvement 0.0001 se kam hai toh count nahi hoga
- **Kyun:** Overfitting prevention — training loss decrease ho raha hai but validation loss increase = memorizing, not learning

---

### Epoch kya hai?

- **1 epoch = poora dataset ek baar dekha**
- **max 200 epochs:** But early stopping se pehle ruk jaayega likely
- **Har epoch mein:** Data shuffle → batches banao → forward pass → loss → backward pass → weight update → repeat

---

### Backpropagation

- **Kya:** Loss function ka gradient (derivative) calculate karo har weight ke respect mein
- **Kaise:** Chain rule — output se input tak layer by layer gradient flow hota hai
- **Kyun:** Ye batata hai ki kaunsa weight kitna aur kis direction mein change karna hai loss kam karne ke liye
- **Automatically hota hai:** PyTorch mein `loss.backward()` ek line se sab ho jaata hai

---
---

# Section G: Expected Questions & Answers

---

**Q: Ye kaam already ho chuka hai kya? Novel kya hai?**
A: SENNet ne blood-specific 2-class separation kiya (macro vs small). Hamaara MixNet **domain-agnostic** hai — koi bhi NMR mixture do (blood, plant, drug), N unknown compound spectra nikalega. Plus compound naming nahi — pure Blind Source Separation. Novel compound discovery possible!

**Q: Gaussian peaks kyun? NMR peaks Lorentzian hote hain.**
A: Aapne confirm kiya tha Gaussian use karna hai. Real data aane pe agar Voigt/Lorentzian better fit kare toh switch kar sakte hain. Architecture independent hai peak shape se.

**Q: Model kitne compounds handle karega?**
A: Currently 2-10 configured hai. Architecture flexible hai — in_channels aur out_channels change karo. Practical limit GPU memory pe depend karega.

**Q: Real data pe kaise test karoge?**
A: Aapne jo blood mixture CSV diya — wo already hamaare pipeline mein load ho raha hai 16384 points pe. Trained model pe test karenge. Baad mein plant extracts ya aur kisi sample pe bhi test kar sakte hain.

**Q: Agar results acche nahi aaye?**
A: Iterative approach — 1) synthetic data quality improve karo, 2) architecture tune karo, 3) loss function modify karo, 4) more data. Research mein iteration expected hai.

**Q: Training time kitna?**
A: GPU pe ~1-2 hours estimated (200 epochs, 5000 training samples). CPU pe significantly slower.

**Q: SENNet ka code kyun nahi use kiya?**
A: Publicly available nahi hai. Paper mein architecture describe ki thi — wahi inspiration lekar apna model banaya. Plus hamaara problem fundamentally different hai — domain-agnostic, N unknown compounds — toh direct use nahi ho sakta tha.

**Q: Overfitting kaise handle karoge?**
A: Multiple levels pe — weight decay (AdamW), early stopping (patience=20), reconstruction loss as implicit regularizer, aur synthetic data augmentation se diverse training samples.

**Q: Paper 4 mein PLS accha kaam kar raha hai, toh DL kyun?**
A: PLS linear method hai — NOESY→CPMG roughly linear mapping hai. Mixture → N unknown compounds non-linear problem hai. Plus PLS domain-specific hai (sirf blood serum pe trained), hamaara model general-purpose hai.

**Q: Skip connections hata dein toh kya hoga?**
A: Output blurry ho jaayega — sharp peaks reconstruct nahi honge. U-Net ki power skip connections mein hai. Bina skip ke ye sirf ek autoencoder ban jaayega.

**Q: Dual loss mein weights kaise decide kiye?**
A: Spectral loss primary objective hai (weight=1.0), reconstruction loss physics constraint hai (weight=0.5). Hyperparameter tuning se optimize kar sakte hain. 0.5 reasonable starting point hai — literature mein similar ratios use hote hain.

**Q: Sirf blood ke liye hai kya?**
A: Bilkul nahi! Blood ek testing use case hai. Model domain-agnostic hai — koi bhi NMR mixture spectrum do. Plant extracts, drug formulations, food quality — sab pe kaam karega.

**Q: New compounds discover kar sakte ho?**
A: Potential hai. Agar model ek separated spectrum nikale jo kisi existing NMR database (HMDB, BMRB) mein match nahi karta — wo potentially unknown/novel compound ho sakta hai. Especially useful for natural product research.

**Q: Model compounds ko naam deta hai?**
A: Nahi — model sirf spectra alag karta hai. Identification alag step hai — separated spectrum ko database se match karke identify kar sakte hain. Model ka kaam sirf decomposition hai.


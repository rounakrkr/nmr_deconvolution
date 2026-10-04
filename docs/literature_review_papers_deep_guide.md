# 📚 Papers — Deep Understanding Guide
### *Ekdum shuru se, hamaare project ke context mein*

---

# 🎯 Pehle Samjho — Hum Kya Bana Rahe Hain?

Imagine kar — tu ek **audio engineer** hai. Tere paas ek audio file hai jisme **5 singers ek saath gaa rahe hain** (ek mixed track). Tera kaam hai — har singer ki awaaz **alag nikaalana**. Tujhe ye bhi nahi pata kaunse singers hain — bas **alag karna** hai. 🎵

Exactly yahi kaam hai hamaara — lekin music ki jagah **NMR spectra** ke saath.

```
Koi bhi NMR mixture spectrum (blood, plant, drug, food, anything!)
= Compound A ka signal + Compound B ka signal + Compound C ka signal + ...
     (sab overlap ho rahe hain ek doosre pe)
```

**Hamaara model (MixNet) kya karta hai:**
```
Input:  M mixture spectra (M = multiple mixed samples)
           ↓  ↓  ↓
        [MixNet — 1D U-Net Neural Network]
           ↓  ↓  ↓
Output: N individual compound spectra (UNNAMED!)
        (Compound 1 alag, Compound 2 alag, Compound 3 alag...)
```

**⚠️ Important distinction:** MixNet compounds **naam nahi deta** — sirf spectra alag karta hai! Ye **Blind Source Separation (BSS)** hai — model ko pata nahi kaunsa compound hai, bas untangle karna seekhta hai.

---

## 🌍 Ye General-Purpose Kyun Hai — aur Kyun Game-Changer?

Blood ka NMR sirf **ek use case** hai — known compounds, finite set. Easy mode. 😌

**But socho:**
- 🌿 **Plant extract** mein unknown compounds ho sakte hain jo kabhi identify nahi hue
- 💊 **Drug formulations** mein impurities detect karna
- 🍷 **Food/wine quality** mein adulteration find karna

Agar MixNet ek pure spectrum alag nikaal de jo **kisi database mein match nahi karta** →

**🤯 Potential NEW COMPOUND DISCOVERY!**

```
Plant extract spectrum → MixNet → 5 separated spectra
                                    ↓
                            Spectrum #3 matches nothing in database
                                    ↓
                            Possible NEW compound! 🧬💊
```

---

## 📍 Status Abhi:

```
✅ Model ready (7.1M parameters, 1D U-Net)
✅ Data pipeline ready (any CSV/spectrum input)
✅ Training code ready
⏳ Realistic training data ka wait
```

Jaise ek fully equipped gym ready hai — bas athlete (data) nahi aaya abhi! 🏋️

---
---

# 📄 Paper 2 — SENNet ⭐⭐⭐
### *Sabse important paper — hamaare model ka direct inspiration!*
**"Using neural networks to obtain NMR spectra of both small and macromolecules from blood samples in a single experiment"**  
*Communications Chemistry, 2024 — Chinese Academy of Sciences, Wuhan*

---

## 🔍 Problem Kya Tha?

Blood ka NMR spectrum le — usme **2 tarah ke signals** hote hain ek saath:

| Type | Example | Peak shape | Linewidth |
|------|---------|-----------|-----------|
| 🔬 **Small molecules** | Glycine, Creatinine, Lactate | Sharp, narrow | < 3.66 Hz |
| 🧬 **Macromolecules** | Albumin (protein), Lipids, Lipoproteins | Broad, wide | ≥ 3.66 Hz |

**Problem:** Dono ek doosre pe overlap karte hain! 😩 Ek spectrum mein sab milke ek jumbled mess ban jaata hai.

**Clinically kyun important hai dono ko alag karna?**
- Small molecules → metabolic health (glucose, amino acids)
- Macromolecules (lipids) → cardiovascular disease risk 🫀

Agar sirf ek nikala toh doosre ki info lost ho jaati hai.

---

## 🛠️ Pehle Log Kya Karte The?

**Option 1 — CPMG + Diffusion-edited experiments separately karo 🕐**
- CPMG experiment → small molecules dikhte hain
- Diffusion-edited experiment → macromolecules dikhte hain
- **Problem:** 2 extra experiments = extra time + extra cost. Large cohort studies mein nightmare.

**Option 2 — Sample preparation 🧪**
- Ultrafiltration ya protein precipitation se macromolecules hata do pehle
- **Problem:** Time-consuming + macromolecule information **completely lost** 😞

**Option 3 — SMolESY (2020 ka method) 📉**
- Mathematical differentiation se small molecules enhance karo
- **Problem:** Macromolecule info lost. Again.

**Common thread:** Koi bhi ek experiment se **dono simultaneously** nahi nikaal paa raha tha! 😤

---

## 💡 SENNet Ka Solution

Ek neural network jo:
1. **NOESY spectrum le** (jisme dono hain — small + macro)
2. **Macromolecule spectrum predict kare** directly
3. Phir simple subtraction: `Small molecules = NOESY - Predicted macro` ✂️

**Ek shot mein dono!** No extra experiments. 🎉

---

## 📊 Data Kaise Banaya? (THE Most Important Part)

Neural network ko train karne ke liye **paired data** chahiye:
- Input: Total spectrum (small + macro)
- Target: Sirf macromolecule spectrum

**Problem:** Real life mein ye pairs easily available nahi hain! 😰

**Solution: Synthetic data generate karo! 🏭**

### Step 1: Real spectrum se peak parameters extract karo
Ek real plasma NOESY spectrum liya. `scipy.signal` library use ki:
- `find_peaks()` → peaks kahan hain
- `peak_widths()` → har peak ki linewidth kitni hai

Har peak ke 3 numbers nikale:
- **Frequency (ν)** → peak kahan hai (ppm mein)
- **Linewidth** → peak kitni chaudi hai (Hz mein)
- **Amplitude** → peak kitni tall hai

### Step 2: Classify karo ✂️
```
Linewidth < 3.66 Hz  → Small molecule
Linewidth ≥ 3.66 Hz  → Macromolecule
```
*(3.66 Hz statistically determine ki thi — optimal threshold)*

### Step 3: Randomly perturb karo 🎲
Extracted parameters ko randomly adjust karo (defined range mein) → thousands of varied synthetic training samples bante hain.

### Step 4: FID equation se spectrum generate karo 🌊
```
FID(t) = Σ Amplitude × exp(2πi × frequency × t) × exp(-α × t)
```
Ye time-domain signal hai. Fourier Transform lagao → frequency-domain spectrum milta hai.

Noise bhi add kiya → real jaisa lagein.

### Step 5: Paired data ready! 🎯
- **Input X** = Total spectrum (small + macro combined)
- **Target Y** = Macromolecule spectrum only

**🔗 Hamaare kaam se connection:**
> Hum bhi synthetic data use kar rahe hain! Hamaare paas bhi real paired data nahi hai (mixture ka corresponding individual compound spectra). Hamaara approach similar hai — Gaussian peaks se synthetic spectra generate karo. Lekin hamaara data **domain-agnostic** hai — kisi specific sample type se tied nahi hai! 🌍

---

## 🏗️ SENNet Architecture

**1D U-Net** — medical imaging ke liye famous architecture ko 1D NMR ke liye adapt kiya! 🏥→🧪

```
NOESY Spectrum (input)
         ↓
    [Encoder] → compress → compress → compress
                                          ↓
                                    [Bottleneck]
                                          ↓
    [Decoder] ← expand ← expand ← expand
         ↓
Macromolecule Spectrum (output)
```

+ **Skip connections** 🔗 — encoder ke har stage se decoder ke corresponding stage ko direct connection. Fine details (sharp peaks) preserve karne ke liye.

**🔗 Hamaare kaam se connection:**
> **Hamaara MixNet SAME base architecture hai!** 1D U-Net, encoder-decoder, skip connections. Key differences:
> - Unka: 1 input → 1 output (blood-specific, 2 categories)
> - Hamaara: M inputs → N outputs (**general-purpose, unlimited compounds, any sample type**) 🚀

---

## 📉 Loss Function — Kyun Special Design Kiya?

Normal MSE hi kyun nahi? 🤔

Kyunki macromolecule spectrum **smooth** hona chahiye — broad peaks, **koi sharp spikes nahi**. Standard MSE se model kabhi kabhi "jagged" output deta hai.

**Unka loss:**
```
Loss = w × TVE + NMSE
```

- **NMSE** (Normalized Mean Squared Error) = prediction kitna sahi hai
- **TVE** (Total Variation Error) = `Σ|output[n] - output[n-1]|`
  → Agar output jagged hai → TVE zyada → penalty → model smooth output prefer karta hai 📈
- **w = 22** → 113 real serum samples pe test kiya, w=22 pe best result mila ✅

**🔗 Hamaare kaam se connection:**
> Hamaara MixNet **TVE use NAHI karta!** Kyunki hamaare output mein **sharp peaks expected hain** (individual compound spectra mein narrow peaks hoti hain). TVE lagaate toh model sharp peaks ko penalize karta — galat! 
>
> Hamaara loss: `1.0 × Spectral_MSE + 0.5 × Reconstruction_MSE`
> Reconstruction loss = physics consistency (predicted compounds ka sum = original mixture) ⚖️

---

## 📈 Results

- **92.6% Pearson correlation** ✅ — SENNet ke small molecule output vs real CPMG experiment
- 600 MHz + 700 MHz dono spectrometers pe kaam kiya 🔬
- Plasma + Serum dono pe
- PCA test: SENNet output se bana PCA plot real experiments jaisa → clinically meaningful grouping preserve ✅

---

## ⚡ SENNet vs MixNet — Key Differences

| | SENNet | MixNet (Hamaara) |
|--|--------|--------|
| **Scope** | Blood-specific 🩸 | **Any sample** 🌍 |
| **Problem** | 2 broad categories | **N unknown individual compounds** |
| **Input** | 1 NOESY spectrum | M mixture spectra |
| **Output** | Macromolecule spectrum | N individual compound spectra |
| **Compound naming** | Category (macro/small) | **No naming — pure BSS** |
| **TVE loss** | Haan (smooth output) | Nahi (sharp peaks expected) |
| **Discovery** | ❌ | ✅ Novel compounds possible |

---

# 📄 Paper 4 — Nature Communications ⭐⭐
### *Simple method, real data, amazing results — lesson hamare liye!*
**"Deriving three one dimensional NMR spectra from a single experiment through machine learning"**  
*Nature Communications, 2025 — University of Florence, Italy*

---

## 🔍 Problem Kya Tha?

Blood/serum NMR metabolomics mein standard = **4 experiments per sample** 😤:
1. **NOESY** — sab dikhata hai (baseline experiment)
2. **CPMG** — sirf small molecules
3. **Diffusion-edited** — sirf macromolecules
4. **pJRES** — overlapping peaks resolve karta hai (J-coupling separate)

Large clinical studies mein hundreds of samples → 4× time = **nightmare**! 💸

**Unka goal:** Sirf NOESY se baaki teen predict karo. 1 experiment → 3 additional spectra. 🎯

---

## 🧮 Approach: PLS Regression

**PLS = Partial Least Squares** — simple math, no neural networks! 😮

**Kyun DL nahi chose?** Unhone khud likha:
> *"even in the age of complex AI models, simpler methods can be remarkably effective. It echoes the wisdom of Occam's razor."*

NOESY → CPMG mapping roughly **linear** hai → PLS sufficient! 

**4 models:**
```
Model 1: NOESY → CPMG (small molecules)
Model 2: NOESY → Diffusion-edited (macromolecules)
Model 3: NOESY → pJRES (direct)
Model 4: Predicted CPMG → pJRES ← BETTER! (cascade)
```

**🌟 Cascade kyun better?**
pJRES = CPMG + J-coupling removed.  
Step 1: Macro hato (NOESY → CPMG)  
Step 2: J-coupling hato (CPMG → pJRES)  
Two simpler steps > one complex step! 🎯

---

## 📊 Dataset — Real aur Large!

- **1,753 real serum spectra** 🏥
- **18 different clinical centers** — different hospitals se (generalization test)
- **3 diseases:** Obesity, Heart attack, Breast cancer
- **Proper split:** Training → Validation → **Independent** test set (alag center ka data — model ne kabhi nahi dekha!)

---

## 🔬 Preprocessing Steps

1. **Alignment** — Glucose peak (δ 5.24 ppm) reference banaya → sab spectra align 📐
2. **Normalization** — Total area normalize → concentration differences remove 📊
3. **Binning** — 0.001 ppm bins → matrix chhota + small shifts absorb 📦

---

## 📈 Results (Exact numbers from Table 1)

**Independent test set (n=232):**

| Output | R² | MRE% |
|--------|-----|------|
| CPMG | **0.995** | 5.97% |
| Diffusion-edited | **0.998** | 3.80% |
| pJRES (cascade) | **0.968** | 12.6% |
| pJRES (direct) | 0.960 | 13.3% |

R² = 0.995 matlab 99.5% variance explain ho rahi hai. Near-perfect! 🎉

---

## 🤝 Paper 4 Paper 2 ko explicitly cite karta hai!

Interesting! Paper 4 ke reviewers ne mention kiya ki SENNet (Paper 2) ka comparison karo:

| | SENNet | This Paper |
|--|--------|------------|
| Training | Synthetic data | **Real** data |
| CPMG | Indirect (subtraction) | **Direct** prediction |
| pJRES | ❌ | ✅ |

---

## 🔗 Hamaare kaam se connection + lesson

**Lesson:** PLS real data pe amazing kaam karta hai. Lekin hamaara case fundamentally alag hai:

| | Paper 4 (PLS) | MixNet (Hamaara) |
|--|-------------|----------------|
| Scope | Blood-serum only 🩸 | **Any mixture** 🌍 |
| Real data | ✅ 1753 samples | ❌ Abhi nahi |
| Problem type | Linear mapping | **Non-linear BSS** |
| Output | 3 known spectrum types | **N unknown compounds** |
| Discovery | ❌ | ✅ Novel compounds possible |

PLS yahan kaam kiya kyunki mapping linear thi + real data tha. Hamaara problem non-linear hai + general-purpose hai → DL zaruri hai! 🤖

---

# 📄 Paper 3 — NMR-Onion 🧅
### *Classical approach — comparison ke liye padha*
**"NMR-Onion — a transparent multi-model based 1D NMR deconvolution algorithm"**  
*Heliyon, 2024 — Technical University of Denmark*

---

## 🔍 Problem Kya Tha?

1D NMR spectrum mein **hundreds of overlapping peaks** — unhe mathematically "fit" karke individual peaks nikalna = **deconvolution** 📉

30 saal se ye problem solve ho rahi thi. Existing approaches mein 3 issues:
1. 🤔 **Model selection:** Kaun sa peak shape use karein? Koi statistical proof nahi
2. 📊 **Uncertainty:** "Peak yahan hai" — but confidence interval kya? Koi estimate nahi
3. 🌐 **Generalization:** Ek spectrum pe kaam karta hai, doosre pe nahi

---

## 🧅 Naam "Onion" Kyun?

Jaise pyaz ki layers ek ek karke peelti hain — spectrum ke overlapping peaks ko **layer by layer** fit karke nikalo! 😄

---

## ⚙️ 5-Step Algorithm

### Step 1: Band-pass filter → ROI 📍
User ek "Region of Interest" specify karta hai. Filter sirf us region pe apply hota hai.  
**Kyun?** Poora spectrum ek saath fit karna computationally expensive.

### Step 2: Peak detection 🔍
**Savitzky-Golay derivative filter** use kiya:
- 1st derivative = zero crossing → peak ka exact position
- 2nd derivative → peak ki sharpness

### Step 3: 3 mathematical models simultaneously fit karo 📐

**Model 1 — Lorentzian (simplest):**  
Ideal NMR peak shape. No distortion assumed.
```
Decay = exp(-αt)
```

**Model 2 — Pseudo-Voigt (realistic):**  
Gaussian + Lorentzian ka mix:
```
Decay = (1-η)×exp(-αt) + η×exp(-αt²)
η=0 → pure Lorentzian | η=1 → pure Gaussian
```

**Model 3 — Power Law (most flexible):**
```
Decay = exp(-αt^β)
β=1 → Lorentzian | β>1 → stretched | β<1 → compressed
```

**+ Asymmetric skew:** Shimming errors se peaks kabhi kabhi asymmetric ho jaate hain.

### Step 4: BIC se best model choose karo 🏆
**BIC = Bayesian Information Criterion:**
```
BIC = -2×log(likelihood) + k×log(n)
```
- Pehla term: fit accha → score kam
- Doosra term: zyada parameters → penalty

Automatically best model choose hota hai — koi manual decision nahi! 🤖

### Step 5: Wild Bootstrap — uncertainty 📊
Residuals ko randomly +/- flip karke bahut baar refit karo → distribution milti hai → **confidence intervals**!

Result: "Peak **3.55 ± 0.02 ppm** pe hai" — with statistical evidence! 📏

---

## 🔗 Hamaare kaam se connection

| | NMR-Onion (Classical) | MixNet (Hamaara) |
|--|---------------------|----------------|
| Approach | Math fitting | Deep learning 🤖 |
| Scope | ROI by ROI | **Any full spectrum** |
| Speed | Slow 🐢 | **Fast** ⚡ |
| Training data | Not needed | Paired data chahiye |
| Uncertainty | ✅ Built-in | ❌ Abhi nahi |
| Discovery | Individual peaks | **Full compound spectra** |
| Scalability | ❌ Limited | ✅ **Any sample type** |

NMR-Onion individual peaks fit karta hai. Hamaara MixNet **full compound spectra** alag karta hai — zyada useful for identification + discovery! 🧬

---

# 📄 Paper 1 — DL in NMR (Review) 📖
### *Big picture — field ka overview*
**"Deep learning and its applications in nuclear magnetic resonance spectroscopy"**  
*Progress in NMR Spectroscopy, 2025 — Xiamen University*

---

## 🔍 Kya Hai Ye?

**Review paper** 📰 — khud koi naya method nahi. 2015 se 2024 tak ke saare DL+NMR papers survey karke categorize kiye.

**Kyun padha:** Hamaara kaam field mein kahan fit hota hai ye samajhne ke liye! 🗺️

---

## 🗺️ 5 Main Areas

### 1. 🔄 NUS Reconstruction
- **Problem:** Multi-dimensional NMR = hours to days of acquisition
- **Solution:** Sirf 20-25% data collect karo, DL rest reconstruct kare
- **Result:** 75-90% time save! ⏱️
- **Hamaare liye:** Directly relevant nahi (hum 1D ke saath hain)

### 2. 🔇 Denoising
- **Problem:** Dilute samples mein SNR poor
- **Solution:** Neural network se noise remove, signal preserve
- **Hamaare liye:** Indirectly relevant — hamaara model bhi noisy input handle karega

### 3. 🎯 Peak Picking & Deconvolution ← **HAMAARA AREA!**
- **Problem:** Manual peak picking = slow + expert bias
- **Solution:** DL-based automated deconvolution
- **MixNet yahan fit hota hai!** 🔥

### 4. 🔮 Chemical Shift Prediction (Forward)
- **Problem:** Structure → Spectrum predict karna
- **Solution:** GNN se milliseconds mein predict
- **Hamaare liye:** We do inverse (spectrum → compounds)

### 5. 🧬 Structure Elucidation (Inverse)
- **Problem:** Spectrum → molecular structure guess karna
- **Hamaare liye:** Related concept, different scope

---

## 🏗️ Architectures Table

| Architecture | Kab use hota hai | Hamaare mein? |
|-------------|----------------|-------------|
| **CNN/U-Net** ← | Spectral features, deconvolution | ✅ **Yes!** |
| RNN/LSTM | FID time series | ❌ |
| Transformer | Long-range dependencies | ❌ |
| GNN | Molecular graphs | ❌ |
| MLP | Simple regression | ❌ |

---
---

# 🧠 Quick Revision — Sir Kuch Bhi Pooche

## Papers pe sawaal

**"SENNet ka data synthetic tha, real pe kaam karta hai?"** 🤔  
✅ Haan! Synthetic data real spectrum ke parameters pe based tha. 92.6% Pearson on real serum prove karta hai.

**"Paper 4 mein simple PLS used, toh deep learning kyun?"** 🤔  
✅ PLS linear method hai — NOESY→CPMG linear hai. Hamaara problem **non-linear BSS** hai + general-purpose hai. Isliye DL.

**"NMR-Onion transparent hai, tumhara black box — drawback nahi?"** 🤔  
✅ Trade-off hai. NMR-Onion ROI by ROI slow hai, manual intervention chahiye. Hamaara ek shot mein complete decomposition → scalable for any sample type.

**"Ye papers tumhare kaam se kaise related hain?"** 🤔  
✅ Paper 2 = architecture inspiration (1D U-Net). Paper 4 = real data important lesson. Paper 3 = classical baseline comparison. Paper 1 = hamaara area established field mein.

---

## Hamaare Model pe sawaal

**"Tumhara model SENNet se alag kaise hai?"** 🤔  
✅ SENNet: blood-specific, 2 categories, naming karta hai (macro/small). Hamaara: **domain-agnostic, N unknown compounds, no naming — pure BSS.** Novel compound discovery possible!

**"Sirf blood ke liye hai?"** 🤔  
✅ **Bilkul nahi!** Blood ek use case hai. Plant extracts, drug formulations, food samples — koi bhi NMR mixture de do, model separate karega. General-purpose tool hai.

**"Training data kahan se aayega?"** 🤔  
✅ Synthetic generate kar rahe hain (Gaussian peaks, random compounds). Domain-agnostic data — kisi specific sample pe dependent nahi. Real data aaye toh fine-tune karenge.

**"Model compounds ko naam deta hai?"** 🤔  
✅ **Nahi!** Model sirf spectra alag karta hai. Baad mein separated spectrum ko NMR database (HMDB, BMRB) se match karke identify kar sakte hain — wo alag step hai.

**"New compounds discover ho sakte hain?"** 🤔  
✅ Haan! Agar MixNet ek separated spectrum nikale jo kisi existing database mein match nahi karta → potentially unknown compound! Especially useful for plant/natural product research. 🌿

**"7.1 million parameters — kyun itne?"** 🤔  
✅ 16,384 spectral points ka complex overlap pattern seekhne ke liye sufficient chahiye. Zyada = overfitting. Kam = underfitting. Ye balance reasonable hai.

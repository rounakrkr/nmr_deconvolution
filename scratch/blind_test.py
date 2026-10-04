"""
TRUE BLIND TEST — 5 completely NEW compounds the model has NEVER seen.
Creates synthetic spectra, makes 20 mixtures, tests the trained model.
"""
import os, sys, torch, numpy as np
sys.path.insert(0, r'c:\Extra Programs\Files\BioTech')

from src.models.unet1d import MixNet

# ============================================================
# STEP 1: Create 5 BRAND NEW compound spectra
# (Different peaks than any of the 30 training compounds)
# ============================================================

spectral_length = 16384
ppm = np.linspace(10.0, 0.0, spectral_length)

def make_peak(ppm_center, width=0.02, height=1.0):
    """Create a Gaussian peak at a specific ppm position."""
    return height * np.exp(-0.5 * ((ppm - ppm_center) / width) ** 2)

def make_multiplet(ppm_center, J_hz=7.0, n_lines=2, width=0.015, height=1.0):
    """Create a multiplet (doublet, triplet, etc.)."""
    ppm_per_hz = 10.0 / spectral_length * 100  # rough conversion
    spacing = J_hz * ppm_per_hz
    spectrum = np.zeros(spectral_length)
    offsets = np.linspace(-(n_lines-1)/2, (n_lines-1)/2, n_lines) * spacing
    for off in offsets:
        spectrum += make_peak(ppm_center + off, width, height / n_lines)
    return spectrum

# 5 completely new "unknown" compounds with unique peak patterns
new_compounds = {}

# Compound A: peaks at 0.85, 1.55, 5.30 (nothing like training compounds)
c_a = make_multiplet(0.85, n_lines=3, height=1.0) + \
      make_peak(1.55, height=0.6) + \
      make_peak(5.30, height=0.3, width=0.03)
c_a = c_a / c_a.max()
new_compounds['Unknown_A'] = c_a

# Compound B: peaks at 2.75, 6.85, 7.45
c_b = make_peak(2.75, height=1.0) + \
      make_multiplet(6.85, n_lines=2, height=0.8) + \
      make_peak(7.45, height=0.5)
c_b = c_b / c_b.max()
new_compounds['Unknown_B'] = c_b

# Compound C: peaks at 1.95, 3.85, 8.20
c_c = make_peak(1.95, height=0.7) + \
      make_multiplet(3.85, n_lines=4, height=1.0) + \
      make_peak(8.20, height=0.4, width=0.04)
c_c = c_c / c_c.max()
new_compounds['Unknown_C'] = c_c

# Compound D: peaks at 0.95, 4.55, 6.10
c_d = make_multiplet(0.95, n_lines=2, height=1.0) + \
      make_peak(4.55, height=0.5) + \
      make_peak(6.10, height=0.7)
c_d = c_d / c_d.max()
new_compounds['Unknown_D'] = c_d

# Compound E: peaks at 3.15, 5.70, 9.20
c_e = make_peak(3.15, height=1.0) + \
      make_multiplet(5.70, n_lines=3, height=0.6) + \
      make_peak(9.20, height=0.3, width=0.05)
c_e = c_e / c_e.max()
new_compounds['Unknown_E'] = c_e

print("5 NEW compounds created (NEVER seen by model!):")
for name, spec in new_compounds.items():
    peaks = ppm[np.where(spec > 0.2)[0]]
    print(f"  {name}: peaks near {[round(p,2) for p in [peaks.min(), peaks.max()]]} ppm")

# ============================================================
# STEP 2: Make 20 mixtures with random concentrations
# ============================================================
np.random.seed(777)
compound_names = list(new_compounds.keys())
compound_matrix = np.stack([new_compounds[n] for n in compound_names])  # (5, 16384)

# Random concentrations (each row sums to 1)
concentrations = np.random.dirichlet([1, 1, 1, 1, 1], size=20).astype(np.float32)  # (20, 5)

# Mix: X = A @ S
mixtures = concentrations @ compound_matrix  # (20, 16384)

# Normalize each mixture
for i in range(20):
    mixtures[i] = mixtures[i] / mixtures[i].max()

print(f"\n20 mixtures created with random concentrations.")
print(f"Concentration range: {concentrations.min():.3f} to {concentrations.max():.3f}")

# ============================================================
# STEP 3: Run through TRAINED model
# ============================================================
model = MixNet(in_channels=20, out_channels=5)
ckpt = torch.load(r'c:\Extra Programs\Files\BioTech\checkpoints\best_model_real.pth',
                  map_location='cpu', weights_only=True)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()

mix_tensor = torch.from_numpy(mixtures.astype(np.float32)).unsqueeze(0)  # (1, 20, 16384)
with torch.no_grad():
    pred = model(mix_tensor).squeeze(0).numpy()  # (5, 16384)

# ============================================================
# STEP 4: Compare predicted vs true
# ============================================================
print("\n" + "=" * 60)
print("  TRUE BLIND TEST RESULTS")
print("  (5 compounds model has NEVER seen in training)")
print("=" * 60)

# The model outputs 5 spectra, but order may be different
# Try all permutations to find best assignment
from itertools import permutations

true_compounds = compound_matrix  # (5, 16384)
# Normalize true compounds same way
for i in range(5):
    true_compounds[i] = true_compounds[i] / true_compounds[i].max()

best_corr = -1
best_perm = None

for perm in permutations(range(5)):
    corrs = []
    for i, j in enumerate(perm):
        c = float(np.corrcoef(true_compounds[i], pred[j])[0, 1])
        corrs.append(c)
    avg = np.mean(corrs)
    if avg > best_corr:
        best_corr = avg
        best_perm = perm
        best_corrs = corrs

print(f"\nBest compound assignment found:")
for i, j in enumerate(best_perm):
    print(f"  True {compound_names[i]} <-> Predicted #{j+1}: correlation = {best_corrs[i]:.4f}")

print(f"\nAverage correlation: {best_corr:.4f}")
print()

if best_corr > 0.9:
    print("VERDICT: MODEL CAN DO TRUE BLIND SEPARATION! (>90%)")
elif best_corr > 0.7:
    print("VERDICT: PARTIAL separation works. Needs improvement.")
elif best_corr > 0.5:
    print("VERDICT: Weak separation. Cross-mixture attention needed.")
else:
    print("VERDICT: Model mostly memorized training compounds. Cannot generalize well.")

"""
TRUE BLIND TEST — Compares V1 vs V2 on 5 UNSEEN compounds.
Run AFTER both models are trained.

Usage: python blind_test_both.py
"""
import os, sys, torch, numpy as np
from itertools import permutations
sys.path.insert(0, r'c:\Extra Programs\Files\BioTech')

from src.models.unet1d import MixNet
from src.models.unet1d_v2 import MixNetV2

# ============================================================
# Create 5 NEW compounds (NEVER seen by either model)
# ============================================================
spectral_length = 16384
ppm = np.linspace(10.0, 0.0, spectral_length)

def make_peak(center, width=0.02, height=1.0):
    return height * np.exp(-0.5 * ((ppm - center) / width) ** 2)

def make_multiplet(center, J_hz=7.0, n_lines=2, width=0.015, height=1.0):
    spacing = J_hz * (10.0 / spectral_length * 100)
    s = np.zeros(spectral_length)
    for off in np.linspace(-(n_lines-1)/2, (n_lines-1)/2, n_lines) * spacing:
        s += make_peak(center + off, width, height / n_lines)
    return s

compounds = {}
c_a = make_multiplet(0.85, n_lines=3, height=1.0) + make_peak(1.55, height=0.6) + make_peak(5.30, height=0.3, width=0.03)
compounds['A'] = c_a / c_a.max()

c_b = make_peak(2.75, height=1.0) + make_multiplet(6.85, n_lines=2, height=0.8) + make_peak(7.45, height=0.5)
compounds['B'] = c_b / c_b.max()

c_c = make_peak(1.95, height=0.7) + make_multiplet(3.85, n_lines=4, height=1.0) + make_peak(8.20, height=0.4, width=0.04)
compounds['C'] = c_c / c_c.max()

c_d = make_multiplet(0.95, n_lines=2, height=1.0) + make_peak(4.55, height=0.5) + make_peak(6.10, height=0.7)
compounds['D'] = c_d / c_d.max()

c_e = make_peak(3.15, height=1.0) + make_multiplet(5.70, n_lines=3, height=0.6) + make_peak(9.20, height=0.3, width=0.05)
compounds['E'] = c_e / c_e.max()

names = list(compounds.keys())
true_matrix = np.stack([compounds[n] for n in names])  # (5, 16384)

# Make 20 mixtures
np.random.seed(777)
conc = np.random.dirichlet([1,1,1,1,1], size=20).astype(np.float32)
mixtures = conc @ true_matrix
for i in range(20):
    mixtures[i] = mixtures[i] / mixtures[i].max()

mix_tensor = torch.from_numpy(mixtures.astype(np.float32)).unsqueeze(0)

# ============================================================
# Test function
# ============================================================
def test_model(model, model_name):
    model.eval()
    with torch.no_grad():
        pred = model(mix_tensor).squeeze(0).numpy()
    
    best_corr = -1
    best_perm = None
    for p in permutations(range(5)):
        corrs = [float(np.corrcoef(true_matrix[i], pred[j])[0,1]) for i,j in enumerate(p)]
        avg = np.mean(corrs)
        if avg > best_corr:
            best_corr = avg
            best_perm = p
            best_corrs = corrs
    
    print(f'\n  {model_name}:')
    for i, j in enumerate(best_perm):
        print(f'    Compound {names[i]} -> Pred #{j+1}: {best_corrs[i]:.4f}')
    print(f'    AVERAGE: {best_corr:.4f}')
    return best_corr

# ============================================================
# Load and test both models
# ============================================================
print('=' * 60)
print('  TRUE BLIND TEST: V1 vs V2')
print('  5 compounds NEITHER model has ever seen!')
print('=' * 60)

# V1
v1 = MixNet(in_channels=20, out_channels=5)
v1_ckpt = torch.load(r'c:\Extra Programs\Files\BioTech\checkpoints\best_model_real.pth',
                     map_location='cpu', weights_only=True)
v1.load_state_dict(v1_ckpt['model_state_dict'])
v1_score = test_model(v1, 'MixNet V1 (no attention)')

# V2
v2_path = r'c:\Extra Programs\Files\BioTech\checkpoints\best_model_v2.pth'
if os.path.exists(v2_path):
    v2 = MixNetV2(in_channels=20, out_channels=5)
    v2_ckpt = torch.load(v2_path, map_location='cpu', weights_only=True)
    v2.load_state_dict(v2_ckpt['model_state_dict'])
    v2_score = test_model(v2, 'MixNet V2 (with attention)')
    
    print('\n' + '=' * 60)
    print(f'  V1: {v1_score:.4f} vs V2: {v2_score:.4f}')
    diff = v2_score - v1_score
    if diff > 0:
        print(f'  V2 is BETTER by {diff:.4f} ({diff/v1_score*100:.1f}% improvement)')
    else:
        print(f'  V1 is still better by {-diff:.4f}')
    print('=' * 60)
else:
    print('\n  V2 model not trained yet! Run: python train_v2.py --epochs 100 --patience 25')

# V3
v3_path = r'c:\Extra Programs\Files\BioTech\checkpoints\best_model_v3.pth'
if os.path.exists(v3_path):
    from src.models.unet1d_v3 import MixNetV3
    v3 = MixNetV3(in_channels=20, out_channels=5)
    v3_ckpt = torch.load(v3_path, map_location='cpu', weights_only=True)
    v3.load_state_dict(v3_ckpt['model_state_dict'])
    v3_score = test_model(v3, 'MixNet V3 (true cross-mixture)')
    
    print('\n' + '=' * 60)
    print(f'  FINAL COMPARISON:')
    print(f'  V1 (baseline):        {v1_score:.4f}')
    print(f'  V2 (bottleneck attn): {v2_score:.4f}' if os.path.exists(v2_path) else '')
    print(f'  V3 (cross-mixture):   {v3_score:.4f}')
    print('=' * 60)
else:
    print('\n  V3 not trained yet! Run: python train_v3.py --epochs 100 --patience 15')

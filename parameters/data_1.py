"""
parameters/data_1.py  –  Dataset 1 configuration
"""
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).parent.parent / "data" / "data_1"
OUT_DIR  = DATA_DIR

ODE_CSV  = DATA_DIR / "csv_mkg" / "ode_trajectory.csv"
QR_CSV   = OUT_DIR  / "qr_ftle.csv"

# ── ODE system ───────────────────────────────────────────────────────────────
A, B, C, DD, E = 1, 1.3, 0.435, 0.1, 0
PARAMS             = (A, B, C, DD, E)
INITIAL_CONDITIONS = [0.1, -1.5, 0.0]
DT     = 1        
TTR    = 100      
N_TOTAL = 6000    

# ── Sliding window ───────────────────────────────────
T_WINDOW_STEPS = 500   
T_STEP         = 30    
T_WINDOW       = T_WINDOW_STEPS * DT   

# ── Poincaré map ─────────────────────────────────────────────────────────────
BINS          = 20
POINCARE_LAG  = 5
CONNECT_LINES = True
LINE_THICK    = 0

# ── ML-FTLE ──────────────────────────────────────────────────
EMB_DIM     = 8
KNN_NEIGH     = 5
H_FTLE_MAX    = 40
TEST_RATIO    = 0.30
SMOOTH_WINDOW = 20

# ── QR-Benettin ──────────────────────────────────────────────────────────────
RENORM_INTERVAL = 10
LLE_THRESH      = 0.005

# ── Poincaré basis & PLSR ─────────────────────────────────────
POINCARE_METHODS  = ["JSD", "SSIM", "HDF", "IOU"]
NN_POINCARE       = 6
MIN_REL_SEPARATION_PCT = 20.0

# ── Plot style ───────────────────────────────────────────────────────────────
LINE_WIDTH    = 1.4
GRID_ALPHA   = 0.25
DPI_PLOT     = 600
DPI_COMPOSITE = 600
METHOD_COLORS = {
    "JSD":  "#2196F3",
    "SSIM": "#4CAF50",
    "HDF":  "#FF5722",
    "IOU":  "#9C27B0",
}
COLORS_REP = ["#378ADD", "#1D9E75", "#D85A30", "#7F77DD"]

for _d in [
    OUT_DIR,
    OUT_DIR / "csv_mkg",
]:
    _d.mkdir(parents=True, exist_ok=True)
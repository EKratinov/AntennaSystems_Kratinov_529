import numpy as np
import math
import matplotlib.pyplot as plt

lam_cm = 3.4
lambd = lam_cm / 100.0
a = 13.0 / 100.0
b = 13.0 / 100.0

def sinc_abs(x):
    out = np.ones_like(x, dtype=float)
    nz = x != 0
    out[nz] = np.abs(np.sin(x[nz]) / x[nz])
    return out

theta = np.linspace(-np.pi/2, np.pi/2, 200001)
theta_deg = np.degrees(theta)
cfac = (1.0 + np.cos(theta)) / 2.0

u = np.pi * a / lambd * np.sin(theta)  # E-plane
v = np.pi * b / lambd * np.sin(theta)  # H-plane
FE = sinc_abs(u) * cfac
FH = sinc_abs(v) * cfac

def hpbw_two_sided(fvals, th):
    target = 1/np.sqrt(2)
    mid = len(fvals)//2
    iR = np.argmin(np.abs(fvals[mid:] - target)) + mid
    iL = np.argmin(np.abs(fvals[:mid] - target))
    return (float(np.degrees(th[iR] - th[iL])),
            float(np.degrees(th[iL])),
            float(np.degrees(th[iR])))

HPBW_H_num, tL_H, tR_H = hpbw_two_sided(FH, theta)
HPBW_E_num, tL_E, tR_E = hpbw_two_sided(FE, theta)

HPBW_E_theor = 53.0 * lambd / b
HPBW_H_theor = 80.0 * lambd / a

def first_null_deg(D):
    if D <= lambd: return None
    return float(np.degrees(np.arcsin(min(1.0, lambd/D))))

null_H = first_null_deg(b)
null_E = first_null_deg(a)

def first_sidelobe_level_db(fvals, th_deg, guard_deg=5.0):
    f = np.asarray(fvals)
    th = np.asarray(th_deg)
    mask = (np.abs(th) > guard_deg)
    idx = np.where((f[1:-1] > f[:-2]) & (f[1:-1] > f[2:]) & mask[1:-1])[0] + 1
    if len(idx) == 0: return None
    i = idx[np.argmax(f[idx])]
    lvl_db = 20*np.log10(max(1e-12, f[i]))
    return float(lvl_db), float(th[i])

SLL_H = first_sidelobe_level_db(FH, theta_deg)
SLL_E = first_sidelobe_level_db(FE, theta_deg)

D_est_hpbw = 32400.0 / (HPBW_E_theor * HPBW_H_theor)
D_est_apert = 4*np.pi*a*b / (lambd**2)

print(f"λ = {lambd:.4f} м   a×b = {a*100:.0f}×{b*100:.0f} см")
print(f"HPBW(H) чисельно ≈ {HPBW_H_num:.2f}°  | теор. (80λ/a) ≈ {HPBW_H_theor:.2f}°")
print(f"HPBW(E) численно ≈ {HPBW_E_num:.2f}°  | теор. (53λ/b) ≈ {HPBW_E_theor:.2f}°")
if null_H: print(f"Перший нуль (H) ≈ ±{null_H:.2f}°")
if null_E: print(f"Перший нуль (E) ≈ ±{null_E:.2f}°")
if SLL_H: print(f"Перший бічний пелюсток H: {SLL_H[0]:.2f} дБ @ θ≈{SLL_H[1]:.1f}°")
if SLL_E: print(f"Перший бічний пелюсток E: {SLL_E[0]:.2f} дБ @ θ≈{SLL_E[1]:.1f}°")
print(f"Спрямованість (оцінка по ШГП) D ≈ {D_est_hpbw:.1f}")
print(f"Спрямованість (апертурна)      D ≈ {D_est_apert:.1f}")

plt.figure(figsize=(10,5))
plt.plot(theta_deg, FH, label=r"$F_H(\theta)$", linewidth=1.0)
plt.plot(theta_deg, FE, label=r"$F_E(\theta)$", linewidth=1.0)
plt.axhline(1/np.sqrt(2), linestyle="--", linewidth=0.6)
plt.plot([tL_H, tR_H], [1/np.sqrt(2), 1/np.sqrt(2)], 'go', markersize=4, label="−3 дБ H")
plt.plot([tL_E, tR_E], [1/np.sqrt(2), 1/np.sqrt(2)], 'ro', markersize=4, label="−3 дБ E")

hpbwE_half = HPBW_E_theor/2
hpbwH_half = HPBW_H_theor/2
for x in (-hpbwE_half, hpbwE_half):
    plt.axvline(x, ymin=0, ymax=0.72, linestyle="--", linewidth=0.6)
for x in (-hpbwH_half, hpbwH_half):
    plt.axvline(x, ymin=0, ymax=0.72, linestyle=":", linewidth=0.6)

plt.xlim(-90, 90)
plt.ylim(-0.02, 1.02)
plt.grid(True, linestyle="--", linewidth=0.3)
plt.xlabel("θ, град")
plt.ylabel("|F(θ)| (нормовані)")
plt.title("Нормовані ДС пірамідального рупора (площини E та H), з множником (1+cosθ)/2")
plt.legend()
plt.tight_layout()
plt.savefig("LR3_EH.png", dpi=300, bbox_inches="tight")
plt.show()


import numpy as np
import matplotlib.pyplot as plt
import math
import os

lam_cm = 4.2
lam = lam_cm / 100

l_cm = 23.7
l = l_cm / 100

d_cp_cm = 2.3
d = d_cp_cm / 100

h_cm = 12.0
h = h_cm / 100

xi = 1.0 + lam / (2.0 * l)

theta = np.linspace(-math.pi/2, math.pi/2, 200001)
theta_deg = np.degrees(theta)

s = np.sin(theta)
c = np.cos(theta)

u = math.pi * l / lam * (xi - c)
Fb = np.ones_like(theta)
nz = np.abs(u) > 1e-8
Fb[nz] = np.sin(u[nz]) / u[nz]

FE1_raw = Fb * c
FH1_raw = Fb

FC = np.cos(math.pi * h / lam * s)

FE2_raw = Fb * FC * c
FH2_raw = Fb * FC

def normalize(x):
    return np.abs(x) / np.max(np.abs(x))

FE1 = normalize(FE1_raw)
FH1 = normalize(FH1_raw)
FE2 = normalize(FE2_raw)
FH2 = normalize(FH2_raw)

def pattern_params(theta_deg, F_norm):
    target = 1 / math.sqrt(2)
    mid = len(F_norm) // 2

    i = mid
    while i < len(F_norm)-1 and F_norm[i] >= target:
        i += 1
    t1, t2 = theta_deg[i-1], theta_deg[i]
    f1, f2 = F_norm[i-1], F_norm[i]
    thR = t1 + (target - f1) * (t2 - t1) / (f2 - f1)

    i = mid
    while i > 0 and F_norm[i] >= target:
        i -= 1
    t1, t2 = theta_deg[i], theta_deg[i+1]
    f1, f2 = F_norm[i], F_norm[i+1]
    thL = t1 + (target - f1) * (t2 - t1) / (f2 - f1)

    HPBW = thR - thL

    peaks = []
    for k in range(1, len(F_norm)-1):
        if F_norm[k] > F_norm[k-1] and F_norm[k] > F_norm[k+1]:
            if abs(theta_deg[k]) > HPBW / 2:
                peaks.append(F_norm[k])

    SLL = None
    if peaks:
        peak = max(peaks)
        SLL = 20 * math.log10(peak)

    return HPBW, SLL

for name, F in [("Одностержнева, E-площина", FE1),
                ("Одностержнева, H-площина", FH1),
                ("Двостержнева, E-площина", FE2),
                ("Двостержнева, H-площина", FH2)]:
    HPBW, SLL = pattern_params(theta_deg, F)
    print(f"\n{name}:")
    print(f"  Ширина головної пелюстки за рівнем половинної потужності ≈ {HPBW:.2f}°")
    if SLL is not None:
        print(f"  Рівень першої бічної пелюстки ≈ {SLL:.2f} дБ")

os.makedirs("figures", exist_ok=True)

fig1, ax1 = plt.subplots(figsize=(10, 6))
ax1.plot(theta_deg, FE1, label="F_E1(θ), одностержнева")
ax1.plot(theta_deg, FH1, label="F_H1(θ), одностержнева")
ax1.set_xlabel("θ, градуси")
ax1.set_ylabel("Нормована F(θ)")
ax1.set_xlim(-90, 90)
ax1.set_ylim(0, 1.05)
ax1.grid(True)
ax1.legend()
fig1.tight_layout()
fig1.savefig("figures/DS_LR4_single_rod.png", dpi=600)

fig2, ax2 = plt.subplots(figsize=(10, 6))
ax2.plot(theta_deg, FE2, label="F_E2(θ), двостержнева")
ax2.plot(theta_deg, FH2, label="F_H2(θ), двостержнева")
ax2.set_xlabel("θ, градуси")
ax2.set_ylabel("Нормована F(θ)")
ax2.set_xlim(-90, 90)
ax2.set_ylim(0, 1.05)
ax2.grid(True)
ax2.legend()
fig2.tight_layout()
fig2.savefig("figures/DS_LR4_two_rod.png", dpi=600)

plt.show()

print(f"\nДовжина хвилі λ = {lam_cm:.2f} см")
print(f"Довжина стрижня l = {l_cm:.1f} см, середній діаметр d_cp = {d_cp_cm:.1f} см, відстань між стрижнями h = {h_cm:.1f} см")
print(f"Коефіцієнт уповільнення ξ ≈ {xi:.4f}")

import numpy
import matplotlib.pyplot as plt
import math
import os
from scipy.special import jv
from scipy.signal import find_peaks

lam = 0.029
D = 0.7
f = 0.3

R0 = D / 2
p = 2 * f
k = 2 * math.pi / lam
v = 3.5 * R0 / p

theta = numpy.linspace(0.0001, math.pi / 2, 4000)
theta_deg = numpy.degrees(theta)

u = k * R0 * numpy.sin(theta)

J0u = jv(0, u)
J1u = jv(1, u)
J2u = jv(2, u)

J0v = jv(0, v)
J1v = jv(1, v)
J1_15v = jv(1, 1.5 * v)
J2_15v = jv(2, 1.5 * v)

v2 = v ** 2
v15 = 1.5 * v
v15_2 = v15 ** 2
u2 = u ** 2
eps = 1e-9

term1_num = v * J1v * J0u - u * J1u * J0v
term1_den = v2 - u2
term1_den[numpy.abs(term1_den) < eps] = eps
term1 = term1_num / term1_den

term2 = numpy.empty_like(u)
mask0 = numpy.abs(u) < 1e-6
term2[mask0] = 0.5
term2[~mask0] = J1u[~mask0] / u[~mask0]

term3_num = u * J1u * J2_15v - v15 * J1_15v * J2u
term3_den = v15_2 - u2
term3_den[numpy.abs(term3_den) < eps] = eps
term3 = term3_num / term3_den

cos2 = numpy.cos(theta / 2) ** 2
norm = 0.74 * J1v / v + 0.13

FH = cos2 * (0.74 * term1 + 0.26 * term2 - 0.25 * term3) / norm
FE = cos2 * (0.74 * term1 + 0.26 * term2 + 0.25 * term3) / norm

FE_norm = numpy.abs(FE) / numpy.max(numpy.abs(FE))
FH_norm = numpy.abs(FH) / numpy.max(numpy.abs(FH))

idx_H = numpy.where(FH_norm >= 0.707)[0]
idx_E = numpy.where(FE_norm >= 0.707)[0]

SGP_H = 2 * theta_deg[idx_H[-1]] if len(idx_H) > 0 else 0.0
SGP_E = 2 * theta_deg[idx_E[-1]] if len(idx_E) > 0 else 0.0

print("Ширина головної пелюстки в площині H ≈", round(SGP_H, 3), "град")
print("Ширина головної пелюстки в площині E ≈", round(SGP_E, 3), "град")

peaks_E, _ = find_peaks(FE_norm, height=0.01)
mins_E, _ = find_peaks(-FE_norm)
peaks_H, _ = find_peaks(FH_norm, height=0.01)
mins_H, _ = find_peaks(-FH_norm)

print("\nМаксимуми в площині E (перші 5):")
for i in peaks_E[:5]:
    print(f"θ ≈ {theta_deg[i]:6.3f} град, |F_E| ≈ {FE_norm[i]:.4f}")

print("\nМінімуми в площині E (перші 5):")
for i in mins_E[:5]:
    print(f"θ ≈ {theta_deg[i]:6.3f} град, |F_E| ≈ {FE_norm[i]:.4f}")

print("\nМаксимуми в площині H (перші 5):")
for i in peaks_H[:5]:
    print(f"θ ≈ {theta_deg[i]:6.3f} град, |F_H| ≈ {FH_norm[i]:.4f}")

print("\nМінімуми в площині H (перші 5):")
for i in mins_H[:5]:
    print(f"θ ≈ {theta_deg[i]:6.3f} град, |F_H| ≈ {FH_norm[i]:.4f}")

fig, ax = plt.subplots(figsize=(20 / 2.54, 12 / 2.54))

ax.plot(theta_deg, FE_norm, linewidth=0.8, label="$F_E(\\theta)$")
ax.plot(theta_deg, FH_norm, linewidth=0.8, label="$F_H(\\theta)$")
ax.axhline(0.707, linestyle='--', linewidth=0.6, color='gray', label="Рівень 0.707")

ax.set_xlabel('θ, градуси', fontsize=10)
ax.set_ylabel('|F(θ)|, нормована', fontsize=10)
plt.xticks(numpy.arange(0, 91, 5), fontsize=7)
plt.yticks(numpy.arange(0, 1.1, 0.1), fontsize=7)
plt.ylim(-0.01, 1.01)
plt.xlim(0, 90.5)
plt.legend(loc="upper right", fontsize=8)
plt.grid(which='both', linestyle='--', linewidth=0.2, color='gray')

os.makedirs("figures", exist_ok=True)
fig.savefig("figures/ДС_дзеркальної_антени_E_H.png", dpi=600)

plt.show()

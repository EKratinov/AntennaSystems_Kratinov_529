import numpy as np
import matplotlib.pyplot as plt
import math
import os

lambd = 0.034
b = 0.13
ap = 0.13

theta_list = []
FE = []
FH = []

zeros_E = []
zeros_H = []
maxima_E = []
maxima_H = []

for theta in np.arange(0.0001, np.pi/2, 0.0001):
    s = np.sin(theta)
    c = np.cos(theta)
    xE = (np.pi * b / lambd) * s
    FE_theta = 1 if abs(xE) < 1e-7 else (np.sin(xE) / xE) * ((1 + c) / 2)
    xH = (np.pi * ap / lambd) * s
    denom = 1 - (2 * ap / lambd * s) ** 2
    FH_theta = 0 if abs(denom) < 1e-7 else (np.cos(xH) / denom) * ((1 + c) / 2)
    theta_deg = math.degrees(theta)
    theta_list.append(theta_deg)
    FE.append(abs(FE_theta))
    FH.append(abs(FH_theta))

for i in range(1, len(theta_list)-1):
    if FE[i] < 0.001 and FE[i-1] > 0.001:
        zeros_E.append((theta_list[i], FE[i]))
    if FH[i] < 0.001 and FH[i-1] > 0.001:
        zeros_H.append((theta_list[i], FH[i]))
    if FE[i] > FE[i-1] and FE[i] > FE[i+1]:
        maxima_E.append((theta_list[i], FE[i]))
    if FH[i] > FH[i-1] and FH[i] > FH[i+1]:
        maxima_H.append((theta_list[i], FH[i]))

fig, ax = plt.subplots(figsize=(10,6))
ax.plot(theta_list, FE, label="F_E(θ)", linewidth=1)
ax.plot(theta_list, FH, label="F_H(θ)", linewidth=1)

if zeros_E:
    ax.plot([pt[0] for pt in zeros_E], [pt[1] for pt in zeros_E], 'ro', markersize=4)
if zeros_H:
    ax.plot([pt[0] for pt in zeros_H], [pt[1] for pt in zeros_H], 'bo', markersize=4)
if maxima_E:
    ax.plot([pt[0] for pt in maxima_E], [pt[1] for pt in maxima_E], 'go', markersize=4)
if maxima_H:
    ax.plot([pt[0] for pt in maxima_H], [pt[1] for pt in maxima_H], 'mo', markersize=4)

ax.set_xlabel("θ (°)")
ax.set_ylabel("F(θ)")
ax.grid(True)
ax.legend()

os.makedirs("figures", exist_ok=True)
fig.savefig("figures/DS_LR3_13x13_points.png", dpi=600)

print(f"λ = {lambd:.4f} м")

print("\nНулі F_E(θ):")
for theta, val in zeros_E:
    print(f"θ = {theta:.2f}° , F_E = {val:.4f}")

print("\nнулі F_H(θ):")
for theta, val in zeros_H:
    print(f"θ = {theta:.2f}° , F_H = {val:.4f}")

print("\nМаксимуми F_E(θ):")
for theta, val in maxima_E:
    print(f"θ = {theta:.2f}° , F_E = {val:.4f}")

print("\nМаксимуми F_H(θ):")
for theta, val in maxima_H:
    print(f"θ = {theta:.2f}° , F_H = {val:.4f}")

if len(zeros_E) >= 2:
    width_FE = zeros_E[1][0] - zeros_E[0][0]
    print(f"\nШирина головної пелюстки F_E ≈ {width_FE:.2f}°")

if len(zeros_H) >= 2:
    width_FH = zeros_H[1][0] - zeros_H[0][0]
    print(f"Ширина головної пелюстки F_H ≈ {width_FH:.2f}°")

if len(maxima_E) >= 2:
    SLL_FE = 20 * math.log10(maxima_E[1][1] / maxima_E[0][1])
    print(f"\nРівень бокового пелюстка F_E (SLL) ≈ {SLL_FE:.2f} дБ")

if len(maxima_H) >= 2:
    SLL_FH = 20 * math.log10(maxima_H[1][1] / maxima_H[0][1])
    print(f"Рівень бокового пелюстка F_H (SLL) ≈ {SLL_FH:.2f} дБ")

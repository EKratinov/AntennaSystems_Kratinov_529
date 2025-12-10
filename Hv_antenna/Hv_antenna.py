import numpy as np
import matplotlib.pyplot as plt
import os


def element_pattern_H(theta):
    ct = np.cos(theta)
    F = np.zeros_like(theta, float)
    mask = np.abs(ct) > 1e-6
    F[mask] = np.cos((np.pi / 2) * np.sin(theta[mask])) / ct[mask]
    return F


def array_factor_H(N, d, theta, lam, lambda_hv, nu):
    k = 2 * np.pi / lam
    psi = k * d * np.sin(theta) - 2 * np.pi * d / lambda_hv + nu * np.pi
    den = np.sin(psi / 2)
    F = np.ones_like(theta, float)
    mask = np.abs(den) > 1e-6
    F[mask] = np.sin(N * psi[mask] / 2) / (N * den[mask])
    return F


def analyse_pattern(theta_deg, F_norm):
    idx0 = np.argmax(F_norm)
    level = 0.707

    left_part = F_norm[:idx0 + 1]
    left_idx = np.where(left_part <= level)[0]
    if left_idx.size == 0:
        theta_left = theta_deg[0]
    else:
        i2 = left_idx[-1]
        if i2 == len(left_part) - 1:
            theta_left = theta_deg[i2]
        else:
            i1 = i2
            i2 = i2 + 1
            theta_left = np.interp(
                level,
                [F_norm[i1], F_norm[i2]],
                [theta_deg[i1], theta_deg[i2]]
            )

    right_part = F_norm[idx0:]
    right_idx = np.where(right_part <= level)[0]
    if right_idx.size == 0:
        theta_right = theta_deg[-1]
    else:
        j2 = right_idx[0] + idx0
        j1 = j2 - 1
        theta_right = np.interp(
            level,
            [F_norm[j1], F_norm[j2]],
            [theta_deg[j1], theta_deg[j2]]
        )

    HPBW = theta_right - theta_left

    # поиск локальных максимумов для SLL
    loc_max = []
    for i in range(1, len(F_norm) - 1):
        if F_norm[i] > F_norm[i - 1] and F_norm[i] >= F_norm[i + 1]:
            loc_max.append(i)

    side_inds = [i for i in loc_max if F_norm[i] < 0.99]

    if not side_inds:
        return HPBW, np.nan

    SLL = np.max(F_norm[side_inds])
    SLL_dB = 20 * np.log10(SLL)
    return HPBW, SLL_dB


def find_local_extrema(F, theta_deg):
    maxima = []
    minima = []
    for i in range(1, len(F) - 1):
        if F[i] > F[i - 1] and F[i] >= F[i + 1]:
            maxima.append((theta_deg[i], F[i]))
        if F[i] < F[i - 1] and F[i] <= F[i + 1]:
            minima.append((theta_deg[i], F[i]))
    return maxima, minima


def main():
    os.makedirs("figures", exist_ok=True)

    lam = 2.9e-2          # λ = 2.9 см
    a = 23e-3             # широка стінка МЕК-100
    lambda_c = 2 * a
    lambda_hv = lam / np.sqrt(1 - (lam / lambda_c) ** 2)

    N = 14
    d = 2.0e-2
    nu = 1

    theta_deg = np.linspace(-90, 90, 20001)
    theta = np.deg2rad(theta_deg)

    F1H = element_pattern_H(theta)
    FHC = array_factor_H(N, d, theta, lam, lambda_hv, nu)
    F_H = F1H * FHC
    F_H = F_H / np.max(np.abs(F_H))
    F_abs = np.abs(F_H)

    HPBW_H, SLL_H_dB = analyse_pattern(theta_deg, F_abs)
    theta_max = theta_deg[np.argmax(F_abs)]

    maxima, minima = find_local_extrema(F_abs, theta_deg)

    print("ХвЩА: поздовжні щілини, шаховий порядок")
    print(f"N = {N}, d = {d*100:.1f} см, λ = {lam*100:.1f} см")
    print(f"Максимум ДС по H при θ ≈ {theta_max:.2f}°")
    print(f"Ширина головної пелюстки (0.707): HPBW ≈ {HPBW_H:.2f}°")
    print(f"Рівень бокових пелюсток (макс. з бічних): SLL ≈ {SLL_H_dB:.1f} dB")

    print("\nМаксимуми в площині H (включно з головним):")
    for th, val in maxima[:15]:
        print(f"θ ≈ {th:8.3f} град, |F_H| ≈ {val:.4f}")

    print("\nМінімуми в площині H:")
    for th, val in minima[:15]:
        print(f"θ ≈ {th:8.3f} град, |F_H| ≈ {val:.4f}")

    max_thetas = [m[0] for m in maxima]
    max_vals = [m[1] for m in maxima]
    min_thetas = [m[0] for m in minima]
    min_vals = [m[1] for m in minima]

    plt.figure()
    plt.plot(theta_deg, F_abs, label="|F_H(θ)|")
    plt.scatter(max_thetas, max_vals, c="red", s=20, label="Максимуми")
    plt.scatter(min_thetas, min_vals, c="black", s=15, label="Мінімуми")
    plt.grid(True)
    plt.xlabel("θ, град")
    plt.ylabel("|F_H(θ)|")
    plt.title("ДС ХвЩА в H-площині з максимумами і мінімумами")
    plt.legend()
    plt.savefig("figures/DS_H.png", dpi=300)

    F_E = np.ones_like(theta)
    plt.figure()
    plt.plot(theta_deg, F_E)
    plt.grid(True)
    plt.xlabel("θ, град")
    plt.ylabel("|F_E(θ)|")
    plt.title("ДС ХвЩА в E-площині")
    plt.savefig("figures/DS_E.png", dpi=300)

    plt.show()


if __name__ == "__main__":
    main()

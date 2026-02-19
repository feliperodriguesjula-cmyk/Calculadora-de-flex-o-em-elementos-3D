import numpy as np
import matplotlib.pyplot as plt
def ler_float(msg: str) -> float:
    txt = input(msg).strip().replace(",", ".")
    return float(txt)



# =========================
# DADOS DO MATERIAL (MPa)
# =========================
materiais = {
    "aco_1020": {"E": 210000, "fy": 350},
    "aco_1045": {"E": 210000, "fy": 530},
    "aluminio": {"E": 70000, "fy": 250}
}


# =========================
# ENTRADA DO USUÁRIO
# =========================

print("=== CÁLCULO DE FLEXÃO DE VIGA ===")

L = ler_float("Comprimento da viga (mm): ")
b = ler_float("Base da seção (mm): ")
h = ler_float("Altura da seção (mm): ")
P_kg = ler_float("Força aplicada (kgf): ")
material = input("Material (aco_1020 / aco_1045 / aluminio): ").strip().lower()
FS = ler_float("Fator de segurança: ")

# Conversão kgf → N
P = P_kg * 9.81

E = materiais[material]["E"]  # MPa
fy = materiais[material]["fy"]

# =========================
# CÁLCULOS
# =========================

# Momento de inércia (mm4)
I = (b * h**3) / 12

# Momento fletor máximo (N.mm)
Mmax = (P * L) / 4

# Tensão máxima (MPa)
sigma_max = (Mmax * (h/2)) / I

# Flecha máxima (mm)
delta_max = (P * L**3) / (48 * E * I)

# Tensão admissível
sigma_adm = fy / FS

# =========================
# RESULTADOS
# =========================

print("\n=== RESULTADOS ===")
print(f"Momento de inércia I = {I:.2f} mm4")
print(f"Momento máximo = {Mmax:.2f} N.mm")
print(f"Tensão máxima = {sigma_max:.2f} MPa")
print(f"Flecha máxima = {delta_max:.4f} mm")
print(f"Tensão admissível = {sigma_adm:.2f} MPa")

if sigma_max <= sigma_adm:
    print("STATUS: ELEMENTO VALIDADO")
else:
    print("STATUS: ELEMENTO NÃO VALIDADO")

# =========================
# GRÁFICO DE FLECHA
# =========================

x = np.linspace(0, L, 100)

# Equação da linha elástica (bi-apoiada carga central)
delta = (P * x * (L**3 - 2*L*x**2 + x**3)) / (48 * E * I)

plt.plot(x, delta)
plt.title("Deformação da Viga")
plt.xlabel("Comprimento (mm)")
plt.ylabel("Deslocamento (mm)")
plt.grid()
plt.show()

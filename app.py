import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# CONFIG
# ============================================================
st.set_page_config(page_title="Flexão de Vigas (CAD-like)", layout="wide")
st.title("Calculadora de FLEXÃO de Vigas — CAD-like (plano + ponto) | Saída em mm")

# ============================================================
# UNIDADES (motor SI; saída principal sempre mm)
# ============================================================
def mm_to_m(x_mm: float) -> float:
    return x_mm / 1000.0

def m_to_mm(x_m: float) -> float:
    return x_m * 1000.0

def MPa_to_Pa(x_MPa: float) -> float:
    return x_MPa * 1e6

# ============================================================
# MATERIAIS (lista aberta)
# ============================================================
DEFAULT_MATS = {
    "aco_1020":   {"E": 210e9, "fy": 350e6, "nu": 0.30},
    "aco_1045":   {"E": 210e9, "fy": 530e6, "nu": 0.29},
    "astm_a36":   {"E": 200e9, "fy": 250e6, "nu": 0.26},
    "inox_304":   {"E": 193e9, "fy": 215e6, "nu": 0.29},
    "inox_316":   {"E": 193e9, "fy": 205e6, "nu": 0.29},
    "al_6061_t6": {"E":  69e9, "fy": 275e6, "nu": 0.33},
    "al_5052":    {"E":  70e9, "fy": 193e6, "nu": 0.33},
    "nylon_pa6":  {"E": 2.7e9, "fy":  70e6, "nu": 0.39},
    "pla":        {"E": 3.5e9, "fy":  60e6, "nu": 0.36},
}

if "materials" not in st.session_state:
    st.session_state.materials = DEFAULT_MATS.copy()

# ============================================================
# PERFIS (exemplo interno mínimo)
# - no nível 2, o principal é importar Excel/CSV
# ============================================================
DEFAULT_PROFILES = pd.DataFrame([
    {"family":"RETANGULO",        "name":"Ret 100 x 10",         "b_mm":100.0, "h_mm":10.0},
    {"family":"BARRA REDONDA",    "name":"Barra Ø20",            "d_mm":20.0},
    {"family":"TUBO REDONDO",     "name":"Tubo Ø60,3 x 3,0",     "od_mm":60.3, "t_mm":3.0},
    {"family":"TUBO QUADRADO",    "name":"Tubo 50x50x3,0",       "b_mm":50.0,  "h_mm":50.0, "t_mm":3.0},
    {"family":"TUBO RETANGULAR",  "name":"Tubo 80x40x3,0",       "b_mm":80.0,  "h_mm":40.0, "t_mm":3.0},
])

# ============================================================
# SEÇÕES: Iy, Iz (m^4) + extremos (para tensão) + info do centro
# Convenção:
# - x ao longo da viga
# - seção: y (horizontal), z (vertical)
# ============================================================
def rect_Iy_Iz(b_m: float, h_m: float):
    Iy = (b_m * h_m**3) / 12.0
    Iz = (h_m * b_m**3) / 12.0
    yext = (-b_m/2, b_m/2)
    zext = (-h_m/2, h_m/2)
    return Iy, Iz, yext, zext, {"yc_m": 0.0, "zc_m": 0.0}

def round_solid_I(d_m: float):
    I = (np.pi/64.0) * d_m**4
    r = d_m/2
    yext = (-r, r)
    zext = (-r, r)
    return I, I, yext, zext, {"yc_m": 0.0, "zc_m": 0.0}

def round_tube_I(od_m: float, t_m: float):
    ro = od_m/2
    ri = max(ro - t_m, 0.0)
    di = 2*ri
    I = (np.pi/64.0) * (od_m**4 - di**4)
    yext = (-ro, ro)
    zext = (-ro, ro)
    return I, I, yext, zext, {"yc_m": 0.0, "zc_m": 0.0}

def rect_tube_I(b_m: float, h_m: float, t_m: float):
    bi = max(b_m - 2*t_m, 0.0)
    hi = max(h_m - 2*t_m, 0.0)
    Iy_out, Iz_out, _, _, _ = rect_Iy_Iz(b_m, h_m)
    if bi > 0 and hi > 0:
        Iy_in, Iz_in, _, _, _ = rect_Iy_Iz(bi, hi)
    else:
        Iy_in, Iz_in = 0.0, 0.0
    Iy = Iy_out - Iy_in
    Iz = Iz_out - Iz_in
    yext = (-b_m/2, b_m/2)
    zext = (-h_m/2, h_m/2)
    return Iy, Iz, yext, zext, {"yc_m": 0.0, "zc_m": 0.0}

def composite_layers_equiv(b_m: float, layers, E_ref: float):
    """
    Camadas empilhadas em z (altura), largura b constante.
    Método da seção transformada por E:
    Para flexão, usamos z̄ (eixo neutro) e Iy equivalente.
    """
    z0 = 0.0
    rects = []
    for lay in layers:
        t = lay["t"]
        E = lay["E"]
        n = E / E_ref
        rects.append({"b": b_m*n, "h": t, "zc": z0 + t/2})
        z0 += t

    Aeq = sum(r["b"]*r["h"] for r in rects)
    zbar = sum((r["b"]*r["h"])*r["zc"] for r in rects) / Aeq

    Iy_eq = 0.0
    for r in rects:
        b = r["b"]; h = r["h"]
        A = b*h
        Iy_local = (b*h**3)/12.0
        dz = r["zc"] - zbar
        Iy_eq += Iy_local + A*(dz**2)

    h_total = sum(lay["t"] for lay in layers)
    Iz_geom = (h_total * b_m**3) / 12.0

    yext = (-b_m/2, b_m/2)
    zext = (-h_total/2, h_total/2)

    return zbar, Iy_eq, Iz_geom, yext, zext, {"yc_m": 0.0, "zc_neutral_from_base_m": zbar, "h_total_m": h_total}

# ============================================================
# FEM Euler-Bernoulli 1D (flexão)
# DOF por nó: [w, theta]
# ============================================================
def beam_element_k(EI, Le):
    L = Le
    return (EI / L**3) * np.array([
        [ 12,   6*L, -12,   6*L],
        [6*L, 4*L**2, -6*L, 2*L**2],
        [-12, -6*L,  12,  -6*L],
        [6*L, 2*L**2, -6*L, 4*L**2],
    ], dtype=float)

def shape_N(Le, xi):
    L = Le
    N1 = 1 - 3*xi**2 + 2*xi**3
    N2 = L*(xi - 2*xi**2 + xi**3)
    N3 = 3*xi**2 - 2*xi**3
    N4 = L*(-xi**2 + xi**3)
    return np.array([N1, N2, N3, N4], float)

def udl_equiv_nodal_load(w, Le):
    L = Le
    return np.array([w*L/2, w*L**2/12, w*L/2, -w*L**2/12], dtype=float)

def apply_bc(K, F, fixed_dofs):
    all_dofs = np.arange(K.shape[0])
    free = np.setdiff1d(all_dofs, np.array(fixed_dofs, dtype=int))
    return free, K[np.ix_(free, free)], F[free]

def solve_beam_FEM(L, EI, apoio_esq, apoio_dir, loads, ne=140):
    """
    Retorna:
      xs, w(x), V(x), M(x), reactions_dict
    reactions_dict:
      - fixed_dofs
      - reactions vector (K d - F) at fixed dofs
    """
    x_nodes = np.linspace(0.0, L, ne+1)
    Le = L/ne
    nn = ne+1
    ndof = 2*nn

    K = np.zeros((ndof, ndof), float)
    F = np.zeros(ndof, float)

    # K global
    for e in range(ne):
        ke = beam_element_k(EI, Le)
        dofs = np.array([2*e, 2*e+1, 2*(e+1), 2*(e+1)+1])
        K[np.ix_(dofs, dofs)] += ke

    # UDL por elemento
    for e in range(ne):
        x0 = x_nodes[e]
        x1 = x_nodes[e+1]
        w_elem = 0.0
        for ld in loads:
            if ld["type"] == "w":
                a = ld["a"]; b = ld["b"]; w = ld["w"]
                if (x1 > a) and (x0 < b):
                    w_elem += w
        if abs(w_elem) > 0:
            feq = udl_equiv_nodal_load(w_elem, Le)
            dofs = np.array([2*e, 2*e+1, 2*(e+1), 2*(e+1)+1])
            F[dofs] += feq

    # Forças concentradas consistentes
    for ld in loads:
        if ld["type"] == "P":
            xP = float(np.clip(ld["x"], 0.0, L))
            P = ld["P"]
            e = min(int(xP / Le), ne-1)
            x0 = x_nodes[e]
            xi = (xP - x0) / Le
            N = shape_N(Le, xi)
            dofs = np.array([2*e, 2*e+1, 2*(e+1), 2*(e+1)+1])
            F[dofs] += P * N

        elif ld["type"] == "M":
            xM = float(np.clip(ld["x"], 0.0, L))
            M = ld["M"]
            i = int(round(xM / Le))
            i = int(np.clip(i, 0, nn-1))
            F[2*i + 1] += M  # momento nodal (theta)

    # BC
    fixed = []
    if apoio_esq == "Engastado":
        fixed += [0, 1]
    elif apoio_esq == "Pino (w=0)":
        fixed += [0]

    last_w = 2*(nn-1)
    last_t = last_w + 1
    if apoio_dir == "Engastado":
        fixed += [last_w, last_t]
    elif apoio_dir == "Pino (w=0)":
        fixed += [last_w]

    free, Kff, Ff = apply_bc(K, F, fixed)
    d = np.zeros(ndof, float)
    if len(free) > 0:
        d[free] = np.linalg.solve(Kff, Ff)

    # REAÇÕES: R = K d - F (somente nos DOFs fixos)
    Rfull = K @ d - F
    reactions = {int(fd): float(Rfull[fd]) for fd in fixed}

    # pós-processamento w, V, M
    xs, ws, Ms, Vs = [], [], [], []

    for e in range(ne):
        dofs = np.array([2*e, 2*e+1, 2*(e+1), 2*(e+1)+1])
        de = d[dofs]
        ke = beam_element_k(EI, Le)

        # feq do elemento (UDL)
        w_elem = 0.0
        x0 = x_nodes[e]
        x1 = x_nodes[e+1]
        for ld in loads:
            if ld["type"] == "w":
                a = ld["a"]; b = ld["b"]; w = ld["w"]
                if (x1 > a) and (x0 < b):
                    w_elem += w
        feq = udl_equiv_nodal_load(w_elem, Le) if abs(w_elem) > 0 else np.zeros(4)

        fint = ke @ de - feq  # [V1, M1, V2, M2]
        V1, M1, V2, M2 = fint

        for xi in np.linspace(0, 1, 6, endpoint=False):
            xg = x_nodes[e] + xi*Le
            N = shape_N(Le, xi)
            wg = float(N @ de)
            xs.append(xg)
            ws.append(wg)
            Ms.append(M1*(1-xi) + M2*xi)
            Vs.append(V1*(1-xi) + V2*xi)

    xs.append(L)
    ws.append(d[last_w])
    Ms.append(Ms[-1] if Ms else 0.0)
    Vs.append(Vs[-1] if Vs else 0.0)

    return np.array(xs), np.array(ws), np.array(Vs), np.array(Ms), reactions

# ============================================================
# A) VISUAL “SIMULATION-LIKE” (plano + ponto + seta)
# ============================================================
def draw_plane_preview(plane_key: str, x_m: float, coord_mm: float, L_m: float, sign: str):
    fig, ax = plt.subplots()
    ax.plot([0, L_m], [0, 0], linewidth=3)
    ax.scatter([x_m], [0], s=60)

    sgn = 1.0 if sign == "+" else -1.0

    if "±Z" in plane_key:
        ax.arrow(x_m, 0, 0, 0.25*sgn, head_width=0.03*L_m, head_length=0.05, length_includes_head=True)
        ax.text(x_m, 0.30*sgn, f"F ⟂ XY (±Z)\ncoord y={coord_mm:.1f} mm", ha="center", va="center")
        ax.add_patch(plt.Rectangle((x_m - 0.06*L_m, -0.08), 0.12*L_m, 0.16, fill=False, linestyle="--"))
    else:
        ax.arrow(x_m, 0, 0.25*sgn, 0, head_width=0.03*L_m, head_length=0.05, length_includes_head=True)
        ax.text(x_m + 0.28*sgn, 0.10, f"F ⟂ XZ (±Y)\ncoord z={coord_mm:.1f} mm", ha="center", va="center")
        ax.add_patch(plt.Rectangle((x_m - 0.06*L_m, -0.08), 0.12*L_m, 0.16, fill=False, linestyle="--"))

    ax.set_title("Preview do plano e direção da força (estilo Simulation)")
    ax.set_xlabel("x (m)")
    ax.set_yticks([])
    ax.grid(True, axis="x", alpha=0.25)
    ax.set_xlim(-0.02*L_m, 1.02*L_m)
    st.pyplot(fig)

# ============================================================
# B) “DESENHO DO APOIO” + reações
# ============================================================
def draw_supports(apoio_esq: str, apoio_dir: str, L_m: float):
    fig, ax = plt.subplots()
    ax.plot([0, L_m], [0, 0], linewidth=3)

    def symbol(x, tipo):
        if tipo == "Engastado":
            ax.plot([x, x], [-0.15, 0.15], linewidth=6)
        elif tipo == "Pino (w=0)":
            ax.scatter([x], [0], s=80)
            ax.plot([x-0.06*L_m, x, x+0.06*L_m], [-0.10, -0.18, -0.10], linewidth=2)
        else:
            ax.scatter([x], [0], s=40)

    symbol(0, apoio_esq)
    symbol(L_m, apoio_dir)

    ax.set_title("Apoios (visual)")
    ax.set_xlabel("x (m)")
    ax.set_yticks([])
    ax.grid(True, axis="x", alpha=0.25)
    ax.set_xlim(-0.02*L_m, 1.02*L_m)
    st.pyplot(fig)

def reactions_table(reac: dict):
    rows = []
    for dof, val in sorted(reac.items()):
        node = dof // 2
        kind = "w (N)" if dof % 2 == 0 else "theta (N·m)"
        rows.append({"DOF": dof, "Nó": node, "Tipo": kind, "Reação": val})
    return pd.DataFrame(rows)

# ============================================================
# SIDEBAR: unidades + edição de materiais + templates
# ============================================================
with st.sidebar:
    st.header("Unidades (entrada)")
    # >>> TROCA AQUI: mm como padrão
    unit_system = st.selectbox("Sistema de entrada", ["SI (m, N, Pa)", "mm (mm, N, MPa)"], index=1)
    unit_len = "m" if unit_system.startswith("SI") else "mm"
    unit_stress = "Pa" if unit_system.startswith("SI") else "MPa"
    st.caption("Motor interno sempre em SI. Saída principal sempre em mm.")

    st.divider()
    st.subheader("Materiais (lista aberta)")
    if st.checkbox("Adicionar/editar material"):
        with st.form("add_mat"):
            name = st.text_input("Nome do material (id)", value="meu_material")
            E_in = st.number_input(f"E ({'Pa' if unit_system.startswith('SI') else 'MPa'})",
                                   value=210000.0 if not unit_system.startswith("SI") else 210e9)
            fy_in = st.number_input(f"fy ({'Pa' if unit_system.startswith('SI') else 'MPa'})",
                                    value=350.0 if not unit_system.startswith("SI") else 350e6)
            nu_in = st.number_input("nu (Poisson)", value=0.30)
            ok = st.form_submit_button("Salvar")
            if ok:
                E_val = E_in if unit_system.startswith("SI") else MPa_to_Pa(E_in)
                fy_val = fy_in if unit_system.startswith("SI") else MPa_to_Pa(fy_in)
                st.session_state.materials[name.strip().lower()] = {"E": float(E_val), "fy": float(fy_val), "nu": float(nu_in)}
                st.success("Material salvo.")

    st.divider()
    st.subheader("Modelo de Excel (copiar e colar)")
    st.caption("Você pode criar um Excel com essas colunas e importar no app.")
    st.code(
        "perfis.xlsx (ou CSV):\n"
        "family,name,b_mm,h_mm,t_mm,d_mm,od_mm\n"
        "RETANGULO,Ret 100x10,100,10,,,\n"
        "BARRA REDONDA,Barra Ø20,,,,20,\n"
        "TUBO REDONDO,Tubo Ø60,3x3,0,,,,,60.3,3.0\n"
        "TUBO QUADRADO,Tubo 50x50x3,0,50,50,3.0,,\n"
        "TUBO RETANGULAR,Tubo 80x40x3,0,80,40,3.0,,\n"
    )

# ============================================================
# 1) GEOMETRIA, APOIOS E MATERIAL
# ============================================================
st.subheader("1) Geometria, apoios e material")

c1, c2, c3, c4 = st.columns([1.2, 1.0, 1.0, 1.2])
with c1:
    L_in = st.number_input(f"Comprimento L ({unit_len})", min_value=0.001,
                           value=2.0 if unit_len=="m" else 2000.0,
                           step=0.1 if unit_len=="m" else 50.0)
with c2:
    apoio_esq = st.selectbox("Apoio esquerdo", ["Engastado","Pino (w=0)","Livre"], index=1)
with c3:
    apoio_dir = st.selectbox("Apoio direito", ["Engastado","Pino (w=0)","Livre"], index=1)
with c4:
    material = st.selectbox("Material", sorted(list(st.session_state.materials.keys())), index=0)

FS = st.number_input("Fator de segurança (FS)", min_value=1.0, value=1.5, step=0.1)

L = mm_to_m(L_in) if unit_len=="mm" else float(L_in)
E = st.session_state.materials[material]["E"]
fy = st.session_state.materials[material]["fy"]

draw_supports(apoio_esq, apoio_dir, L)

# ============================================================
# 2) SEÇÃO TRANSVERSAL
# ============================================================
st.subheader("2) Seção transversal (Tabela / Custom / Composta / Import Excel)")

sec_mode = st.radio("Modo", ["Tabela interna", "Custom (dimensões)", "Seção composta (camadas)", "Importar Excel/CSV (catálogo)"], horizontal=True)

Iy = Iz = 0.0
yext = (0.0, 0.0)
zext = (0.0, 0.0)
sec_desc = ""
sec_center = {"yc_m": 0.0, "zc_m": 0.0}

if sec_mode == "Tabela interna":
    df = DEFAULT_PROFILES.copy()
    fam = st.selectbox("Família", sorted(df["family"].unique()))
    df2 = df[df["family"] == fam]
    name = st.selectbox("Perfil", df2["name"].tolist())
    row = df2[df2["name"] == name].iloc[0].to_dict()
    sec_desc = f"{fam} | {name}"

    if fam == "RETANGULO":
        Iy, Iz, yext, zext, sec_center = rect_Iy_Iz(mm_to_m(float(row["b_mm"])), mm_to_m(float(row["h_mm"])))
    elif fam == "BARRA REDONDA":
        Iy, Iz, yext, zext, sec_center = round_solid_I(mm_to_m(float(row["d_mm"])))
    elif fam == "TUBO REDONDO":
        Iy, Iz, yext, zext, sec_center = round_tube_I(mm_to_m(float(row["od_mm"])), mm_to_m(float(row["t_mm"])))
    elif fam in ["TUBO QUADRADO","TUBO RETANGULAR"]:
        Iy, Iz, yext, zext, sec_center = rect_tube_I(mm_to_m(float(row["b_mm"])), mm_to_m(float(row["h_mm"])), mm_to_m(float(row["t_mm"])))

elif sec_mode == "Custom (dimensões)":
    fam = st.selectbox("Família", ["RETANGULO", "BARRA REDONDA", "TUBO REDONDO", "TUBO RETANGULAR/QUADRADO"], index=0)
    sec_desc = f"Custom {fam}"

    if fam == "RETANGULO":
        b_in = st.number_input(f"b ({unit_len})", min_value=0.001, value=0.10 if unit_len=="m" else 100.0)
        h_in = st.number_input(f"h ({unit_len})", min_value=0.001, value=0.010 if unit_len=="m" else 10.0)
        b_m = mm_to_m(b_in) if unit_len=="mm" else float(b_in)
        h_m = mm_to_m(h_in) if unit_len=="mm" else float(h_in)
        Iy, Iz, yext, zext, sec_center = rect_Iy_Iz(b_m, h_m)

    elif fam == "BARRA REDONDA":
        d_in = st.number_input(f"Diâmetro d ({unit_len})", min_value=0.001, value=0.02 if unit_len=="m" else 20.0)
        d_m = mm_to_m(d_in) if unit_len=="mm" else float(d_in)
        Iy, Iz, yext, zext, sec_center = round_solid_I(d_m)

    elif fam == "TUBO REDONDO":
        od_in = st.number_input(f"OD ({unit_len})", min_value=0.001, value=0.0603 if unit_len=="m" else 60.3)
        t_in  = st.number_input(f"t ({unit_len})", min_value=0.0005, value=0.003 if unit_len=="m" else 3.0)
        od_m = mm_to_m(od_in) if unit_len=="mm" else float(od_in)
        t_m  = mm_to_m(t_in) if unit_len=="mm" else float(t_in)
        Iy, Iz, yext, zext, sec_center = round_tube_I(od_m, t_m)

    else:
        b_in = st.number_input(f"B ({unit_len})", min_value=0.001, value=0.10 if unit_len=="m" else 100.0)
        h_in = st.number_input(f"H ({unit_len})", min_value=0.001, value=0.05 if unit_len=="m" else 50.0)
        t_in = st.number_input(f"t ({unit_len})", min_value=0.0005, value=0.003 if unit_len=="m" else 3.0)
        b_m = mm_to_m(b_in) if unit_len=="mm" else float(b_in)
        h_m = mm_to_m(h_in) if unit_len=="mm" else float(h_in)
        t_m = mm_to_m(t_in) if unit_len=="mm" else float(t_in)
        Iy, Iz, yext, zext, sec_center = rect_tube_I(b_m, h_m, t_m)

elif sec_mode == "Seção composta (camadas)":
    st.caption("Camadas empilhadas em z (altura). Largura b constante. Ex.: 6,35 + 3 mm.")
    b_in = st.number_input(f"Largura b ({unit_len})", min_value=0.001, value=0.10 if unit_len=="m" else 100.0)
    b_m = mm_to_m(b_in) if unit_len=="mm" else float(b_in)

    n_layers = st.number_input("Quantidade de camadas", min_value=2, max_value=6, value=2, step=1)
    layers = []
    for i in range(int(n_layers)):
        cc1, cc2 = st.columns([1.2, 1.2])
        with cc1:
            t_in = st.number_input(f"Camada {i+1} — espessura ({unit_len})", min_value=0.0005,
                                   value=(6.35 if unit_len=="mm" and i==0 else 3.0 if unit_len=="mm" and i==1 else 0.00635 if unit_len=="m" and i==0 else 0.003),
                                   key=f"t_{i}")
        with cc2:
            mat_i = st.selectbox(f"Camada {i+1} — material", sorted(list(st.session_state.materials.keys())), index=0, key=f"m_{i}")
        t_m = mm_to_m(t_in) if unit_len=="mm" else float(t_in)
        layers.append({"t": t_m, "E": st.session_state.materials[mat_i]["E"]})

    E_ref = layers[0]["E"]
    zbar, Iy, Iz, yext, zext, sec_center = composite_layers_equiv(b_m, layers, E_ref)
    sec_desc = "Composta (camadas)"

    st.info(f"Eixo neutro z̄ (a partir da base): {m_to_mm(zbar):.4f} mm | Iy_eq={Iy:.3e} m⁴ | Iz≈{Iz:.3e} m⁴")

else:
    st.caption("Importe um catálogo Excel/CSV para perfis.")
    up = st.file_uploader("Enviar Excel/CSV de perfis", type=["xlsx","xls","csv"])
    if up is not None:
        if up.name.lower().endswith(".csv"):
            cat = pd.read_csv(up)
        else:
            cat = pd.read_excel(up)

        st.dataframe(cat, width="stretch")

        fam = st.selectbox("Família", sorted(cat["family"].unique()))
        cat2 = cat[cat["family"] == fam]
        name = st.selectbox("Perfil", cat2["name"].tolist())
        row = cat2[cat2["name"] == name].iloc[0].to_dict()

        famU = str(fam).upper().strip()
        sec_desc = f"{famU} | {name}"

        if famU == "RETANGULO":
            Iy, Iz, yext, zext, sec_center = rect_Iy_Iz(mm_to_m(float(row["b_mm"])), mm_to_m(float(row["h_mm"])))
        elif famU == "BARRA REDONDA":
            Iy, Iz, yext, zext, sec_center = round_solid_I(mm_to_m(float(row["d_mm"])))
        elif famU == "TUBO REDONDO":
            Iy, Iz, yext, zext, sec_center = round_tube_I(mm_to_m(float(row["od_mm"])), mm_to_m(float(row["t_mm"])))
        elif famU in ["TUBO QUADRADO","TUBO RETANGULAR"]:
            Iy, Iz, yext, zext, sec_center = rect_tube_I(mm_to_m(float(row["b_mm"])), mm_to_m(float(row["h_mm"])), mm_to_m(float(row["t_mm"])))
        else:
            st.error("Família não suportada ainda no import. (Expandimos depois.)")

if Iy <= 0 or Iz <= 0:
    st.warning("Atenção: seção com inércia inválida (Iy ou Iz ≤ 0). Verifique a seção.")

st.caption(f"Seção ativa: **{sec_desc if sec_desc else '—'}** | Iy={Iy:.3e} m⁴ | Iz={Iz:.3e} m⁴")

# ============================================================
# 3) CARREGAMENTOS CAD-LIKE (plano + ponto)
# - força sempre perpendicular ao plano (flexão)
# ============================================================
st.subheader("3) Carregamentos — CAD-like (PLANO + ponto no plano) | FLEXÃO")

if "loads" not in st.session_state:
    st.session_state.loads = []

plane_map = {
    "Superior (X–Y)  → F ⟂ plano = ±Z": "Z",
    "Frontal (X–Z)   → F ⟂ plano = ±Y": "Y",
}

cA, cB, cC = st.columns([1.6, 1.2, 1.0])
with cA:
    plane = st.selectbox("Plano de aplicação (como no SolidWorks Simulation)", list(plane_map.keys()))
with cB:
    kind = st.selectbox("Tipo", ["Força concentrada", "Momento concentrado", "Distribuída (UDL) em trecho"])
with cC:
    sign = st.selectbox("Sentido", ["+", "-"])

x_in = st.number_input(f"Posição ao longo da viga x ({unit_len})", min_value=0.0, max_value=float(L_in), value=float(L_in)/2)
x_m = mm_to_m(x_in) if unit_len=="mm" else float(x_in)

coord_in = st.number_input(
    f"Coordenada no plano ({'y' if '±Z' in plane else 'z'}) ({unit_len}) — livre",
    value=0.0
)
coord_mm = coord_in if unit_len=="mm" else coord_in*1000.0
coord_m = mm_to_m(coord_mm)

force_center_toggle = st.checkbox("Forçar aplicação no centro (evitar torção)", value=False)

draw_plane_preview(plane, x_m, coord_mm, L, sign)

if kind == "Força concentrada":
    P_N = st.number_input("Magnitude (N)", value=1000.0, step=100.0)
    P = float(P_N) * (1.0 if sign == "+" else -1.0)
    M = 0.0
    a_m = b_m = w = 0.0

elif kind == "Momento concentrado":
    M_Nm = st.number_input("Magnitude (N·m)", value=100.0, step=10.0)
    M = float(M_Nm) * (1.0 if sign == "+" else -1.0)
    P = 0.0
    a_m = b_m = w = 0.0

else:
    c1, c2, c3 = st.columns([1.0,1.0,1.2])
    with c1:
        a_in = st.number_input(f"Início a ({unit_len})", value=0.0)
    with c2:
        b_in = st.number_input(f"Fim b ({unit_len})", value=float(L_in))
    with c3:
        w_Nm = st.number_input("Intensidade w (N/m)", value=500.0, step=50.0)

    a_m = mm_to_m(min(a_in,b_in)) if unit_len=="mm" else float(min(a_in,b_in))
    b_m = mm_to_m(max(a_in,b_in)) if unit_len=="mm" else float(max(a_in,b_in))
    w = float(w_Nm) * (1.0 if sign == "+" else -1.0)
    P = 0.0
    M = 0.0

torsion_risk = False
if kind == "Força concentrada" and abs(coord_m) > 1e-9:
    torsion_risk = True

if torsion_risk and not force_center_toggle:
    st.warning("Esse ponto fora do centro induz TORÇÃO. Como o app é FLEXÃO, isso não será contabilizado. Se quiser, marque “Forçar no centro”.")

if st.button("Adicionar carga"):
    coord_eff_mm = 0.0 if force_center_toggle else coord_mm

    st.session_state.loads.append({
        "plane": plane,
        "kind": kind,
        "x_m": x_m,
        "sign": sign,
        "coord_eff_mm": coord_eff_mm,
        "coord_axis": ("y" if "±Z" in plane else "z"),
        "P": float(P) if kind=="Força concentrada" else 0.0,
        "M": float(M) if kind=="Momento concentrado" else 0.0,
        "a_m": float(a_m) if kind.startswith("Distribuída") else 0.0,
        "b_m": float(b_m) if kind.startswith("Distribuída") else 0.0,
        "w": float(w) if kind.startswith("Distribuída") else 0.0,
        "forced_center": bool(force_center_toggle),
    })
    st.success("Carga adicionada.")

cL, cR = st.columns([1,1])
with cL:
    if st.button("Limpar cargas"):
        st.session_state.loads = []
        st.success("Cargas removidas.")
with cR:
    st.caption("Flexão 1D: coord fora do centro implicaria torção (não calculada).")

if st.session_state.loads:
    st.dataframe(pd.DataFrame(st.session_state.loads), width="stretch")
else:
    st.info("Nenhuma carga cadastrada ainda.")

# ============================================================
# 4) CRITÉRIO DE FLECHA
# ============================================================
st.subheader("4) Critério de flecha")
lim = st.selectbox("Limite de flecha", ["L/200","L/250","L/300","L/400"], index=1)
den = {"L/200":200,"L/250":250,"L/300":300,"L/400":400}[lim]
delta_adm_m = L / den

# ============================================================
# CALCULAR
# ============================================================
st.divider()
if st.button("CALCULAR", type="primary"):

    loads_z = []  # plano XY -> Fz -> usa Iy
    loads_y = []  # plano XZ -> Fy -> usa Iz

    for ld in st.session_state.loads:
        if "±Z" in ld["plane"]:
            if ld["kind"] == "Força concentrada":
                loads_z.append({"type":"P", "x": ld["x_m"], "P": ld["P"]})
            elif ld["kind"] == "Momento concentrado":
                loads_z.append({"type":"M", "x": ld["x_m"], "M": ld["M"]})
            else:
                loads_z.append({"type":"w", "a": ld["a_m"], "b": ld["b_m"], "w": ld["w"]})
        else:
            if ld["kind"] == "Força concentrada":
                loads_y.append({"type":"P", "x": ld["x_m"], "P": ld["P"]})
            elif ld["kind"] == "Momento concentrado":
                loads_y.append({"type":"M", "x": ld["x_m"], "M": ld["M"]})
            else:
                loads_y.append({"type":"w", "a": ld["a_m"], "b": ld["b_m"], "w": ld["w"]})

    if Iy <= 0 or Iz <= 0:
        st.error("Seção inválida (Iy ou Iz ≤ 0).")
        st.stop()

    ne = 160

    # Resolver plano Z (usa Iy)
    xs = np.linspace(0.0, L, ne+1)
    wz = np.zeros_like(xs)
    Vz = np.zeros_like(xs)
    My = np.zeros_like(xs)
    reac_z = {}
    if len(loads_z) > 0:
        xs, wz, Vz, My, reac_z = solve_beam_FEM(L, E*Iy, apoio_esq, apoio_dir, loads_z, ne=ne)

    # Resolver plano Y (usa Iz)
    xs2 = np.linspace(0.0, L, ne+1)
    wy = np.zeros_like(xs2)
    Vy = np.zeros_like(xs2)
    Mz = np.zeros_like(xs2)
    reac_y = {}
    if len(loads_y) > 0:
        xs2, wy, Vy, Mz, reac_y = solve_beam_FEM(L, E*Iz, apoio_esq, apoio_dir, loads_y, ne=ne)

    # Resultante
    if len(xs2) == len(xs):
        w_res = np.sqrt(wy**2 + wz**2)
    else:
        w_res = np.sqrt(np.interp(xs, xs2, wy)**2 + wz**2)

    delta_max_m = float(np.max(np.abs(w_res)))
    delta_max_mm = m_to_mm(delta_max_m)
    idx_max = int(np.argmax(np.abs(w_res)))
    x_at_max = float(xs[idx_max])

    # Momentos máximos
    My_max = float(np.max(np.abs(My))) if len(My) else 0.0
    Mz_max = float(np.max(np.abs(Mz))) if len(Mz) else 0.0

    # Cortante máximo (pra fixação/parafusos)
    Vz_max = float(np.max(np.abs(Vz))) if len(Vz) else 0.0
    Vy_max = float(np.max(np.abs(Vy))) if len(Vy) else 0.0
    Vmax = max(Vz_max, Vy_max)

    if Vz_max >= Vy_max and len(Vz):
        iV = int(np.argmax(np.abs(Vz)))
        xV = float(xs[iV])
        Vplane = "Plano XY (V em Z) → Vz"
    elif len(Vy):
        iV = int(np.argmax(np.abs(Vy)))
        xV = float(xs2[iV])
        Vplane = "Plano XZ (V em Y) → Vy"
    else:
        xV = 0.0
        Vplane = "—"

    # Tensões (flexão biaxial)
    ymin, ymax = yext
    zmin, zmax = zext
    corners = [(ymin,zmin),(ymin,zmax),(ymax,zmin),(ymax,zmax)]

    def sigma_at(y, z, My_, Mz_):
        # cuidado com divisão por zero (mas Iy/Iz já validados)
        return (My_ * z) / Iy + (Mz_ * y) / Iz

    sigma_max = float(np.max(np.abs([sigma_at(y,z,My_max,Mz_max) for (y,z) in corners])))

    # >>> TROCA AQUI: validações de tensão (von Mises simplificado) + fy e fy/FS
    sigma_adm = fy / FS
    sigma_vm = abs(sigma_max)  # flexão pura -> tau=0

    ok_defl = delta_max_m <= delta_adm_m
    ok_sigma_adm = sigma_vm <= sigma_adm
    ok_yield = sigma_vm <= fy

    # ============================================================
    # RESULTADOS (prioridade: flecha em mm)
    # ============================================================
    st.subheader("Resultados (prioridade: FLEXÃO / FLECHA)")

    cR1, cR2, cR3, cR4 = st.columns(4)
    with cR1:
        st.metric("Flecha máxima δmax", f"{delta_max_mm:.4f} mm")
        st.caption(f"Limite {lim}: {m_to_mm(delta_adm_m):.4f} mm")
    with cR2:
        st.metric("Posição do pico", f"x = {x_at_max:.3f} m")
        st.caption("Onde a deformação resultante é máxima.")
    with cR3:
        plano_dom = "Z (XY)" if np.max(np.abs(wz)) >= np.max(np.abs(wy)) else "Y (XZ)"
        st.metric("Plano dominante", plano_dom)
        st.caption("Qual direção está “mandando” na flecha.")
    with cR4:
        st.metric("Status Flecha", "OK ✅" if ok_defl else "NÃO OK ❌")
        st.caption("Validação por critério de flecha.")

    st.subheader("Verificação de tensão (objetiva)")
    cS1, cS2, cS3 = st.columns(3)
    with cS1:
        st.metric("σ_von Mises,max", f"{sigma_vm/1e6:.2f} MPa")
    with cS2:
        st.metric("σ_adm = fy/FS", f"{sigma_adm/1e6:.2f} MPa")
        st.caption("Critério de projeto")
    with cS3:
        st.metric("fy (escoamento)", f"{fy/1e6:.2f} MPa")
        st.caption("Limite do material")

    st.write(f"σvm ≤ σadm: {'OK ✅' if ok_sigma_adm else 'NÃO OK ❌'}")
    st.write(f"σvm ≤ fy (escoamento): {'OK ✅' if ok_yield else 'NÃO OK ❌'}")

    st.subheader("Cortante (para dimensionar fixação/parafusos depois)")
    cV1, cV2 = st.columns(2)
    with cV1:
        st.metric("|V|max", f"{Vmax:.2f} N")
    with cV2:
        st.metric("Posição aprox.", f"x ≈ {xV:.3f} m")
        st.caption(Vplane)

    if ok_defl and ok_sigma_adm and ok_yield:
        st.success("✅ VALIDADO (flecha + σadm + escoamento)")
    else:
        falhas = []
        if not ok_defl: falhas.append("flecha")
        if not ok_sigma_adm: falhas.append("σvm > σadm")
        if not ok_yield: falhas.append("σvm > fy")
        st.error(f"❌ NÃO VALIDADO ({', '.join(falhas)})")

    # ============================================================
    # REAÇÕES NOS APOIOS (por plano)
    # ============================================================
    st.subheader("Reações nos apoios (por plano)")

    if reac_z:
        st.markdown("**Plano XY → forças em Z (solver com Iy)**")
        st.dataframe(reactions_table(reac_z), width="stretch")
    else:
        st.info("Sem cargas no plano XY (Z).")

    if reac_y:
        st.markdown("**Plano XZ → forças em Y (solver com Iz)**")
        st.dataframe(reactions_table(reac_y), width="stretch")
    else:
        st.info("Sem cargas no plano XZ (Y).")

    # ============================================================
    # GRÁFICOS
    # ============================================================
    st.subheader("Diagramas / Deformadas (saída em mm)")

    g1, g2 = st.columns(2)

    with g1:
        fig, ax = plt.subplots()
        ax.plot(xs, My, label="My(x) [XY→Fz]")
        ax.plot(xs2, Mz, label="Mz(x) [XZ→Fy]")
        ax.set_title("Momentos fletores")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("M (N·m)")
        ax.grid(True)
        ax.legend()
        st.pyplot(fig)

    with g2:
        fig, ax = plt.subplots()
        ax.plot(xs, m_to_mm(wz), label="wz(x) (mm)")
        ax.plot(xs2, m_to_mm(wy), label="wy(x) (mm)")
        ax.plot(xs, m_to_mm(w_res), label="w_resultante(x) (mm)", linewidth=2)
        ax.scatter([x_at_max], [m_to_mm(w_res[idx_max])], s=60)
        ax.set_title("Deformadas (mm)")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("w (mm)")
        ax.grid(True)
        ax.legend()
        st.pyplot(fig)

    st.subheader("Diagrama de Cortante (N)")
    figV, axV = plt.subplots()
    axV.plot(xs, Vz, label="Vz(x) [XY→Fz]")
    axV.plot(xs2, Vy, label="Vy(x) [XZ→Fy]")
    axV.set_title("Força cortante")
    axV.set_xlabel("x (m)")
    axV.set_ylabel("V (N)")
    axV.grid(True)
    axV.legend()
    st.pyplot(figV)



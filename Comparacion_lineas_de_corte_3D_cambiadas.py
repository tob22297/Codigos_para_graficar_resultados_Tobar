# -*- coding: utf-8 -*-
"""
Gráficas del póster: herida triangular, 5 µA, electrodos de 3 mm estáticos.
Se muestran en el panel de Plots de Spyder (no se guarda nada).

"""

# %% 0. CONFIGURACIÓN
import zipfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm

# --- archivos (ponlos en la carpeta del script o escribe la ruta completa) ---
CSV_2D = "resultados_tesis_consolidados4.csv"
ZIP_3D = "DATOS_3D.zip"                                  # línea de corte original
ZIP_3D_CAMBIADA = "datos_3D_linea_de_corte_cambiada.zip"  # línea de corte cambiada
ARCHIVOS_3D = {"CE": "CE_TRIANGULAR_3D.csv", "DC": "DC_TRIANGULAR_3D.csv"}
ARCHIVOS_3D_CAMBIADA = {"CE": "CE_TRIANGULAR_3D_linea_cambiada.csv",
                        "DC": "DC_TRIANGULAR_3D_linea_cambiada.csv"}

# --- parámetros fijos del póster ---
CORRIENTE = 5             # µA
CORRIENTES = [1, 3, 5, 7, 9]   # orden de las curvas dentro de los CSV 3D
ANCHO = 3                 # mm
DISTANCIA = "estatica"
ESCALAS = [1.0, 0.8, 0.6, 0.4, 0.2]

# columnas del CSV 2D
COL_VALOR = {"CE": "campo_electrico_V_m", "DC": "densidad_corriente_A_m2"}
COL_ARCO = "longitud_arco_mm"

# capas de la piel: (nombre, y_superior, y_inferior) en mm
CAPAS = [("Estrato córneo", 0.007, -0.007),
         ("Epidermis", -0.007, -0.307),
         ("Dermis", -0.307, -2.5)]
GRISES = ["#d9d9d9", "#bdbdbd", "#9e9e9e"]

COLOR_FORMA = {"triangular": "#1f4e9c", "circular": "#1b8a5a", "rectangular": "#c0392b"}
NOMBRE_FORMA = {"triangular": "Triangular", "circular": "Ovalada (circular)",
                "rectangular": "Rectangular"}
ETIQ = {"CE": ("Campo eléctrico", "V/m"), "DC": ("Densidad de corriente", "A/m²")}

plt.rcParams.update({"font.size": 12, "axes.grid": True, "grid.alpha": 0.25})


# %% 1. CARGA Y FUNCIONES
def promedio(arco, val):
    """Promedio espacial con trapecios sobre la longitud de arco."""
    if len(arco) < 2:
        return np.nan
    integral = getattr(np, "trapezoid", None) or np.trapz   # NumPy 1 y 2
    return integral(val, arco) / (arco.max() - arco.min())


def y_de_arco(arco, s):
    """Coordenada y (mm) a lo largo de la línea de corte triangular:
    de (-1.55*s, 0.005) a (-0.05, -1*s), proporcional a la longitud de arco."""
    y0, y1 = 0.005, -1.0 * s
    t = (arco - arco.min()) / (arco.max() - arco.min())
    return y0 + t * (y1 - y0)


df = pd.read_csv(CSV_2D, encoding="utf-8-sig")
sel = df[(df["forma_herida"] == "triangular")
         & (df["distancia_electrodos"] == DISTANCIA)
         & (df["ancho_electrodo_mm"] == ANCHO)
         & (df["corriente_uA"] == CORRIENTE)]


def curva2d(forma, escala, variable):
    d = df[(df["forma_herida"] == forma)
           & (df["distancia_electrodos"] == DISTANCIA)
           & (df["ancho_electrodo_mm"] == ANCHO)
           & (df["corriente_uA"] == CORRIENTE)
           & np.isclose(df["escala"], escala)]
    d = d.sort_values(COL_ARCO)
    return d[COL_ARCO].to_numpy(float), d[COL_VALOR[variable]].to_numpy(float)


def curva3d(zip_path, nombre):
    """Cada CSV 3D trae 5 curvas seguidas (1, 3, 5, 7 y 9 µA); la longitud de
    arco vuelve a 0 al empezar cada una. Aquí se toma la de CORRIENTE."""
    with zipfile.ZipFile(zip_path) as z:
        with z.open(nombre) as f:
            d = pd.read_csv(f, comment="%", header=None, names=["arco", "v"])
    arco, v = d["arco"].to_numpy(float), d["v"].to_numpy(float)
    corte = np.r_[0, np.where(np.diff(arco) < 0)[0] + 1, len(arco)]
    bloques = [(arco[corte[i]:corte[i + 1]], v[corte[i]:corte[i + 1]])
               for i in range(len(corte) - 1)]
    assert len(bloques) == len(CORRIENTES), f"{nombre}: {len(bloques)} curvas"
    return bloques[CORRIENTES.index(CORRIENTE)]


azules = cm.Blues(np.linspace(1.0, 0.35, len(ESCALAS)))


# %% 2. FIGURA A: valor vs longitud de arco, una curva por escala
def figura_A(variable="CE"):
    nombre, unidad = ETIQ[variable]
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    prom = []
    for s, col in zip(ESCALAS, azules):
        arco, val = curva2d("triangular", s, variable)
        ax.plot(arco, val, color=col, lw=2.5, label=f"S = {s}")
        prom.append(promedio(arco, val))
    ax.set_xlabel("Longitud de arco (mm)")
    ax.set_ylabel(f"{nombre} ({unidad})")
    ax.set_title(f"{nombre} a {CORRIENTE} µA · herida triangular")
    ax.legend(loc="upper right", ncol=5, fontsize=9, frameon=True, columnspacing=1, handlelength=1.2)

    ins = ax.inset_axes([0.47, 0.47, 0.24, 0.27])      # promedio por escala
    ins.bar(range(len(ESCALAS)), prom, color=azules)
    ins.set_xticks(range(len(ESCALAS)))
    ins.set_xticklabels([str(s) for s in ESCALAS])
    ins.set_title("Promedio por escala", fontsize=9)
    ins.tick_params(labelsize=8)
    ins.grid(False)

    esq = ax.inset_axes([0.75, 0.45, 0.23, 0.30])      # esquema línea de corte
    esq.fill([-1.5, 1.5, 0], [0, 0, -1], color="#85B7EB", alpha=0.7)
    esq.plot([-1.55, -0.05], [0.005, -1.0], "k--", lw=1.2)
    esq.set_title("Línea de corte", fontsize=9)
    esq.set_aspect("equal")
    esq.axis("off")
    plt.tight_layout()
    plt.show()


figura_A("CE")
figura_A("DC")


# %% 3. FIGURA B: efecto de las capas (valor vs y, fondo gris por capa)
def figura_B(variable="CE"):
    nombre, unidad = ETIQ[variable]
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for (cap, ys, yi), g in zip(CAPAS, GRISES):
        ax.axvspan(yi, ys, color=g, alpha=0.55, lw=0)
    for s, col in zip(ESCALAS, azules):
        arco, val = curva2d("triangular", s, variable)
        ax.plot(y_de_arco(arco, s), val, color=col, lw=2.5, label=f"S = {s}")
    ax.set_xlim(0.01, -1.05)
    ax.set_xlabel("Coordenada y (mm)")
    ax.set_ylabel(f"{nombre} ({unidad})")
    ax.set_title(f"Efecto de las capas · {CORRIENTE} µA, herida triangular")
    ymax = ax.get_ylim()[1]
    ax.annotate("Estrato córneo", xy=(0, ymax * 0.97), xytext=(-0.1, ymax * 0.97),
                fontsize=9, va="top", arrowprops=dict(arrowstyle="-", lw=0.8))
    ax.text(-0.16, ymax * 0.78, "Epidermis", fontsize=9, ha="center")
    ax.text(-0.70, ymax * 0.90, "Dermis", fontsize=9, ha="center")
    ax.legend(loc="center right", frameon=True)
    plt.tight_layout()
    plt.show()


figura_B("CE")
figura_B("DC")


# %% 4. FIGURA E: promedio espacial por forma de herida, 5 µA, SIN normalizar
def figura_E(variable="CE"):
    nombre, unidad = ETIQ[variable]
    fig, ax = plt.subplots(figsize=(7.5, 4.5))

    marcadores = {
        "triangular": "^",
        "circular": "o",
        "rectangular": "s"
    }

    for forma in ["triangular", "circular", "rectangular"]:
        prom = [promedio(*curva2d(forma, s, variable)) for s in ESCALAS]
        ax.plot(range(len(ESCALAS)), prom,
                linestyle="-", marker=marcadores[forma], lw=4, ms=12,
                color=COLOR_FORMA[forma], label=NOMBRE_FORMA[forma])

    ax.set_xticks(range(len(ESCALAS)))
    ax.set_xticklabels([str(s) for s in ESCALAS])
    ax.set_xlabel("Escala de la herida (S)")
    ax.set_ylabel(f"{nombre} ({unidad})")
    ax.set_title(f"{nombre} promedio con corriente de {CORRIENTE} µA")
    ax.set_ylim(bottom=0)
    ax.legend(title="Forma de la herida", frameon=True)
    plt.tight_layout()
    plt.show()


figura_E("CE")
figura_E("DC")


# %% 5. FIGURA F: modelo 3D, línea de corte original vs cambiada (triangular)
def figura_F(variable="CE"):
    nombre, unidad = ETIQ[variable]
    a0, v0 = curva3d(ZIP_3D, ARCHIVOS_3D[variable])
    a1, v1 = curva3d(ZIP_3D_CAMBIADA, ARCHIVOS_3D_CAMBIADA[variable])
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(a0, v0, color="#1f4e9c", lw=2.5,
            label=f"Original (prom. {promedio(a0, v0):.3g})")
    ax.plot(a1, v1, color="#90D5FF", lw=2.5, ls="--",
            label=f"Cambiada (prom. {promedio(a1, v1):.3g})")
    ax.set_xlabel("Longitud de arco (mm)")
    ax.set_ylabel(f"{nombre} ({unidad})")
    ax.set_title(f"Modelo 3D · línea de corte original vs cambiada")
    ax.legend(frameon=True)
    plt.tight_layout()
    plt.show()


figura_F("CE")
figura_F("DC")

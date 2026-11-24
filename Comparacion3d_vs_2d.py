# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 21:16:40 2026

@author: mtobar
"""

# -*- coding: utf-8 -*-
"""
Comparación 2D vs 3D del campo eléctrico y de la densidad de corriente
a lo largo de la línea de corte (electrodos de 3 mm, estáticos, escala 1.0).

Genera 3 figuras (circular, rectangular, triangular). Cada figura tiene:
    columna izquierda  -> modelo 2D
    columna derecha    -> modelo 3D
    fila superior      -> campo eléctrico (V/m)
    fila inferior      -> densidad de corriente (A/m²)
Las curvas son las 5 corrientes (1, 3, 5, 7 y 9 µA) y el fondo gris marca las
capas de la piel. Los ejes Y se comparten entre 2D y 3D para poder compararlos
directamente. No se guarda ningún archivo: las figuras se muestran en Spyder.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# =============================================================================
# CONFIGURACIÓN  (cambia solo esta parte)
# =============================================================================
CARPETA = r"."   # carpeta donde están los archivos (ej. r"C:\Users\...\TESIS")

# 2D: CSV consolidado (circular y rectangular salen de aquí)
ARCHIVO_2D = "resultados_tesis_consolidados4.csv"

# 2D triangular con los datos nuevos (carpeta TRIANGULO). Pon None para usar
# la triangular que viene dentro del consolidado.
TRIANGULAR_2D_NUEVO = {
    "CE": os.path.join("TRIANGULO", "CE_triangular_electrodos3mm_estaticos1.csv"),
    "DC": os.path.join("TRIANGULO", "DC_triangular_electrodos3mm_estaticos1.csv"),
}

# 3D: (archivo de campo eléctrico, archivo de densidad de corriente)
ARCHIVOS_3D = {
    "circular":    (os.path.join("DATOS_3D", "CE_CRICULAR_3D.csv"),
                    os.path.join("DATOS_3D", "DC_CIRCULAR_3D.csv")),
    "rectangular": (os.path.join("DATOS_3D", "CE_RECTANGULAR_3D_final.csv"),
                    os.path.join("DATOS_3D", "DC_RECTANGULAR_3D.csv")),
    "triangular":  (os.path.join("DATOS_3D", "CE_TRIANGULAR_3D.csv"),
                    os.path.join("DATOS_3D", "DC_TRIANGULAR_3D.csv")),
}

# 3D con la línea de corte desplazada (carpeta con los archivos del zip nuevo)
ARCHIVOS_3D_CAMBIADA = {
    "circular":    (os.path.join("DATOS_3D_LINEA_CAMBIADA", "CE_CRICULAR_3D_linea_cambiada.csv"),
                    os.path.join("DATOS_3D_LINEA_CAMBIADA", "DC_CIRCULAR_3D_linea_cambiada.csv")),
    "rectangular": (os.path.join("DATOS_3D_LINEA_CAMBIADA", "CE_RECTANGULAR_3D_LINEA_CAMBIADA.csv"),
                    os.path.join("DATOS_3D_LINEA_CAMBIADA", "DC_RECTANGULAR_3D_LINEA_CAMBIADA.csv")),
    "triangular":  (os.path.join("DATOS_3D_LINEA_CAMBIADA", "CE_TRIANGULAR_3D_linea_cambiada.csv"),
                    os.path.join("DATOS_3D_LINEA_CAMBIADA", "DC_TRIANGULAR_3D_linea_cambiada.csv")),
}

TAM_NUMEROS_EJES = 14                 # tamaño de los números de los ejes
TAM_TITULOS_EJES = 14                 # tamaño de los textos de los ejes
CORRIENTES = [1, 3, 5, 7, 9]          # µA, en el orden en que están los bloques
COLOR_BASE = {"circular": "#238b5a",  # verde
              "rectangular": "#c1404a",  # rojo
              "triangular": "#2159a5"}   # azul
NOMBRE = {"circular": "circular", "rectangular": "rectangular",
          "triangular": "triangular"}

# Conductividades del modelo (S/m) para identificar las capas desde J/E:
# estrato córneo 2e-6, epidermis 0.026, dermis 0.222.
# Los umbrales son los puntos medios (en escala logarítmica) entre ellas.
UMBRAL_SC_EPI = np.sqrt(2e-6 * 0.026)
UMBRAL_EPI_DER = np.sqrt(0.026 * 0.222)
GRIS_CAPA = {"Estrato córneo": "#ececec", "Epidermis": "#d9d9d9", "Dermis": "#b9b9b9"}

trapecio = getattr(np, "trapezoid", None) or np.trapz


# =============================================================================
# LECTURA DE DATOS
# =============================================================================
def ruta(nombre):
    return os.path.join(CARPETA, nombre)


def leer_comsol(archivo):
    """Lee un CSV de COMSOL (líneas '%' = encabezado). Devuelve una lista de
    curvas (s, valor); cada vez que la longitud de arco vuelve a bajar empieza
    una curva nueva."""
    d = pd.read_csv(ruta(archivo), comment="%", header=None, names=["s", "v"],
                    encoding="utf-8", encoding_errors="replace")
    s = d["s"].to_numpy(float)
    v = d["v"].to_numpy(float)
    cortes = np.where(np.diff(s) < 0)[0] + 1
    return [(s[i], v[i]) for i in np.split(np.arange(len(s)), cortes)]


def leer_2d(forma):
    """Devuelve (curvas_E, curvas_J): listas de (s, valor), una por corriente."""
    if forma == "triangular" and TRIANGULAR_2D_NUEVO:
        cE = leer_comsol(TRIANGULAR_2D_NUEVO["CE"])
        cJ = leer_comsol(TRIANGULAR_2D_NUEVO["DC"])
        if len(cE) != 25 or len(cJ) != 25:
            raise ValueError("Se esperaban 25 curvas (5 corrientes x 5 escalas) "
                             f"y hay {len(cE)} y {len(cJ)}.")
        # orden: para cada corriente, escalas 0.2, 0.4, 0.6, 0.8 y 1.0 -> escala 1.0 = 5*i+4
        idx = [5 * i + 4 for i in range(len(CORRIENTES))]
        return [cE[i] for i in idx], [cJ[i] for i in idx]

    df = pd.read_csv(ruta(ARCHIVO_2D), encoding="utf-8-sig")
    sel = df[(df["forma_herida"] == forma)
             & (df["distancia_electrodos"] == "estatica")
             & (df["ancho_electrodo_mm"] == 3)
             & (np.isclose(df["escala"], 1.0))]
    cE, cJ = [], []
    for I in CORRIENTES:
        g = sel[sel["corriente_uA"] == I].sort_values("indice_punto_curva")
        if g.empty:
            raise ValueError(f"No hay datos 2D para {forma} a {I} µA.")
        s = g["longitud_arco_mm"].to_numpy(float)
        cE.append((s, g["campo_electrico_V_m"].to_numpy(float)))
        cJ.append((s, g["densidad_corriente_A_m2"].to_numpy(float)))
    return cE, cJ


def leer_3d(forma, archivos=None):
    archivo_E, archivo_J = (archivos or ARCHIVOS_3D)[forma]
    cE, cJ = leer_comsol(archivo_E), leer_comsol(archivo_J)
    if len(cE) != len(CORRIENTES) or len(cJ) != len(CORRIENTES):
        raise ValueError(f"3D {forma}: se esperaban {len(CORRIENTES)} curvas y hay "
                         f"{len(cE)} (E) y {len(cJ)} (J).")
    # comprobación: el máximo debe crecer proporcional a la corriente
    mx = np.array([c[1].max() for c in cE])
    if not np.allclose(mx / mx[0], np.array(CORRIENTES) / CORRIENTES[0], rtol=0.02):
        print(f"AVISO ({forma}, 3D): los máximos no crecen como 1:3:5:7:9; "
              "revisa el orden de las corrientes.")
    return cE, cJ


# =============================================================================
# UTILIDADES
# =============================================================================
def degradado(color_base, n):
    """n colores de más claro (corriente baja) a más oscuro (corriente alta)."""
    base = np.array(mcolors.to_rgb(color_base))
    claro = base + (1 - base) * 0.65
    oscuro = base * 0.45
    return [tuple(claro + (oscuro - claro) * t) for t in np.linspace(0, 1, n)]


def limites_capas(curvas_E, curvas_J):
    """Longitudes de arco donde termina el estrato córneo y la epidermis,
    deducidas de la conductividad local sigma = J/E."""
    i = len(curvas_E) // 2
    s, E = curvas_E[i]
    J = curvas_J[i][1]
    with np.errstate(divide="ignore", invalid="ignore"):
        sigma = np.where(E > 0, J / E, np.nan)
    def frontera(umbral):
        k = np.where(sigma > umbral)[0][0]          # primer punto ya dentro de la capa
        return 0.5 * (s[k - 1] + s[k]) if k > 0 else s[0]
    return frontera(UMBRAL_SC_EPI), frontera(UMBRAL_EPI_DER)


def dibujar_panel(ax, curvas, colores, limites, x_max):
    b1, b2 = limites
    ax.axvspan(0, b1, color=GRIS_CAPA["Estrato córneo"], zorder=0, lw=0)
    ax.axvspan(b1, b2, color=GRIS_CAPA["Epidermis"], zorder=0, lw=0)
    ax.axvspan(b2, x_max, color=GRIS_CAPA["Dermis"], zorder=0, lw=0)
    for (s, v), c in zip(curvas, colores):
        ax.plot(s, v, color=c, lw=1.6, zorder=3)
    ax.set_xlim(0, x_max)
    ax.grid(alpha=0.25, zorder=1)


def promedio_por_uA(curvas):
    """Promedio espacial (trapecios sobre la longitud de arco) por µA."""
    vals = []
    for (s, v), I in zip(curvas, CORRIENTES):
        vals.append(trapecio(v, s) / (s[-1] - s[0]) / I)
    return float(np.mean(vals))


def maximo_por_uA(curvas):
    return float(np.mean([v.max() / I for (s, v), I in zip(curvas, CORRIENTES)]))


# =============================================================================
# FIGURAS
# =============================================================================
def construir_figuras():
    figuras = []
    for forma in ["circular", "rectangular", "triangular"]:
        E2, J2 = leer_2d(forma)
        E3, J3 = leer_3d(forma)
        colores = degradado(COLOR_BASE[forma], len(CORRIENTES))

        x_max = max(E2[0][0][-1], E3[0][0][-1])
        lim2 = limites_capas(E2, J2)
        lim3 = limites_capas(E3, J3)

        if abs(E2[0][0][-1] - E3[0][0][-1]) / E3[0][0][-1] > 0.05:
            print(f"AVISO ({forma}): la longitud de la línea difiere entre 2D "
                  f"({E2[0][0][-1]:.3f} mm) y 3D ({E3[0][0][-1]:.3f} mm).")

        fig, ax = plt.subplots(2, 2, figsize=(13, 8), sharex=True, sharey="row")
        dibujar_panel(ax[0, 0], E2, colores, lim2, x_max)
        dibujar_panel(ax[0, 1], E3, colores, lim3, x_max)
        dibujar_panel(ax[1, 0], J2, colores, lim2, x_max)
        dibujar_panel(ax[1, 1], J3, colores, lim3, x_max)

        # mismo eje Y para 2D y 3D (el mayor valor entre los dos modelos)
        ymax_E = max(c[1].max() for c in E2 + E3)
        ymax_J = max(c[1].max() for c in J2 + J3)
        ax[0, 0].set_ylim(0, ymax_E * 1.05)
        ax[1, 0].set_ylim(0, ymax_J * 1.05)

        ax[0, 0].set_title("Modelo 2D", fontsize=13)
        ax[0, 1].set_title("Modelo 3D", fontsize=13)
        ax[0, 0].set_ylabel("Campo eléctrico (V/m)", fontsize=TAM_TITULOS_EJES)
        ax[1, 0].set_ylabel("Densidad de corriente (A/m²)", fontsize=TAM_TITULOS_EJES)
        for a in ax[1, :]:
            a.set_xlabel("Longitud de arco (mm)", fontsize=TAM_TITULOS_EJES)
        for a in ax.ravel():
            a.tick_params(axis="both", labelsize=TAM_NUMEROS_EJES)

        # una sola leyenda: corrientes + capas
        h = [Line2D([0], [0], color=c, lw=2, label=f"{I} µA")
             for c, I in zip(colores, CORRIENTES)]
        h += [Patch(facecolor=col, label=nombre) for nombre, col in GRIS_CAPA.items()]
        fig.legend(handles=h, loc="lower center", ncol=len(h), frameon=False,
                   fontsize=10)
        fig.suptitle(f"Herida {NOMBRE[forma]} · electrodos de 3 mm estáticos · "
                     "escala 1.0", fontsize=14)
        fig.tight_layout(rect=[0, 0.06, 1, 0.96])
        figuras.append(fig)
    return figuras


def imprimir_resumen():
    """Tabla en consola con máximos y promedios espaciales por µA (2D y 3D)."""
    filas = []
    for forma in ["circular", "rectangular", "triangular"]:
        E2, J2 = leer_2d(forma)
        E3, J3 = leer_3d(forma)
        for nombre, c2, c3 in [("Campo eléctrico (V/m)/µA", E2, E3),
                               ("Densidad de corriente (A/m²)/µA", J2, J3)]:
            for medida, f in [("máximo", maximo_por_uA), ("promedio", promedio_por_uA)]:
                v2, v3 = f(c2), f(c3)
                filas.append([forma, nombre, medida, v2, v3, v3 / v2,
                              (v3 / v2 - 1) * 100])
    t = pd.DataFrame(filas, columns=["herida", "variable", "medida", "2D", "3D",
                                     "razón 3D/2D", "diferencia 3D vs 2D (%)"])
    with pd.option_context("display.max_columns", None, "display.width", 200,
                           "display.float_format", "{:.4g}".format):
        print(t.to_string(index=False))
    return t


# =============================================================================
# VERSIONES COMPACTAS
# =============================================================================
FORMAS = ["circular", "rectangular", "triangular"]


def curva_normalizada(curvas):
    """Curva promedio dividida entre la corriente (la respuesta es lineal, así
    que las 5 corrientes colapsan en una sola curva por µA)."""
    s = curvas[0][0]
    v = np.mean([np.interp(s, c[0], c[1]) / I for c, I in zip(curvas, CORRIENTES)],
                axis=0)
    return s, v


def figura_curvas_compacta(corriente_uA=5):
    """2 filas (campo, densidad de corriente) x 3 columnas (formas). En cada
    panel, el 2D (línea continua) y el 3D (línea discontinua).
    corriente_uA = 5  -> valores reales (sin normalizar) a 5 µA
    corriente_uA = None -> curvas normalizadas por µA"""
    if corriente_uA is not None and corriente_uA not in CORRIENTES:
        raise ValueError(f"La corriente debe ser una de {CORRIENTES}")
    k = None if corriente_uA is None else CORRIENTES.index(corriente_uA)
    fig, ax = plt.subplots(2, 3, figsize=(15, 7), sharex="col")
    for j, forma in enumerate(FORMAS):
        E2, J2 = leer_2d(forma)
        E3, J3 = leer_3d(forma)
        lim = limites_capas(E2, J2)
        x_max = max(E2[0][0][-1], E3[0][0][-1])
        color = COLOR_BASE[forma]
        for i, (c2, c3) in enumerate([(E2, E3), (J2, J3)]):
            a = ax[i, j]
            a.axvspan(0, lim[0], color=GRIS_CAPA["Estrato córneo"], zorder=0, lw=0)
            a.axvspan(lim[0], lim[1], color=GRIS_CAPA["Epidermis"], zorder=0, lw=0)
            a.axvspan(lim[1], x_max, color=GRIS_CAPA["Dermis"], zorder=0, lw=0)
            if k is None:
                s2, v2 = curva_normalizada(c2)
                s3, v3 = curva_normalizada(c3)
            else:
                s2, v2 = c2[k]
                s3, v3 = c3[k]
            a.plot(s2, v2, color=color, lw=2.2, zorder=3)
            a.plot(s3, v3, color=color, lw=2.2, ls=(0, (4, 2)), zorder=3)
            a.set_xlim(0, x_max)
            a.set_ylim(bottom=0)
            a.grid(alpha=0.25, zorder=1)
            a.tick_params(axis="both", labelsize=TAM_NUMEROS_EJES)
            if i == 0:
                a.set_title(f"Herida {NOMBRE[forma]}", fontsize=TAM_TITULOS_EJES + 1)
            if i == 1:
                a.set_xlabel("Longitud de arco (mm)", fontsize=TAM_TITULOS_EJES)
            if j == 0:
                unidad = "/µA" if k is None else ""
                a.set_ylabel([f"Campo eléctrico\n(V/m){unidad}",
                              f"Densidad de corriente\n(A/m²){unidad}"][i],
                             fontsize=TAM_TITULOS_EJES)
    h = [Line2D([0], [0], color="k", lw=2.2, label="Modelo 2D"),
         Line2D([0], [0], color="k", lw=2.2, ls=(0, (4, 2)), label="Modelo 3D")]
    h += [Patch(facecolor=col, label=n) for n, col in GRIS_CAPA.items()]
    fig.legend(handles=h, loc="lower center", ncol=len(h), frameon=False, fontsize=11)
    cond = "normalizado por µA" if k is None else f"{corriente_uA} µA"
    fig.suptitle(f"2D vs 3D · {cond} · electrodos de 3 mm estáticos · "
                 "escala 1.0", fontsize=14)
    fig.tight_layout(rect=[0, 0.06, 1, 0.95])
    return fig


def figura_barras():
    """2 filas (campo, densidad de corriente) x 2 columnas (máximo, promedio).
    Barras 2D (rayadas) y 3D (llenas) por forma, con el cambio porcentual."""
    fig, ax = plt.subplots(2, 2, figsize=(11, 7))
    ancho = 0.36
    x = np.arange(len(FORMAS))
    datos = {f: (leer_2d(f), leer_3d(f)) for f in FORMAS}
    filas = [("Campo eléctrico (V/m)/µA", 0), ("Densidad de corriente (A/m²)/µA", 1)]
    cols = [("Máximo", maximo_por_uA), ("Promedio espacial", promedio_por_uA)]
    for (nom_v, iv) in filas:
        for jc, (nom_m, f) in enumerate(cols):
            a = ax[iv, jc]
            v2 = [f(datos[fm][0][iv]) for fm in FORMAS]
            v3 = [f(datos[fm][1][iv]) for fm in FORMAS]
            for k, fm in enumerate(FORMAS):
                c = COLOR_BASE[fm]
                a.bar(x[k] - ancho / 2, v2[k], ancho, facecolor="white",
                      edgecolor=c, hatch="///", lw=1.5)
                a.bar(x[k] + ancho / 2, v3[k], ancho, facecolor=c, edgecolor=c)
                a.text(x[k] + ancho / 2, v3[k], f"{(v3[k] / v2[k] - 1) * 100:+.0f} %",
                       ha="center", va="bottom", fontsize=TAM_NUMEROS_EJES - 1,
                       fontweight="bold")
            a.set_xticks(x)
            a.set_xticklabels([n.capitalize() for n in FORMAS])
            a.set_ylim(0, max(v3 + v2) * 1.15)
            a.tick_params(axis="both", labelsize=TAM_NUMEROS_EJES)
            a.grid(axis="y", alpha=0.25)
            a.set_axisbelow(True)
            if iv == 0:
                a.set_title(nom_m, fontsize=TAM_TITULOS_EJES + 1)
            if jc == 0:
                a.set_ylabel(nom_v, fontsize=TAM_TITULOS_EJES - 1)
    h = [Patch(facecolor="white", edgecolor="0.3", hatch="///", label="Modelo 2D"),
         Patch(facecolor="0.3", edgecolor="0.3", label="Modelo 3D (cambio respecto al 2D)")]
    fig.legend(handles=h, loc="lower center", ncol=2, frameon=False, fontsize=11)
    fig.suptitle("2D vs 3D · electrodos de 3 mm estáticos · escala 1.0", fontsize=14)
    fig.tight_layout(rect=[0, 0.05, 1, 0.95])
    return fig


def figura_linea_corte_3D(corriente_uA=5):
    """Modelo 3D: línea de corte al centro (continua) vs línea de corte
    desplazada (discontinua). 2 filas (campo, densidad de corriente) x 3 formas.
    Las líneas verticales finas marcan la frontera epidermis-dermis de cada línea."""
    if corriente_uA not in CORRIENTES:
        raise ValueError(f"La corriente debe ser una de {CORRIENTES}")
    k = CORRIENTES.index(corriente_uA)
    fig, ax = plt.subplots(2, 3, figsize=(15, 7))
    for j, forma in enumerate(FORMAS):
        Ec, Jc = leer_3d(forma)                          # centro
        Ed, Jd = leer_3d(forma, ARCHIVOS_3D_CAMBIADA)    # desplazada
        color = COLOR_BASE[forma]
        lc, ld = limites_capas(Ec, Jc)[1], limites_capas(Ed, Jd)[1]
        x_max = max(Ec[0][0][-1], Ed[0][0][-1])
        for i, (cc, cd) in enumerate([(Ec, Ed), (Jc, Jd)]):
            a = ax[i, j]
            a.axvline(lc, color=color, lw=1.2, alpha=0.4, zorder=1)
            a.axvline(ld, color=color, lw=1.2, alpha=0.4, ls=(0, (4, 2)), zorder=1)
            a.plot(*cc[k], color=color, lw=2.2, zorder=3)
            a.plot(*cd[k], color=color, lw=2.2, ls=(0, (4, 2)), zorder=3)
            a.set_xlim(0, x_max)
            a.set_ylim(bottom=0)
            a.grid(alpha=0.25, zorder=0)
            a.tick_params(axis="both", labelsize=TAM_NUMEROS_EJES)
            if i == 0:
                a.set_title(f"Herida {NOMBRE[forma]}", fontsize=TAM_TITULOS_EJES + 1)
            if i == 1:
                a.set_xlabel("Longitud de arco (mm)", fontsize=TAM_TITULOS_EJES)
            if j == 0:
                a.set_ylabel(["Campo eléctrico\n(V/m)", "Densidad de corriente\n(A/m²)"][i],
                             fontsize=TAM_TITULOS_EJES)
    h = [Line2D([0], [0], color="k", lw=2.2, label="Línea de corte al centro"),
         Line2D([0], [0], color="k", lw=2.2, ls=(0, (4, 2)), label="Línea de corte desplazada"),
         Line2D([0], [0], color="0.6", lw=1.2, label="Frontera epidermis-dermis")]
    fig.legend(handles=h, loc="lower center", ncol=3, frameon=False, fontsize=11)
    fig.suptitle(f"Modelo 3D · efecto de la posición de la línea de corte · {corriente_uA} µA "
                 "· electrodos de 3 mm estáticos · escala 1.0", fontsize=14)
    fig.tight_layout(rect=[0, 0.06, 1, 0.95])
    return fig


def imprimir_resumen_linea_corte_3D():
    """Máximo y promedio espacial por µA con la línea al centro vs desplazada."""
    filas = []
    for forma in FORMAS:
        Ec, Jc = leer_3d(forma)
        Ed, Jd = leer_3d(forma, ARCHIVOS_3D_CAMBIADA)
        for nombre, c0, c1 in [("Campo eléctrico (V/m)/µA", Ec, Ed),
                               ("Densidad de corriente (A/m²)/µA", Jc, Jd)]:
            for medida, f in [("máximo", maximo_por_uA), ("promedio", promedio_por_uA)]:
                v0, v1 = f(c0), f(c1)
                filas.append([forma, nombre, medida, v0, v1, (v1 / v0 - 1) * 100])
        filas.append([forma, "Longitud de la línea (mm)", "-", Ec[0][0][-1],
                      Ed[0][0][-1], (Ed[0][0][-1] / Ec[0][0][-1] - 1) * 100])
    t = pd.DataFrame(filas, columns=["herida", "variable", "medida", "centro",
                                     "desplazada", "cambio (%)"])
    with pd.option_context("display.max_columns", None, "display.width", 200,
                           "display.float_format", "{:.4g}".format):
        print(t.to_string(index=False))
    return t


if __name__ == "__main__":
    imprimir_resumen()
    # Versiones compactas (comenta las que no quieras):
    figura_curvas_compacta(5)  # curvas 2D vs 3D a 5 µA (sin normalizar); usa None para normalizar por µA
    figura_barras()            # barras de máximo y promedio con % de cambio
    construir_figuras()        # figuras detalladas (una por herida, 5 corrientes)
    imprimir_resumen_linea_corte_3D()
    figura_linea_corte_3D(5)   # 3D: línea de corte al centro vs desplazada (a 5 µA)
    plt.show()
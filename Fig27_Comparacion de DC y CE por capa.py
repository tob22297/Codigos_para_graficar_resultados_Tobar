"""Seis figuras de tres líneas: capa por escala, normalizada por µA.

Tres formas × campo eléctrico o densidad de corriente. Electrodos de 3 mm,
distancia estática. Coloca este .py junto a resultados_tesis_consolidados.csv.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CARPETA = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
CSV = CARPETA / "resultados_tesis_consolidados4.csv"
SALIDA = CARPETA / "figuras_capas_eje_y_identico"
GUARDAR_PNG = True
ANCHO_ELECTRODO_MM = 3
POSICION = "estatica"
FORMAS = ("circular", "rectangular", "triangular")
CAPAS = ("Estrato córneo", "Epidermis", "Dermis")
ESCALAS = (0.2, 0.4, 0.6, 0.8, 1.0)
CORRIENTES = (1, 3, 5, 7, 9)
VARIABLES = (
    ("campo_electrico_V_m", "campo eléctrico", "(V/m)/µA", "campo"),
    ("densidad_corriente_A_m2", "densidad de corriente", "(A/m²)/µA", "densidad"),
)
# Límites y marcas IDÉNTICOS en las tres figuras de cada magnitud.
# Se usan rangos separados porque campo y densidad tienen unidades distintas.
EJE_Y = {
    "campo": {"limites": (0.0, 0.6), "marcas": tuple(i/10 for i in range(7))},
    "densidad": {"limites": (8e-5, 1e-1), "marcas": (1e-4, 1e-3, 1e-2)},
}
# Cada forma conserva su color: circular verde, rectangular rojo, triangular azul.
# Los tres tonos, de claro a oscuro, representan las capas.
COLORES = {
    "circular": ("#79c98e", "#238557", "#075036"),
    "rectangular": ("#f29c94", "#c33f47", "#790c21"),
    "triangular": ("#80c4e5", "#2359a3", "#13076b"),
}
MARCADORES = ("o", "s", "^")


def tramos_en_arco(forma, escala, longitud):
    """Cruces de la línea de corte con y=+.007, -.007 y -.317 mm."""
    if forma == "circular":
        # x=-1.55*scale*cos(pi*t), y=-1.05*scale*sin(pi*t), t=-.007..0.5.
        t = np.linspace(-0.007, 0.5, 20001)
        velocidad = np.pi*np.hypot(1.55*np.sin(np.pi*t), 1.05*np.cos(np.pi*t))
        arco = np.r_[0., np.cumsum((velocidad[:-1]+velocidad[1:])*np.diff(t)/2)]
        factor = longitud/arco[-1]
        ys = -1.05*factor*np.sin(np.pi*t)

        def cruce(nivel):
            if nivel >= ys[0]:
                return 0.0
            if nivel <= ys[-1]:
                return longitud
            parametro = np.interp(nivel, ys[::-1], t[::-1])
            return float(np.interp(parametro, t, arco)*factor)

    elif forma == "triangular":
        # y desciende linealmente desde +0.005 hasta -scale.
        def cruce(nivel):
            return float(np.clip((0.005-nivel)/(0.005+escala)*longitud, 0, longitud))

    elif forma == "rectangular":
        # Tramo vertical desde +0.005 hasta -scale; después horizontal.
        def cruce(nivel):
            if nivel >= 0.005:
                return 0.0
            if nivel <= -escala:
                return longitud
            return float(np.clip(0.005-nivel, 0, longitud))
    else:
        raise ValueError(f"Forma desconocida: {forma}")

    bordes = (cruce(+0.007), cruce(-0.007), cruce(-0.317), longitud)
    if any(b < a-1e-9 for a, b in zip(bordes[:-1], bordes[1:])):
        raise ValueError(f"Cruces desordenados: {forma}, escala {escala}")
    return dict(zip(CAPAS, zip(bordes[:-1], bordes[1:])))


def promedio_en_capa(arco, valores, inicio, fin):
    """Integral trapezoidal del tramo dividida por su longitud de arco."""
    if fin-inicio <= 1e-9:
        return np.nan  # En esta escala la herida aún no alcanza la capa.
    dentro = (arco > inicio) & (arco < fin)
    x = np.r_[inicio, arco[dentro], fin]
    y = np.r_[np.interp(inicio, arco, valores), valores[dentro],
              np.interp(fin, arco, valores)]
    return float(np.trapezoid(y, x)/(fin-inicio))


def resumir(datos):
    filas = []
    for (forma, escala, corriente), g in datos.groupby(
        ["forma_herida", "escala", "corriente_uA"]
    ):
        g = g.sort_values("indice_punto_curva")
        arco = g.longitud_arco_mm.to_numpy(dtype=float)
        if (len(arco) < 2 or arco[-1] <= arco[0]
                or np.any(np.diff(arco) < 0) or not np.isfinite(arco).all()):
            raise ValueError(f"Arco inválido: {forma}, {escala}, {corriente} µA")
        limites = tramos_en_arco(forma, escala, arco[-1])
        for capa, (a, b) in limites.items():
            fila = {"forma": forma, "escala": escala,
                    "corriente_uA": corriente, "capa": capa}
            for columna, _, _, nombre in VARIABLES:
                valores = g[columna].to_numpy(dtype=float)
                if not np.isfinite(valores).all():
                    raise ValueError(f"Datos incompletos: {forma}, {escala}, {corriente}")
                fila[nombre+"_medio_por_uA"] = (
                    promedio_en_capa(arco, valores, a, b)/corriente
                )
            filas.append(fila)
    tabla = pd.DataFrame(filas)
    esperado = {(f, s, c, capa) for f in FORMAS for s in ESCALAS
                for c in CORRIENTES for capa in CAPAS}
    real = set(zip(tabla.forma, tabla.escala, tabla.corriente_uA, tabla.capa))
    if real != esperado or len(tabla) != len(esperado):
        raise ValueError(f"Faltan combinaciones: {sorted(esperado-real)}")
    columnas_valor = [nombre+"_medio_por_uA" for _, _, _, nombre in VARIABLES]
    return tabla.groupby(["forma", "escala", "capa"], as_index=False)[columnas_valor].mean()


def comprobar_rangos(resumen):
    """Evita recortar datos si se usa un CSV distinto al validado."""
    for _, _, _, nombre in VARIABLES:
        valores = resumen[nombre+"_medio_por_uA"].dropna().to_numpy(dtype=float)
        minimo, maximo = EJE_Y[nombre]["limites"]
        if len(valores) == 0 or valores.min() < minimo or valores.max() > maximo:
            raise ValueError(f"Los datos de {nombre} exceden el eje Y: "
                             f"{(minimo, maximo)}. Ajusta EJE_Y.")


def graficar(resumen, forma, variable):
    _, magnitud, unidad, nombre = variable
    fig, ax = plt.subplots(figsize=(11, 7), layout="constrained")
    for capa, color, marcador in zip(CAPAS, COLORES[forma], MARCADORES):
        serie = resumen[(resumen.forma == forma) & (resumen.capa == capa)]
        y = serie.set_index("escala").reindex(ESCALAS)[nombre+"_medio_por_uA"].to_numpy()
        ax.plot(ESCALAS, y, color=color, marker="o", markersize=15,
                linewidth=7, label=capa)
    if nombre == "densidad":
        # La dermis puede superar 100 veces al estrato córneo.
        ax.set_yscale("log")
        ax.grid(which="minor", axis="y", alpha=0.1)
    ax.set_ylim(*EJE_Y[nombre]["limites"])
    ax.set_yticks(EJE_Y[nombre]["marcas"])
    ax.set_xlabel("Escala de la herida", fontsize=20)
    ax.set_ylabel(f"{magnitud.capitalize()} promedio [{unidad}]", fontsize=20)
    ax.set_xticks(ESCALAS)
    ax.set_xlim(0.17, 1.03)
    ax.tick_params(labelsize=20)
    ax.grid(alpha=0.28, linestyle="--")
    ax.legend(title="Capa", fontsize=18, title_fontsize=18, loc="best")
    # La nota aparece únicamente en la primera figura (campo circular).
    if forma == FORMAS[0] and nombre == VARIABLES[0][3]:
        ax.text(0.02, 0.03, "Dermis: escala 0.2 sin datos (el corte no la alcanza)",
                transform=ax.transAxes, fontsize=18,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.88})
    if GUARDAR_PNG:
        fig.savefig(SALIDA / f"{nombre}_por_capa_{forma}_normalizado.png", dpi=185)
    return fig


def comparar_lado_a_lado(resumen, variable):
    """Tres heridas en una figura: un solo eje Y compartido de verdad."""
    _, magnitud, unidad, nombre = variable
    fig, ejes = plt.subplots(1, 3, figsize=(19, 6.5), sharey=True,
                             layout="constrained")
    fig.suptitle(f"{magnitud.capitalize()} por capa y por µA · electrodos de 3 mm estáticos",
                 fontsize=18)
    for ax, forma in zip(ejes, FORMAS):
        for capa, color, marcador in zip(CAPAS, COLORES[forma], MARCADORES):
            serie = resumen[(resumen.forma == forma) & (resumen.capa == capa)]
            y = serie.set_index("escala").reindex(ESCALAS)[nombre+"_medio_por_uA"].to_numpy()
            ax.plot(ESCALAS, y, color=color, marker=marcador, linewidth=2.4,
                    markersize=6, label=capa)
        if nombre == "densidad":
            ax.set_yscale("log")
        ax.set_ylim(*EJE_Y[nombre]["limites"])
        ax.set_yticks(EJE_Y[nombre]["marcas"])
        ax.tick_params(axis="y", labelleft=True, labelsize=10)
        ax.tick_params(axis="x", labelsize=10)
        ax.set_xticks(ESCALAS)
        ax.set_xlim(0.17, 1.03)
        ax.set_title(forma.capitalize(), fontsize=16)
        ax.set_xlabel("Escala de la herida", fontsize=12)
        ax.set_ylabel(f"Promedio [{unidad}]", fontsize=11)
        ax.grid(alpha=0.28, linestyle="--")
        ax.legend(title="Capa", loc="best", fontsize=10, title_fontsize=10)
        if forma == FORMAS[0] and nombre == "campo":
            ax.text(0.02, 0.03, "Escala 0.2: el corte no alcanza la dermis",
                    transform=ax.transAxes, fontsize=9,
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.88})
    if GUARDAR_PNG:
        fig.savefig(SALIDA / f"COMPARACION_TRES_HERIDAS_{nombre}_EJE_Y_COMPARTIDO.png",
                    dpi=185)
    return fig


def main():
    if not CSV.is_file():
        raise FileNotFoundError(f"Coloca el CSV consolidado junto al script: {CSV}")
    columnas = ["forma_herida", "distancia_electrodos", "ancho_electrodo_mm",
                "escala", "corriente_uA", "indice_punto_curva", "longitud_arco_mm"]
    columnas += [v[0] for v in VARIABLES]
    datos = pd.read_csv(CSV, usecols=columnas)
    datos = datos[(datos.ancho_electrodo_mm == ANCHO_ELECTRODO_MM)
                  & (datos.distancia_electrodos == POSICION)]
    resumen = resumir(datos)
    comprobar_rangos(resumen)
    if GUARDAR_PNG:
        SALIDA.mkdir(exist_ok=True)
        resumen.to_csv(SALIDA / "promedios_por_capa_y_por_uA.csv",
                       index=False, encoding="utf-8-sig")
    for forma in FORMAS:
        for variable in VARIABLES:
            graficar(resumen, forma, variable)
    for variable in VARIABLES:
        comparar_lado_a_lado(resumen, variable)
    print("Seis figuras individuales y dos comparaciones lado a lado listas.")
    print("Ejes Y comunes por magnitud:", EJE_Y)
    print("Tabla resumen:", len(resumen), "filas (3 formas × 5 escalas × 3 capas)")
    plt.show()


if __name__ == "__main__":
    main()


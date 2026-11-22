"""Cuatro gráficas corriente-respuesta para una configuración fija de COMSOL."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Cambia estos tres valores para comparar otra configuración.
ESCALA = 1.0
ANCHO_ELECTRODO_MM = 3
DISTANCIA_ELECTRODOS = "estatica"  # "estatica" o "dinamica"

CARPETA = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
CSV = CARPETA / "resultados_tesis_consolidados4.csv"
CORRIENTES = [1, 3, 5, 7, 9]
FORMAS = ("circular", "rectangular", "triangular")
COLORES = {
    "circular": "#238557",    # Verde
    "rectangular": "#c33f47", # Rojo
    "triangular": "#2359a3",  # Azul
}
MAGNITUDES = (
    ("campo_electrico_V_m", "campo eléctrico", "V/m"),
    ("densidad_corriente_A_m2", "densidad de corriente", "A/m²"),
)


def metricas_por_curva(datos):
    """Máximo y promedio espacial (integral sobre arco / longitud total)."""
    filas = []
    for (forma, corriente), curva in datos.groupby(
        ["forma_herida", "corriente_uA"], sort=False
    ):
        curva = curva.sort_values("indice_punto_curva")
        x = curva["longitud_arco_mm"].to_numpy(dtype=float)
        if len(x) < 2 or x[-1] <= x[0] or np.any(np.diff(x) < 0):
            raise ValueError(f"Arco inválido para {forma}, {corriente} µA")
        fila = {"forma": forma, "corriente_uA": corriente}
        for columna, _, _ in MAGNITUDES:
            y = curva[columna].to_numpy(dtype=float)
            if not np.isfinite(y).all():
                raise ValueError(f"Valores faltantes para {forma}, {corriente} µA")
            fila[f"max_{columna}"] = float(np.max(y))
            fila[f"promedio_{columna}"] = float(np.trapezoid(y, x) / (x[-1] - x[0]))
        filas.append(fila)
    tabla = pd.DataFrame(filas)
    esperado = {(f, c) for f in FORMAS for c in CORRIENTES}
    encontrado = set(zip(tabla["forma"], tabla["corriente_uA"]))
    if encontrado != esperado or len(tabla) != len(esperado):
        raise ValueError(f"Se esperaban 15 curvas; faltan {sorted(esperado - encontrado)}")
    return tabla


def graficar(tabla, columna, nombre, unidad, estadistico):
    fig, ax = plt.subplots(figsize=(12, 7), layout="constrained")
    for forma in FORMAS:
        serie = tabla.loc[tabla["forma"] == forma].set_index("corriente_uA")
        ax.plot(CORRIENTES, serie.loc[CORRIENTES, f"{estadistico}_{columna}"],
                marker="o", markersize=8, linewidth=2.8,
                color=COLORES[forma], label=forma.capitalize())
    etiqueta = "Máximo" if estadistico == "max" else "Promedio espacial"
    ax.set_title(f"{etiqueta} de {nombre} según la corriente\n"
                 f"(escala {ESCALA:g}, electrodo de {ANCHO_ELECTRODO_MM} mm, "
                 f"{DISTANCIA_ELECTRODOS})", fontsize=18, pad=15)
    ax.set_xlabel("Corriente aplicada (µA)", fontsize=16)
    ax.set_ylabel(f"{etiqueta} de {nombre} ({unidad})", fontsize=16)
    ax.set_xticks(CORRIENTES)
    ax.set_xlim(0.5, 9.5)
    ax.set_ylim(bottom=0)
    ax.tick_params(axis="both", labelsize=13)
    ax.grid(alpha=0.3)
    ax.legend(title="Forma de la herida", fontsize=13, title_fontsize=13,
              loc="best")
    return fig


def main():
    if not CSV.is_file():
        raise FileNotFoundError(f"No se encontró {CSV}. Coloca el .py junto al CSV.")
    columnas = ["forma_herida", "distancia_electrodos", "ancho_electrodo_mm",
                "escala", "corriente_uA", "indice_punto_curva", "longitud_arco_mm"]
    columnas += [m[0] for m in MAGNITUDES]
    datos = pd.read_csv(CSV, usecols=columnas)
    datos = datos.loc[
        datos["escala"].eq(ESCALA)
        & datos["ancho_electrodo_mm"].eq(ANCHO_ELECTRODO_MM)
        & datos["distancia_electrodos"].eq(DISTANCIA_ELECTRODOS)
    ]
    if datos.empty:
        raise ValueError("No hay datos con esa escala, ancho y distancia.")
    tabla = metricas_por_curva(datos)
    for columna, nombre, unidad in MAGNITUDES:
        for estadistico in ("max", "promedio"):
            graficar(tabla, columna, nombre, unidad, estadistico)
    print("Se muestran cuatro gráficas para una sola configuración:")
    print(tabla.to_string(index=False))
    plt.show()


if __name__ == "__main__":
    main()

from pathlib import Path


import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd


CARPETA = (
    Path(__file__).resolve().parent
    if "__file__" in globals()
    else Path.cwd()
)

CSV = CARPETA / "resultados_tesis_consolidados4.csv"
SALIDA = CARPETA / "graficas_corrientes_nuevas"

# CONFIGURACIÓN
ESCALAS_A_GRAFICAR = (1.0,)
# Para todas las escalas:
# ESCALAS_A_GRAFICAR = (0.2, 0.4, 0.6, 0.8, 1.0)

ANCHO_ELECTRODO_MM = 3
POSICION = "estatica"  # "estatica" o "dinamica"
CORRIENTES_UA = (1, 3, 5, 7, 9)

GUARDAR_FIGURAS = True
MOSTRAR_FIGURAS = True

# Tamaños de letra.
TAMANO_TITULO = 13
TAMANO_EJES = 30
TAMANO_NUMEROS = 30
TAMANO_LEYENDAS = 17
TAMANO_TITULOS_LEYENDAS = 21

# Menos marcas en los ejes.
INTERVALOS_EJE_X = 4
INTERVALOS_EJE_Y = 6

COLORES = {
    "Triangular": [
        "#9be7f5", "#55b4db", "#328fb9", "#265887", "#13076b"
    ],
    "Circular": [
        "#a8e6a3", "#66c477", "#36a464", "#187348", "#095137"
    ],
    "Rectangular": [
        "#f8b3a9", "#ed817a", "#df554f", "#b62834", "#790c21"
    ],
}

MAGNITUDES = {
    "Campo eléctrico": ("campo_electrico_V_m", "V/m", "campo"),
    "Densidad de corriente": (
        "densidad_corriente_A_m2", "A/m²", "densidad"
    ),
}

COLORES_CAPAS = {
    "Estrato córneo": "#f2f2f2",
    "Epidermis": "#dddddd",
    "Dermis": "#bdbdbd",
}

# Interfaces verticales del código anterior, en mm.
Y_SUPERFICIE = 0.007
Y_ESTRATO_EP = -0.007
Y_EP_DERMIS = -0.317

# El CSV no contiene coordenadas de profundidad.
# True: sombrear con las trayectorias anteriores como referencia.
# False: sombrear solo si se registran límites actuales abajo.
USAR_CAPAS_DE_REFERENCIA = True

# Límites actuales opcionales, EN LONGITUD DE ARCO, no profundidad.
# Formato:
# "Rectangular": {
#     (3, "estatica", 1.0): (inicio_corneo, fin_corneo, fin_epidermis)
# }
# Estas entradas tienen prioridad sobre la geometría de referencia.
LIMITES_CAPAS_MM = {
    "Triangular": {},
    "Circular": {},
    "Rectangular": {},
}


def leer_curvas(datos, forma, escala):
    filtro = (
        (datos["forma_herida"] == forma.lower())
        & (datos["ancho_electrodo_mm"] == ANCHO_ELECTRODO_MM)
        & (datos["distancia_electrodos"] == POSICION)
        & np.isclose(datos["escala"], escala)
    )
    seleccion = datos.loc[filtro]

    if set(seleccion["corriente_uA"].unique()) != set(CORRIENTES_UA):
        raise ValueError(
            f"Faltan corrientes en {forma}, escala {escala}."
        )

    curvas = {}

    for corriente in CORRIENTES_UA:
        curva = seleccion[
            seleccion["corriente_uA"] == corriente
        ].sort_values("indice_punto_curva")

        if (
            curva["indice_curva_archivo"].nunique() != 1
            or curva["indice_punto_curva"].duplicated().any()
        ):
            raise ValueError(
                f"Curvas duplicadas: {forma}, {escala}, {corriente} µA."
            )

        x = curva["longitud_arco_mm"].to_numpy(dtype=float)
        y = curva[
            [configuracion[0] for configuracion in MAGNITUDES.values()]
        ].to_numpy(dtype=float)

        if (
            len(x) < 2
            or not np.isfinite(x).all()
            or not np.isfinite(y).all()
            or np.any(np.diff(x) < 0)
        ):
            raise ValueError(
                f"Curva inválida: {forma}, {escala}, {corriente} µA."
            )

        curvas[corriente] = curva

    return curvas


def limites_de_referencia(forma, escala, longitud):
    """
    Límites estimados usando las trayectorias del código anterior.
    No identifica capas mediante los picos de las curvas.
    """
    if forma == "Rectangular":
        return (
            0.0,
            0.005 - Y_ESTRATO_EP,
            0.005 - Y_EP_DERMIS,
        )

    if forma == "Triangular":
        return (
            0.0,
            longitud * (0.005 - Y_ESTRATO_EP) / (0.005 + escala),
            longitud * (0.005 - Y_EP_DERMIS) / (0.005 + escala),
        )

    if forma == "Circular":
        t = np.linspace(-0.007, 0.5, 20001)

        velocidad = np.pi * np.hypot(
            1.55 * escala * np.sin(np.pi * t),
            1.05 * escala * np.cos(np.pi * t),
        )

        arco = np.r_[
            0.0,
            np.cumsum(
                (velocidad[:-1] + velocidad[1:])
                * np.diff(t)
                / 2
            ),
        ]

        def arco_en_y(y):
            if y < -1.05 * escala:
                # La trayectoria no alcanza esta interfaz.
                return float("inf")

            parametro = (
                np.arcsin(np.clip(-y / (1.05 * escala), -1, 1))
                / np.pi
            )
            return float(np.interp(parametro, t, arco))

        return tuple(
            arco_en_y(y)
            for y in (Y_SUPERFICIE, Y_ESTRATO_EP, Y_EP_DERMIS)
        )

    raise ValueError(f"Forma desconocida: {forma}")


def sombrear_capas(ax, forma, escala, longitud):
    clave = (ANCHO_ELECTRODO_MM, POSICION, escala)
    limites = LIMITES_CAPAS_MM[forma].get(clave)

    referencia = limites is None and USAR_CAPAS_DE_REFERENCIA

    if referencia:
        limites = limites_de_referencia(forma, escala, longitud)

    if limites is None:
        return [], False

    inicio, fin_corneo, fin_epidermis = limites

    if not (0 <= inicio < fin_corneo < fin_epidermis):
        raise ValueError(
            "Los límites de capa deben aumentar en longitud de arco."
        )

    tramos = (
        (inicio, fin_corneo, "Estrato córneo"),
        (fin_corneo, fin_epidermis, "Epidermis"),
        (fin_epidermis, longitud, "Dermis"),
    )

    leyenda_capas = []

    for a, b, capa in tramos:
        b = min(b, longitud)

        if a < b:
            ax.axvspan(
                a,
                b,
                facecolor=COLORES_CAPAS[capa],
                zorder=0,
            )
            leyenda_capas.append(
                Patch(color=COLORES_CAPAS[capa], label=capa)
            )

    for limite in limites:
        if 0 < limite < longitud:
            ax.axvline(
                limite,
                color="#888888",
                linestyle="--",
                linewidth=1,
                zorder=1,
            )

    return leyenda_capas, referencia


def dibujar(forma, escala, magnitud, curvas, limite_y):
    columna, unidad, nombre = MAGNITUDES[magnitud]

    longitud = max(
        curva["longitud_arco_mm"].max()
        for curva in curvas.values()
    )

    fig, ax = plt.subplots(
        figsize=(16, 9),
        layout="constrained",
    )

    capas, referencia = sombrear_capas(
        ax, forma, escala, longitud
    )

    for corriente, color in zip(CORRIENTES_UA, COLORES[forma]):
        curva = curvas[corriente]

        ax.plot(
            curva["longitud_arco_mm"],
            curva[columna],
            color=color,
            linewidth=2.25,
            label=f"{corriente} µA",
            zorder=2,
        )

    ax.set_xlim(0, longitud)
    ax.set_ylim(0, limite_y)

    ax.xaxis.set_major_locator(
        MaxNLocator(nbins=INTERVALOS_EJE_X, min_n_ticks=3)
    )
    ax.yaxis.set_major_locator(
        MaxNLocator(nbins=INTERVALOS_EJE_Y, min_n_ticks=3)
    )
    ax.minorticks_off()
    
    # Eje X: mostrar 0 en lugar de 0.0.
    ax.xaxis.set_major_formatter(
    FuncFormatter(lambda valor, posicion: f"{valor:g}")
)

# Eje Y: ocultar únicamente la etiqueta del cero.
    ax.yaxis.set_major_formatter(
    FuncFormatter(
        lambda valor, posicion: "" if valor == 0 else f"{valor:g}"
    )
)

    ax.set_xlabel(
        "Longitud de arco (mm)",
        fontsize=TAMANO_EJES,
        labelpad=9,
    )
    ax.set_ylabel(
        f"{magnitud} ({unidad})",
        fontsize=TAMANO_EJES,
        labelpad=9,
    )

    posicion_texto = (
        "estáticos" if POSICION == "estatica" else "dinámicos"
    )

    ax.set_title(
        f"{magnitud} · herida {forma.lower()} · escala {escala:g}\n"
        f"Electrodos {ANCHO_ELECTRODO_MM} mm {posicion_texto}",
        fontsize=TAMANO_TITULO,
        pad=16,
    )

    ax.tick_params(
        axis="both",
        labelsize=TAMANO_NUMEROS,
    )
    ax.grid(
        alpha=0.35,
        linestyle="--",
        zorder=0.5,
    )

    leyenda_corrientes = ax.legend(
        title="Corriente aplicada",
        loc="upper right",
        fontsize=TAMANO_LEYENDAS,
        title_fontsize=TAMANO_TITULOS_LEYENDAS,
        framealpha=0.95,
    )

    if capas:
        ax.add_artist(leyenda_corrientes)
        ax.legend(
            handles=capas,
            title="Capas de la piel",
            loc="center right",
            fontsize=TAMANO_LEYENDAS,
            title_fontsize=TAMANO_TITULOS_LEYENDAS,
            framealpha=0.95,
        )

    if referencia:
        fig.supxlabel(
            "Sombreado: límites de referencia de las trayectorias anteriores.",
            fontsize=11,
        )

    if GUARDAR_FIGURAS:
        ruta = SALIDA / (
            f"{nombre}_{forma.lower()}_escala_{escala:g}_"
            f"{ANCHO_ELECTRODO_MM}mm_{POSICION}.png"
        )
        fig.savefig(ruta, dpi=180, bbox_inches="tight")

    return fig


def main():
    if not CSV.is_file():
        raise FileNotFoundError(
            f"Coloca el consolidado nuevo junto al código: {CSV}"
        )

    if ANCHO_ELECTRODO_MM not in (1, 3, 5):
        raise ValueError("El ancho debe ser 1, 3 o 5 mm.")

    if POSICION not in ("estatica", "dinamica"):
        raise ValueError("La posición debe ser 'estatica' o 'dinamica'.")

    if (
        not ESCALAS_A_GRAFICAR
        or any(
            escala not in (0.2, 0.4, 0.6, 0.8, 1.0)
            for escala in ESCALAS_A_GRAFICAR
        )
    ):
        raise ValueError(
            "Las escalas disponibles son 0.2, 0.4, 0.6, 0.8 y 1.0."
        )

    datos = pd.read_csv(CSV, float_precision="round_trip")

    if GUARDAR_FIGURAS:
        SALIDA.mkdir(exist_ok=True)

    for escala in ESCALAS_A_GRAFICAR:
        seleccion = {
            forma: leer_curvas(datos, forma, escala)
            for forma in COLORES
        }

        for magnitud, configuracion in MAGNITUDES.items():
            columna = configuracion[0]

            # Un mismo límite Y para las tres formas de esta magnitud.
            limite_y = 1.08 * max(
                curva[columna].max()
                for curvas in seleccion.values()
                for curva in curvas.values()
            )

            for forma, curvas in seleccion.items():
                fig = dibujar(
                    forma,
                    escala,
                    magnitud,
                    curvas,
                    limite_y,
                )

                if not MOSTRAR_FIGURAS:
                    plt.close(fig)

    print(
        f"Se generaron {6 * len(ESCALAS_A_GRAFICAR)} gráficas "
        "con los datos originales, sin normalizar."
    )

    if MOSTRAR_FIGURAS:
        plt.show()


if __name__ == "__main__":
    main()

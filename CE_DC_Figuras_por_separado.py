from pathlib import Path
import argparse
import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.backends.backend_pdf import PdfPages

CARPETA = Path(__file__).resolve().parent if '__file__' in globals() else Path.cwd()
ARCHIVO = CARPETA / 'resultados_tesis_consolidados4.csv'
ESCALA = 1.0
ANCHO_MM = 3
DISTANCIA = 'estatica'
CORRIENTES_UA = (1, 3, 5, 7, 9)
TAMANO_TITULO = 10
TAMANO_EJES = 23
TAMANO_NUMEROS = 23
TAMANO_LEYENDAS = 15
TAMANO_TITULOS_LEYENDAS = 19
COLORES = {
    'Triangular': ['#9be7f5', '#55b4db', '#328fb9', '#265887', '#13076b'],
    'Circular': ['#a8e6a3', '#66c477', '#36a464', '#187348', '#095137'],
    'Rectangular': ['#f8b3a9', '#ed817a', '#df554f', '#b62834', '#790c21'],
}
COLORES_CAPAS = {'Estrato córneo': '#f2f2f2', 'Epidermis': '#dddddd', 'Dermis': '#bdbdbd'}
Y_SUPERFICIE = 0.007
Y_ESTRATO_EP = -0.007
Y_EP_DERMIS = -0.317
MAGNITUDES = {
    'Campo eléctrico': ('campo_electrico_V_m', 'V/m', 'campo_electrico'),
    'Densidad de corriente': ('densidad_corriente_A_m2', 'A/m²', 'densidad_corriente'),
}


def leer_curvas(ruta, escala, ancho, distancia):
    """Filtra el único CSV y agrupa por forma y corriente, sin mezclar escalas."""
    grupos = {(forma, corriente): [] for forma in COLORES for corriente in CORRIENTES_UA}
    with Path(ruta).open('r', encoding='utf-8-sig', newline='') as archivo:
        lector = csv.DictReader(archivo)
        requeridas = {'forma_herida', 'distancia_electrodos', 'ancho_electrodo_mm',
                      'corriente_uA', 'escala', 'indice_punto_curva', 'longitud_arco_mm',
                      'campo_electrico_V_m', 'densidad_corriente_A_m2'}
        faltantes = requeridas - set(lector.fieldnames or [])
        if faltantes:
            raise ValueError(f'Faltan columnas: {sorted(faltantes)}')
        for fila in lector:
            if not (np.isclose(float(fila['escala']), escala)
                    and np.isclose(float(fila['ancho_electrodo_mm']), ancho)
                    and fila['distancia_electrodos'].strip().lower() == distancia):
                continue
            clave = (fila['forma_herida'].strip().capitalize(), float(fila['corriente_uA']))
            if clave in grupos:
                grupos[clave].append((int(fila['indice_punto_curva']),
                                     float(fila['longitud_arco_mm']),
                                     float(fila['campo_electrico_V_m']),
                                     float(fila['densidad_corriente_A_m2'])))
    curvas = {}
    for clave, puntos in grupos.items():
        if not puntos:
            raise ValueError(f'No hay datos para {clave}, escala={escala}, ancho={ancho}, {distancia}.')
        datos = np.asarray(sorted(puntos))
        if len(np.unique(datos[:, 0])) != len(datos):
            raise ValueError(f'Hay índices de punto repetidos en {clave}; revisa la selección.')
        curva = datos[:, 1:]
        if not np.isfinite(curva).all() or np.any(np.diff(curva[:, 0]) < 0):
            raise ValueError(f'Datos no válidos o arco desordenado en {clave}.')
        curvas[clave] = curva
    return curvas


def limites_capas(forma, longitud_total, escala):

    if forma == 'Circular':
        t = np.linspace(-0.007, 0.5, 20001)
        velocidad = np.pi * np.hypot(1.55 * np.sin(np.pi*t), 1.05 * np.cos(np.pi*t))
        arco = np.r_[0., np.cumsum((velocidad[:-1] + velocidad[1:]) * np.diff(t) / 2)]
        escala_arco = longitud_total / arco[-1]
        def arco_en_y(y):
            razon = -y / (1.05 * escala_arco)
            if razon > 1:
                return longitud_total
            parametro = np.arcsin(np.clip(razon, -1, 1)) / np.pi
            return np.interp(parametro, t, arco) * escala_arco
        limites = (arco_en_y(Y_SUPERFICIE), arco_en_y(Y_ESTRATO_EP), arco_en_y(Y_EP_DERMIS))
    elif forma == 'Triangular':
        # Inicio (-1.55*scale, 0.005), final (-0.05, -scale).
        limites = (0., longitud_total * (0.005 - Y_ESTRATO_EP) / (0.005 + escala),
                   longitud_total * (0.005 - Y_EP_DERMIS) / (0.005 + escala))
    elif forma == 'Rectangular':
        limites = (0., 0.005 - Y_ESTRATO_EP, 0.005 - Y_EP_DERMIS)
    else:
        raise ValueError(forma)
    return tuple(np.clip(limites, 0, longitud_total))


def dibujar(curvas, forma, magnitud, escala, ancho, distancia):
    _, unidad, _ = MAGNITUDES[magnitud]
    columna = 1 if magnitud == 'Campo eléctrico' else 2
    longitud_maxima = max(curvas[forma, i][-1, 0] for i in CORRIENTES_UA)
    inicio_corneo, fin_corneo, fin_epidermis = limites_capas(forma, longitud_maxima, escala)
    fig, ax = plt.subplots(figsize=(13, 7), layout='constrained')
    tramos = ((inicio_corneo, fin_corneo, 'Estrato córneo'),
              (fin_corneo, fin_epidermis, 'Epidermis'),
              (fin_epidermis, longitud_maxima, 'Dermis'))
    for inicio, fin, capa in tramos:
        if fin > inicio:
            ax.axvspan(inicio, fin, facecolor=COLORES_CAPAS[capa], zorder=0)
    for limite in (inicio_corneo, fin_corneo, fin_epidermis):
        if 0 < limite < longitud_maxima:
            ax.axvline(limite, color='#888888', linestyle='--', linewidth=1, zorder=1)
    for corriente, color in zip(CORRIENTES_UA, COLORES[forma]):
        curva = curvas[forma, corriente]
        ax.plot(curva[:, 0], curva[:, columna], color=color, linewidth=2.25,
                label=f'{corriente} µA', zorder=2)
    ax.set_xlim(0, longitud_maxima)
    ax.set_ylim(bottom=0)
    ax.set_xlabel('Longitud de arco (mm)', fontsize=TAMANO_EJES, labelpad=9)
    ax.set_ylabel(f'{magnitud} ({unidad})', fontsize=TAMANO_EJES, labelpad=9)
    posicion = 'estáticos' if distancia == 'estatica' else 'dinámicos'
    ax.set_title(f'{magnitud} · herida {forma.lower()} · electrodos {ancho:g} mm {posicion} · escala {escala:g}',
                 fontsize=TAMANO_TITULO, pad=16)
    ax.tick_params(axis='both', labelsize=TAMANO_NUMEROS)
    ax.grid(alpha=0.35, linestyle='--', zorder=0.5)
    leyenda = ax.legend(title='Corriente aplicada', loc='upper right',
                        fontsize=TAMANO_LEYENDAS, title_fontsize=TAMANO_TITULOS_LEYENDAS, framealpha=0.95)
    ax.add_artist(leyenda)
    ax.legend(handles=[Patch(color=gris, label=capa) for capa, gris in COLORES_CAPAS.items()],
              title='Capas de la piel', loc='center right', ncol=1,
              fontsize=TAMANO_LEYENDAS, title_fontsize=TAMANO_TITULOS_LEYENDAS, framealpha=0.95)
    return fig


def main():
    parser = argparse.ArgumentParser(description='Seis gráficas desde el CSV consolidado de COMSOL.')
    parser.add_argument('--archivo', type=Path, default=ARCHIVO)
    parser.add_argument('--escala', type=float, default=ESCALA)
    parser.add_argument('--ancho', type=float, default=ANCHO_MM)
    parser.add_argument('--distancia', choices=['estatica', 'dinamica'], default=DISTANCIA)
    parser.add_argument('--salida', type=Path, default=CARPETA / 'graficas')
    parser.add_argument('--sin-mostrar', action='store_true')
    args = parser.parse_args()
    curvas = leer_curvas(args.archivo, args.escala, args.ancho, args.distancia)
    args.salida.mkdir(parents=True, exist_ok=True)
    with PdfPages(args.salida / 'graficas_consolidados.pdf') as pdf:
        for forma in COLORES:
            for magnitud, (_, _, nombre) in MAGNITUDES.items():
                fig = dibujar(curvas, forma, magnitud, args.escala, args.ancho, args.distancia)
                fig.savefig(args.salida / f'{forma.lower()}_{nombre}.png', dpi=220)
                pdf.savefig(fig)
                if args.sin_mostrar:
                    plt.close(fig)
    print(f'Se generaron seis gráficas y un PDF en: {args.salida.resolve()}')
    if not args.sin_mostrar:
        plt.show()


if __name__ == '__main__':
    main()

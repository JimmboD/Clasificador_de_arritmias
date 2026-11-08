import os
import wfdb
import numpy as np
import pandas as pd
from pathlib import Path

# Lista oficial de los 48 registros de la base MIT-BIH
REGISTROS_MIT_BIH = [
    '100', '101', '102', '103', '104', '105', '106', '107', '108', '109',
    '111', '112', '113', '114', '115', '116', '117', '118', '119', '121',
    '122', '123', '124', '200', '201', '202', '203', '205', '207', '208',
    '209', '210', '212', '213', '214', '215', '217', '219', '220', '221',
    '222', '223', '228', '230', '231', '232', '233', '234'
]

def buscar_taquicardia_ventricular(ruta_base_datos="Data/Database"):
    """
    Rastrea todos los registros de la base de datos para localizar episodios
    de Taquicardia Ventricular (VT):
    1. Por anotación de ritmo explícita '(VT'.
    2. Por ráfaga de >= 3 latidos ventriculares ('V') consecutivos.
    """
    hallazgos = []
    print(f"Buscando episodios de Taquicardia Ventricular en '{ruta_base_datos}'...")

    for rec in REGISTROS_MIT_BIH:
        path_registro = os.path.join(ruta_base_datos, rec)
        if not os.path.exists(f"{path_registro}.atr"):
            continue

        try:
            ann = wfdb.rdann(path_registro, 'atr')
        except Exception:
            continue

        # 1. Búsqueda por etiqueta de ritmo '(VT'
        for sample, symbol, aux in zip(ann.sample, ann.symbol, ann.aux_note):
            aux_limpio = aux.strip('\x00').strip()
            if symbol == '+' and '(VT' in aux_limpio:
                tiempo_seg = sample / ann.fs
                minutos = int(tiempo_seg // 60)
                segundos = tiempo_seg % 60
                hallazgos.append({
                    'registro': str(rec),
                    'tipo_deteccion': 'Etiqueta de ritmo (VT',
                    'muestra': int(sample),
                    'tiempo': f"{minutos:02d}:{segundos:05.2f}"
                })

        # 2. Búsqueda por ráfaga de >= 3 latidos 'V' consecutivos
        v_consecutivos = 0
        idx_inicio = None

        for i, (sample, symbol) in enumerate(zip(ann.sample, ann.symbol)):
            if symbol == 'V':
                if v_consecutivos == 0:
                    idx_inicio = i
                v_consecutivos += 1
            else:
                if v_consecutivos >= 3:
                    sample_inicio = ann.sample[idx_inicio]
                    tiempo_seg = sample_inicio / ann.fs
                    minutos = int(tiempo_seg // 60)
                    segundos = tiempo_seg % 60
                    hallazgos.append({
                        'registro': str(rec),
                        'tipo_deteccion': f'Ráfaga de {v_consecutivos} latidos V',
                        'muestra': int(sample_inicio),
                        'tiempo': f"{minutos:02d}:{segundos:05.2f}"
                    })
                v_consecutivos = 0
                idx_inicio = None

        if v_consecutivos >= 3:
            sample_inicio = ann.sample[idx_inicio]
            tiempo_seg = sample_inicio / ann.fs
            minutos = int(tiempo_seg // 60)
            segundos = tiempo_seg % 60
            hallazgos.append({
                'registro': str(rec),
                'tipo_deteccion': f'Ráfaga de {v_consecutivos} latidos V',
                'muestra': int(sample_inicio),
                'tiempo': f"{minutos:02d}:{segundos:05.2f}"
            })

    print(f"\nSe encontraron {len(hallazgos)} eventos de Taquicardia Ventricular:\n")
    print(f"{'Registro':<10} | {'Tiempo (MM:SS)':<15} | {'Muestra':<10} | {'Tipo de Detección'}")
    print("-" * 65)
    for r in hallazgos:
        print(f"{r['registro']:<10} | {r['tiempo']:<15} | {r['muestra']:<10} | {r['tipo_deteccion']}")

    return hallazgos

def analizar_intervalos_rr(picos_o_ruta_csv, fs=360):
    """
    Calcula distancias R-R, estadísticas de variabilidad y frecuencia cardíaca promedio (BPM).
    Acepta un arreglo/lista de picos o la ruta a un archivo CSV.
    """
    if isinstance(picos_o_ruta_csv, (str, Path)):
        df = pd.read_csv(picos_o_ruta_csv)
        columna = 'Pulsos' if 'Pulsos' in df.columns else df.columns[0]
        picos = df[columna].values
    else:
        picos = np.asarray(picos_o_ruta_csv)

    if len(picos) < 2:
        print("Insuficientes picos para calcular intervalos R-R.")
        return None, None

    distancias_muestras = np.diff(picos)
    distancias_segundos = distancias_muestras / fs

    estadisticas = {
        'Promedio (Media)': float(np.mean(distancias_segundos)),
        'Mediana': float(np.median(distancias_segundos)),
        'Desviación Estándar': float(np.std(distancias_segundos)),
        'Mínimo': float(np.min(distancias_segundos)),
        'Máximo': float(np.max(distancias_segundos)),
        'BPM Promedio': float(60.0 / np.mean(distancias_segundos))
    }

    print(f"\n--- Análisis de Intervalos R-R (Frecuencia: {fs} Hz) ---")
    print(f"Total de ciclos analizados: {len(distancias_segundos)}")
    print("-" * 45)
    for k, v in estadisticas.items():
        if 'BPM' in k:
            print(f"{k}: {v:.1f} latidos/min")
        else:
            print(f"{k}: {v:.4f} s")
    print("-" * 45)

    return distancias_segundos, estadisticas

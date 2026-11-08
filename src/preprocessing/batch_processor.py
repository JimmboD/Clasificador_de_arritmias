import os
import wfdb
import numpy as np
from pathlib import Path
from .preprocessor import PreprocesadorECG

def procesar_registro_ecg(ruta_registro, carpeta_destino='Data/Raw_data'):
    """
    Lee un archivo de la base de datos WFDB (.dat/.hea/.atr),
    aplica el preprocesamiento y guarda un archivo comprimido .npz.
    """
    ruta_norm = str(Path(ruta_registro).with_suffix(''))
    nombre_base = Path(ruta_norm).name

    try:
        record = wfdb.rdrecord(ruta_norm)
        annotation = wfdb.rdann(ruta_norm, 'atr')
    except Exception as e:
        print(f"[{nombre_base}] No se pudo leer el registro: {e}")
        return False

    voltaje_ch0 = record.p_signal[:, 0]
    etiquetas_oficiales = list(zip(annotation.sample, annotation.symbol))

    preprocesador = PreprocesadorECG(frecuencia_muestreo=record.fs, frecuencia_corte=40)
    resultado = preprocesador.pre_procesar(voltaje_ch0, etiquetas_bd=etiquetas_oficiales)

    X_dataset = resultado['X']
    Y_etiquetas = resultado['Y']

    os.makedirs(carpeta_destino, exist_ok=True)
    nombre_archivo = os.path.join(carpeta_destino, f'{nombre_base}.npz')
    np.savez_compressed(nombre_archivo, X=X_dataset, Y=Y_etiquetas)

    print(f"[{nombre_base}] ¡Procesado con éxito! Latidos: {len(X_dataset)} | Tensor: {X_dataset.shape}")
    return True

def procesar_base_datos(carpeta_base='Data/Database', carpeta_destino='Data/Raw_data', inicio=100, fin=234):
    """
    Ejecuta el procesamiento por lotes para el rango de registros especificado.
    """
    print(f"=== INICIANDO PROCESAMIENTO POR LOTES ({inicio} al {fin}) ===")
    procesados = 0
    ignorados = 0

    for i in range(inicio, fin + 1):
        ruta_actual = os.path.join(carpeta_base, str(i))
        # Verificar si al menos existe el archivo .hea o .dat
        if os.path.exists(f"{ruta_actual}.hea") or os.path.exists(f"{ruta_actual}.dat"):
            ok = procesar_registro_ecg(ruta_actual, carpeta_destino=carpeta_destino)
            if ok:
                procesados += 1
            else:
                ignorados += 1
        else:
            ignorados += 1

    print("===============================================================")
    print(f"Procesamiento masivo finalizado. Éxitos: {procesados} | Ignorados: {ignorados}")
    return procesados

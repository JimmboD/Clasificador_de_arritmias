import io
import os
import re
import numpy as np
import pandas as pd
from scipy import signal
from .preprocessor import PreprocesadorECG
from ..models.classifier import ClasificadorArritmiasCNN
from ..dataset.dataset_manager import CLASES_VIABLES

# Nombres clínicos estándar según AAMI EC57
NOMBRES_CLINICOS = {
    'N': 'Latido Normal (Sinus Beat)',
    '/': 'Latido Estimulado por Marcapasos (Paced Beat)',
    'R': 'Bloqueo de Rama Derecha (RBBB)',
    'L': 'Bloqueo de Rama Izquierda (LBBB)',
    'V': 'Contracción Ventricular Prematura (PVC)',
    'A': 'Contracción Auricular Prematura (PAC)',
    'f': 'Fusión de Latido Marcapasos y Normal',
    'F': 'Fusión de Latido Ventricular y Normal',
    'No_Clasificable': 'Rechazo Clínico / Incertidumbre Diagnóstica'
}

ETIQUETAS_FINALES = CLASES_VIABLES + ['No_Clasificable']

def cargar_senal_csv(archivo_o_ruta, fs_por_defecto=360):
    """
    Lee un archivo CSV o de texto con datos de electrocardiograma (voltaje vs tiempo o solo voltaje).
    Detecta automáticamente delimitadores (coma, punto y coma, tabulador, espacio)
    y separadores decimales.
    
    Retorna:
    - voltaje: array 1D numpy (float64)
    - fs_estimada: frecuencia de muestreo detectada o la provista por defecto
    - tiempo: array 1D numpy (segundos)
    """
    if hasattr(archivo_o_ruta, 'read'):
        contenido = archivo_o_ruta.read()
        if isinstance(contenido, bytes):
            contenido = contenido.decode('utf-8', errors='replace')
        buffer = io.StringIO(contenido)
    else:
        with open(archivo_o_ruta, 'r', encoding='utf-8', errors='replace') as f:
            contenido = f.read()
        buffer = io.StringIO(contenido)

    # Detección de separador decimal con coma (ej. 1,234; 0,56)
    primera_linea = buffer.readline()
    buffer.seek(0)
    
    # Intentar leer con pandas detectando automáticamente el separador
    try:
        df = pd.read_csv(buffer, sep=None, engine='python')
    except Exception:
        buffer.seek(0)
        df = pd.read_csv(buffer, sep=',', decimal='.')

    # Si se cargó como un único string con comas decimales y punto y coma
    if df.shape[1] == 1 and ';' in str(df.iloc[0, 0]):
        buffer.seek(0)
        df = pd.read_csv(buffer, sep=';', decimal=',')

    # Limpiar columnas vacías
    df = df.dropna(how='all', axis=1)

    if df.empty or df.shape[1] == 0:
        raise ValueError("El archivo CSV no contiene columnas con datos legibles.")

    cols = list(df.columns)
    col_tiempo = None
    col_voltaje = None

    # 1. Identificar columnas por nombres
    for col in cols:
        nombre = str(col).lower().strip()
        if any(k in nombre for k in ['time', 'tiempo', 't', 'sec', 'seg', 'timestamp']):
            col_tiempo = col
        elif any(k in nombre for k in ['volt', 'ecg', 'signal', 'ch0', 'lead', 'v', 'mv', 'val']):
            col_voltaje = col

    # 2. Si no se identificó por nombre, evaluar por posición o tipo
    if col_voltaje is None:
        if df.shape[1] == 1:
            col_voltaje = cols[0]
        else:
            col_tiempo = cols[0]
            col_voltaje = cols[1]

    # Convertir a valores numéricos
    voltaje = pd.to_numeric(df[col_voltaje].astype(str).str.replace(',', '.'), errors='coerce').dropna().values

    if len(voltaje) < 100:
        raise ValueError(f"Señal demasiado corta ({len(voltaje)} puntos). Se requieren al menos 100 muestras.")

    # Estimar frecuencia de muestreo si hay columna de tiempo
    fs_estimada = float(fs_por_defecto)
    tiempo = None

    if col_tiempo is not None and col_tiempo in df:
        t_vals = pd.to_numeric(df[col_tiempo].astype(str).str.replace(',', '.'), errors='coerce').dropna().values
        if len(t_vals) == len(voltaje) and len(t_vals) > 1:
            diffs = np.diff(t_vals)
            diffs_pos = diffs[diffs > 0]
            if len(diffs_pos) > 0:
                dt_medio = np.median(diffs_pos)
                if dt_medio > 0:
                    fs_calculada = 1.0 / dt_medio
                    # Si la fs calculada tiene sentido fisiológico (ej. entre 50 y 2000 Hz)
                    if 30 <= fs_calculada <= 5000:
                        fs_estimada = round(fs_calculada, 1)
                        tiempo = t_vals

    if tiempo is None:
        tiempo = np.arange(len(voltaje)) / fs_estimada

    return voltaje, fs_estimada, tiempo

def analizar_senal_ecg(voltaje, fs=360, modelo="6.0", umbral_pendiente=None):
    """
    Ejecuta el pipeline completo para una señal externa 1D:
    1. Filtro digital Butterworth pasa-bajas 40 Hz (fase cero).
    2. Segmentación de picos R.
    3. Ventana asimétrica (0.75 s) y remuestreo estándar a 270 puntos.
    4. Extracción de espectrogramas STFT 2D normalizados (29, 112).
    5. Inferencia por latido con la Red Neuronal Convolucional (CNN).
    6. Cálculo de métricas hemodinámicas (BPM, intervalos R-R) y severidad clínica.
    """
    preprocesador = PreprocesadorECG(frecuencia_muestreo=fs, frecuencia_corte=40)
    
    # 1. Filtro Butterworth
    voltaje_filtrado = preprocesador.filtros_digitales(voltaje)

    # 2. Segmentación de picos R
    picos_r = preprocesador.segmentar_pulsos(voltaje_filtrado, umbral_pendiente=umbral_pendiente)

    # Si con el umbral rígido no se detectaron picos (ej. baja amplitud o morfologías aberrantes anchas),
    # aplicar umbral adaptativo sobre el percentil 95 de la derivada positiva
    if len(picos_r) == 0 and umbral_pendiente is None:
        diff_filtrada = np.diff(voltaje_filtrado)
        pos_diffs = diff_filtrada[diff_filtrada > 0]
        if len(pos_diffs) > 0:
            umbral_adaptativo = float(np.percentile(pos_diffs, 95) * 0.7)
            if umbral_adaptativo > 0.005:
                picos_r = preprocesador.segmentar_pulsos(voltaje_filtrado, umbral_pendiente=umbral_adaptativo)

    if len(picos_r) == 0:
        return {
            "error": "No se detectaron complejos QRS / picos R en la señal proporcionada. Verifica la frecuencia de muestreo y la ganancia."
        }

    # 3. Ventanas y remuestreo a 270 puntos
    marcadores = preprocesador.generar_marcadores_pulsos(voltaje_filtrado, picos_r, tiempo_ventana=0.75)

    if len(marcadores) == 0:
        return {
            "error": "La señal es demasiado corta para formar al menos una ventana de latido completa (0.75 segundos)."
        }

    muestras_270 = []
    tensores_stft = []
    picos_validos = []

    muestras_totales_ventana = int(0.75 * fs)
    muestras_izq = int(muestras_totales_ventana * 0.35)

    for (inicio, fin), pico_r in zip(marcadores, picos_r):
        ventana = voltaje_filtrado[inicio:fin]
        ventana_270 = preprocesador.estandarizar_ventana(ventana, muestras_esperadas=270)
        stft_dict = preprocesador.obtener_tensor_espectrograma(ventana_270)
        
        muestras_270.append(ventana_270)
        tensores_stft.append(stft_dict["datos"])
        picos_validos.append(pico_r)

    tensores_stft_np = np.array(tensores_stft)

    # 4. Clasificación CNN
    clasificador = ClasificadorArritmiasCNN(version_o_ruta=modelo)
    if clasificador.modelo is None:
        raise RuntimeError(f"No se pudo inicializar el modelo CNN '{modelo}'.")

    predicciones_num, probas = clasificador.predecir(tensores_stft_np)

    num_salida = clasificador.modelo.output_shape[-1]
    etiquetas_modelo = ETIQUETAS_FINALES if num_salida >= 9 else CLASES_VIABLES

    # 5. Métricas de intervalo R-R y BPM
    tiempos_seg = [p / fs for p in picos_validos]
    rr_intervals_ms = []
    bpm_instantaneo = []

    for i in range(len(picos_validos)):
        if i == 0:
            if len(picos_validos) > 1:
                rr = (picos_validos[1] - picos_validos[0]) / fs * 1000.0
            else:
                rr = 800.0
        else:
            rr = (picos_validos[i] - picos_validos[i - 1]) / fs * 1000.0

        rr_intervals_ms.append(round(rr, 1))
        bpm = 60000.0 / rr if rr > 0 else 75.0
        bpm_instantaneo.append(round(bpm, 1))

    bpm_medio = float(np.mean(bpm_instantaneo)) if len(bpm_instantaneo) > 0 else 0.0

    # 6. Estructurar lista de pulsos con diagnóstico clínico
    pulsos_resultado = []
    conteo_clases = {}

    for idx, (pico, t_seg, rr_ms, bpm_inst, pred_idx, dist_proba, w270, s29_112) in enumerate(
        zip(picos_validos, tiempos_seg, rr_intervals_ms, bpm_instantaneo, predicciones_num, probas, muestras_270, tensores_stft)
    ):
        clase_str = etiquetas_modelo[pred_idx] if pred_idx < len(etiquetas_modelo) else 'No_Clasificable'
        confianza = float(dist_proba[pred_idx]) * 100.0
        nombre_clinico = NOMBRES_CLINICOS.get(clase_str, clase_str)

        conteo_clases[clase_str] = conteo_clases.get(clase_str, 0) + 1

        # Nivel de severidad clínica
        if clase_str == 'N':
            severidad = 'Normal'
            badge_class = 'badge-success'
        elif clase_str in ['V', 'F']:
            severidad = 'Arritmia Ventricular (Crítica)'
            badge_class = 'badge-danger'
        elif clase_str in ['A', 'R', 'L']:
            severidad = 'Anomalía Auricular / Bloqueo'
            badge_class = 'badge-warning'
        elif clase_str in ['/', 'f']:
            severidad = 'Ritmo Marcapasos'
            badge_class = 'badge-info'
        else:
            severidad = 'Rechazo Clínico / Ruido'
            badge_class = 'badge-outline'

        minutos = int(t_seg // 60)
        segundos = t_seg % 60
        t_formateado = f"{minutos:02d}:{segundos:05.2f}"

        # Vector de probabilidades formateado por clase
        probas_dict = {
            etiquetas_modelo[i]: round(float(dist_proba[i]) * 100, 2)
            for i in range(min(len(etiquetas_modelo), len(dist_proba)))
        }

        pulsos_resultado.append({
            "id": idx + 1,
            "muestra_r": int(pico),
            "tiempo_seg": round(float(t_seg), 3),
            "tiempo_str": t_formateado,
            "rr_ms": rr_ms,
            "bpm": bpm_inst,
            "clase": clase_str,
            "nombre_clinico": nombre_clinico,
            "confianza": round(confianza, 1),
            "severidad": severidad,
            "badge_class": badge_class,
            "distribucion_probas": probas_dict,
            # Forma de onda 1D (270 puntos)
            "corte_1d": [round(float(v), 4) for v in w270],
            # Espectrograma 2D (29x112)
            "espectrograma_2d": [[round(float(val), 3) for val in fila] for fila in s29_112]
        })

    duracion_total_s = float(len(voltaje) / fs)

    # Detección de posibles ráfagas ventriculares consecutivas (alerta de taquicardia)
    alertas_taquicardia = []
    v_contiguos = 0
    idx_inicio = None
    for i, p in enumerate(pulsos_resultado):
        if p["clase"] == 'V':
            if v_contiguos == 0:
                idx_inicio = i
            v_contiguos += 1
        else:
            if v_contiguos >= 3:
                alertas_taquicardia.append({
                    "inicio_pulso": idx_inicio + 1,
                    "fin_pulso": i,
                    "latidos_v": v_contiguos,
                    "tiempo_inicio": pulsos_resultado[idx_inicio]["tiempo_str"]
                })
            v_contiguos = 0
            idx_inicio = None
    if v_contiguos >= 3:
        alertas_taquicardia.append({
            "inicio_pulso": idx_inicio + 1,
            "fin_pulso": len(pulsos_resultado),
            "latidos_v": v_contiguos,
            "tiempo_inicio": pulsos_resultado[idx_inicio]["tiempo_str"]
        })

    # Resumen general
    total_pulsos = len(pulsos_resultado)
    normales = conteo_clases.get('N', 0)
    anormales = total_pulsos - normales
    tasa_arritmias = round((anormales / total_pulsos * 100), 2) if total_pulsos > 0 else 0.0

    # Muestra de la señal filtrada continua (primeros 2500 puntos) para el monitor global
    max_preview = min(len(voltaje_filtrado), 2500)
    preview_senal = [round(float(val), 4) for val in voltaje_filtrado[:max_preview]]
    picos_preview = [int(p) for p in picos_validos if p < max_preview]

    return {
        "resumen": {
            "total_pulsos": total_pulsos,
            "duracion_seg": round(duracion_total_s, 2),
            "fs": float(fs),
            "bpm_medio": round(bpm_medio, 1),
            "bpm_min": round(float(np.min(bpm_instantaneo)), 1) if bpm_instantaneo else 0,
            "bpm_max": round(float(np.max(bpm_instantaneo)), 1) if bpm_instantaneo else 0,
            "normales": normales,
            "anormales": anormales,
            "tasa_arritmias_pct": tasa_arritmias,
            "distribucion_clases": conteo_clases,
            "alertas_taquicardia": alertas_taquicardia,
            "modelo_usado": modelo
        },
        "preview_senal": preview_senal,
        "picos_preview": picos_preview,
        "pulsos": pulsos_resultado
    }

def extraer_registro_a_csv(registro, duracion_seg=20.0, inicio_seg=0.0, canal=0, ruta_base="Data/Database", ruta_salida=None):
    """
    Extrae un segmento puro de señal cruda (Voltaje vs Tiempo) directamente desde
    la base de datos natural MIT-BIH y lo guarda en formato CSV.
    
    GARANTÍA DE CERO FUGA DE DATOS (Data Leakage Free):
    - NO incluye etiquetas médicas, anotaciones oficiales (.atr), marcas de picos R ni diagnósticos.
    - Exporta exclusivamente las columnas físicas 'tiempo' (en segundos) y 'voltaje' (en mV).
    - La señal queda lista para ingresar de forma completamente ciega al pipeline de análisis
      (filtrado Butterworth, segmentación R, STFT y predicción CNN).
    
    Parámetros:
    - registro: Identificador del registro (ej: '100', '200')
    - duracion_seg: Duración del corte en segundos (ej: 20)
    - inicio_seg: Offset de inicio en segundos (ej: 0)
    - canal: Canal de derivación a extraer (por defecto 0, MLII)
    - ruta_base: Directorio donde residen los archivos .hea y .dat
    - ruta_salida: Ruta del CSV destino. Si es None, se guarda en 'Data/Extraidos/'
    
    Retorna:
    - dict con metadatos del archivo generado: ruta, nombre, muestras, fs, duracion_seg, etc.
    """
    import wfdb

    registro_str = str(registro).strip()
    ruta_norm = os.path.join(ruta_base, registro_str)

    if not (os.path.exists(f"{ruta_norm}.hea") or os.path.exists(f"{ruta_norm}.dat")):
        raise FileNotFoundError(f"No se encontró el registro '{registro_str}' en '{ruta_base}'.")

    header = wfdb.rdheader(ruta_norm)
    fs = float(header.fs)
    total_muestras = int(header.sig_len)
    duracion_total = total_muestras / fs

    inicio_seg = max(0.0, float(inicio_seg))
    duracion_seg = max(1.0, float(duracion_seg))

    sampfrom = int(inicio_seg * fs)
    if sampfrom >= total_muestras:
        raise ValueError(f"El inicio ({inicio_seg}s) supera la duración total del registro ({duracion_total:.1f}s).")

    sampto = min(total_muestras, sampfrom + int(duracion_seg * fs))
    muestras_extraidas = sampto - sampfrom

    record = wfdb.rdrecord(ruta_norm, channels=[canal], sampfrom=sampfrom, sampto=sampto)
    voltaje = record.p_signal[:, 0]

    # Vector de tiempo relativo en segundos dentro del corte
    tiempo = np.round(np.arange(muestras_extraidas) / fs, 5)
    voltaje_redondeado = np.round(voltaje, 5)

    # DataFrame SIN NINGUNA ANOTACIÓN O METADATO CLÍNICO (100% ciega / cero fuga)
    df = pd.DataFrame({
        "tiempo": tiempo,
        "voltaje": voltaje_redondeado
    })

    if ruta_salida is None:
        carpeta_extraidos = os.path.join("Data", "Extraidos")
        os.makedirs(carpeta_extraidos, exist_ok=True)
        if inicio_seg > 0:
            nombre_archivo = f"ecg_registro_{registro_str}_{int(duracion_seg)}s_desde_{int(inicio_seg)}s.csv"
        else:
            nombre_archivo = f"ecg_registro_{registro_str}_{int(duracion_seg)}s.csv"
        ruta_salida = os.path.join(carpeta_extraidos, nombre_archivo)
    else:
        carpeta_padre = os.path.dirname(ruta_salida)
        if carpeta_padre:
            os.makedirs(carpeta_padre, exist_ok=True)
        nombre_archivo = os.path.basename(ruta_salida)

    df.to_csv(ruta_salida, index=False)

    return {
        "success": True,
        "registro": registro_str,
        "duracion_seg": round(float(muestras_extraidas / fs), 2),
        "inicio_seg": round(inicio_seg, 2),
        "puntos": int(muestras_extraidas),
        "fs": fs,
        "ruta_archivo": ruta_salida.replace("\\", "/"),
        "nombre_archivo": nombre_archivo,
        "total_muestras_registro": total_muestras,
        "duracion_total_registro_seg": round(duracion_total, 2)
    }


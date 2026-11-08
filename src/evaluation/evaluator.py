import os
import wfdb
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from pathlib import Path

from ..models.classifier import ClasificadorArritmiasCNN
from ..preprocessing.preprocessor import PreprocesadorECG
from ..dataset.dataset_manager import CLASES_VIABLES

ETIQUETAS_FINALES_9X9 = CLASES_VIABLES + ['No_Clasificable']

def evaluar_modelo_desde_npz(ruta_npz, version_modelo="6.0", mostrar_grafica=True, ruta_guardar_grafica=None):
    """
    Evalúa el modelo sobre un dataset .npz preprocesado.
    Calcula matriz de confusión estricta de 9x9 (incluyendo rechazo) y reporte de métricas.
    """
    print(f"\n=== EVALUANDO MODELO (V{version_modelo}) DESDE: {ruta_npz} ===")
    if not os.path.exists(ruta_npz):
        print(f"ERROR: No se encontró el archivo '{ruta_npz}'.")
        return None

    datos = np.load(ruta_npz, allow_pickle=True)
    X_test = datos['X']
    Y_reales = datos['Y']
    print(f"-> Total de latidos cargados: {len(X_test)}")

    if len(X_test) == 0:
        print("Dataset vacío. Abortando.")
        return None

    # Inicializar motor de inferencia
    clasificador = ClasificadorArritmiasCNN(version_o_ruta=version_modelo)
    if clasificador.modelo is None:
        return None

    predicciones_num, _ = clasificador.predecir(X_test)

    # Mapeo a 9 clases (índice 8 = No_Clasificable)
    mapeo_clases = {clase: idx for idx, clase in enumerate(CLASES_VIABLES)}
    Y_reales_num = np.array([mapeo_clases.get(str(etiqueta), 8) for etiqueta in Y_reales])

    # Si el modelo tiene salida de 8 clases (pre-5.0) y predijo 8 clases
    num_clases_salida = clasificador.modelo.output_shape[-1]
    etiquetas_eval = ETIQUETAS_FINALES_9X9 if num_clases_salida >= 9 else CLASES_VIABLES
    num_labels = len(etiquetas_eval)

    if num_clases_salida == 8:
        # Filtrar solo válidos para modelos pre-5.0
        mascara_validos = Y_reales_num < 8
        Y_reales_num = Y_reales_num[mascara_validos]
        predicciones_num = predicciones_num[mascara_validos]

    matriz_conf = confusion_matrix(Y_reales_num, predicciones_num, labels=np.arange(num_labels))

    print("\n=== REPORTE DE CLASIFICACIÓN ===")
    reporte = classification_report(
        Y_reales_num, predicciones_num,
        target_names=etiquetas_eval,
        labels=np.arange(num_labels),
        zero_division=0
    )
    print(reporte)

    if mostrar_grafica or ruta_guardar_grafica:
        plt.figure(figsize=(10, 8))
        sns.heatmap(matriz_conf, annot=True, fmt='d', cmap='Blues',
                    xticklabels=etiquetas_eval, yticklabels=etiquetas_eval)
        plt.title(f"Matriz de Confusión - Modelo V{version_modelo}\nDataset: {Path(ruta_npz).name}")
        plt.xlabel("Predicción del Modelo")
        plt.ylabel("Etiqueta Real (Ground Truth)")
        plt.tight_layout()

        if ruta_guardar_grafica:
            os.makedirs(os.path.dirname(ruta_guardar_grafica), exist_ok=True)
            plt.savefig(ruta_guardar_grafica, dpi=300)
            print(f"Gráfica guardada en: {ruta_guardar_grafica}")

        if mostrar_grafica:
            plt.show()
        else:
            plt.close()

    return {"matriz": matriz_conf, "reporte": reporte}

def evaluar_paciente_crudo(ruta_registro, version_modelo="6.0", mostrar_grafica=True, ruta_guardar_grafica=None):
    """
    Ejecuta el pipeline completo de extremo a extremo:
    1. Lee la señal cruda ECG y anotaciones WFDB (.dat/.hea/.atr).
    2. Aplica filtro, detección de picos R y extracción de espectrogramas.
    3. Clasifica con la CNN (V6.0 con rechazo por defecto).
    4. Genera matriz de confusión y reporte clínico.
    """
    ruta_norm = str(Path(ruta_registro).with_suffix(''))
    nombre_paciente = Path(ruta_norm).name
    print(f"\n=== EVALUACIÓN END-TO-END DE PACIENTE CRUDO: {nombre_paciente} ===")

    try:
        record = wfdb.rdrecord(ruta_norm)
        annotation = wfdb.rdann(ruta_norm, 'atr')
    except Exception as e:
        print(f"ERROR: No se pudo cargar el registro WFDB '{ruta_norm}': {e}")
        return None

    voltaje_ch0 = record.p_signal[:, 0]
    etiquetas_oficiales = list(zip(annotation.sample, annotation.symbol))
    print(f"-> Señal leída: {len(voltaje_ch0)} muestras @ {record.fs} Hz | Anotaciones: {len(etiquetas_oficiales)}")

    preprocesador = PreprocesadorECG(frecuencia_muestreo=record.fs, frecuencia_corte=40)
    resultado = preprocesador.pre_procesar(voltaje_ch0, etiquetas_bd=etiquetas_oficiales)

    X_test = resultado['X']
    Y_reales = resultado['Y']
    print(f"-> Latidos procesados y convertidos a tensores: {len(X_test)}")

    if len(X_test) == 0:
        print("No se extrajeron latidos válidos. Abortando.")
        return None

    clasificador = ClasificadorArritmiasCNN(version_o_ruta=version_modelo)
    if clasificador.modelo is None:
        return None

    predicciones_num, _ = clasificador.predecir(X_test)

    mapeo_clases = {clase: idx for idx, clase in enumerate(CLASES_VIABLES)}
    Y_reales_num = np.array([mapeo_clases.get(etiqueta, 8) for etiqueta in Y_reales])

    num_clases_salida = clasificador.modelo.output_shape[-1]
    etiquetas_eval = ETIQUETAS_FINALES_9X9 if num_clases_salida >= 9 else CLASES_VIABLES
    num_labels = len(etiquetas_eval)

    if num_clases_salida == 8:
        mascara_validos = Y_reales_num < 8
        Y_reales_num = Y_reales_num[mascara_validos]
        predicciones_num = predicciones_num[mascara_validos]

    matriz_conf = confusion_matrix(Y_reales_num, predicciones_num, labels=np.arange(num_labels))

    print(f"\n=== REPORTE CLÍNICO DEL PACIENTE {nombre_paciente} ===")
    reporte = classification_report(
        Y_reales_num, predicciones_num,
        target_names=etiquetas_eval,
        labels=np.arange(num_labels),
        zero_division=0
    )
    print(reporte)

    if mostrar_grafica or ruta_guardar_grafica:
        plt.figure(figsize=(10, 8))
        sns.heatmap(matriz_conf, annot=True, fmt='d', cmap='Blues',
                    xticklabels=etiquetas_eval, yticklabels=etiquetas_eval)
        plt.title(f"Rendimiento Crudo End-to-End (V{version_modelo})\nPaciente: {nombre_paciente}")
        plt.xlabel("Predicción de la CNN (con Rechazo)")
        plt.ylabel("Diagnóstico del Cardiólogo (Ground Truth)")
        plt.tight_layout()

        if ruta_guardar_grafica:
            os.makedirs(os.path.dirname(ruta_guardar_grafica), exist_ok=True)
            plt.savefig(ruta_guardar_grafica, dpi=300)
            print(f"Gráfica guardada en: {ruta_guardar_grafica}")

        if mostrar_grafica:
            plt.show()
        else:
            plt.close()

    return {"matriz": matriz_conf, "reporte": reporte}

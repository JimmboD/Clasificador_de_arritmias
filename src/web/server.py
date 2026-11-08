import os
import glob
import json
import threading
import webbrowser
import numpy as np
import wfdb
from pathlib import Path
from flask import Flask, render_template, request, jsonify

# Importar módulos del pipeline src
from ..preprocessing.preprocessor import PreprocesadorECG
from ..preprocessing.batch_processor import procesar_registro_ecg, procesar_base_datos
from ..dataset.dataset_manager import (
    unificar_datasets,
    filtrar_dataset,
    CLASES_VIABLES
)
from ..dataset.inspector import inspeccionar_npz
from ..models.classifier import ClasificadorArritmiasCNN
from ..models.trainer import entrenar_modelo
from ..evaluation.clinical_metrics import buscar_taquicardia_ventricular, REGISTROS_MIT_BIH
from ..preprocessing.csv_analyzer import cargar_senal_csv, analizar_senal_ecg, extraer_registro_a_csv
import io
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report

directorio_web = Path(__file__).resolve().parent
template_dir = directorio_web / "templates"
static_dir = directorio_web / "static"

app = Flask(
    __name__,
    template_folder=str(template_dir),
    static_folder=str(static_dir)
)

ETIQUETAS_FINALES_9X9 = CLASES_VIABLES + ['No_Clasificable']

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/info", methods=["GET"])
def api_info():
    """Retorna el estado de la base de datos, modelos y archivos .npz disponibles."""
    # 1. Registros en Database
    records = []
    database_dir = Path("Data/Database")
    if database_dir.exists():
        for dat in sorted(database_dir.glob("*.dat")):
            records.append(dat.stem)

    # 2. Archivos en Raw_data
    raw_dir = Path("Data/Raw_data")
    npz_files = []
    unificado_ok = False
    limpio_ok = False
    total_limpio = 0

    if raw_dir.exists():
        for f in sorted(raw_dir.glob("*.npz")):
            if f.name == "dataset_unificado.npz":
                unificado_ok = True
            elif f.name == "dataset_limpio_y_unificado.npz":
                limpio_ok = True
                try:
                    d = np.load(f)
                    total_limpio = len(d['Y'])
                except Exception:
                    pass
            else:
                npz_files.append(f.name)

    # 3. Modelos disponibles
    models_dir = Path("Data/Models")
    available_models = []
    if Path("mejor_modelo.keras").exists():
        available_models.append("mejor_modelo.keras")
    if models_dir.exists():
        for m in sorted(models_dir.glob("*.keras")):
            available_models.append(m.name)

    return jsonify({
        "status": "online",
        "records_count": len(records),
        "records": records if records else REGISTROS_MIT_BIH,
        "npz_files": npz_files,
        "unificado_ok": unificado_ok,
        "limpio_ok": limpio_ok,
        "total_limpio": total_limpio,
        "available_models": available_models,
        "clases_viables": CLASES_VIABLES
    })

# Cache del último análisis de CSV externo para consulta y exportación
ultimo_analisis_csv = None

@app.route("/api/analyze/csv", methods=["POST"])
def api_analyze_csv():
    """
    Workflow Principal:
    Recibe un archivo CSV (o texto plano con voltaje vs tiempo), aplica filtro
    digital Butterworth, segmenta picos R, genera los cortes de cada pulso (270 pts),
    calcula la transformada STFT (29x112) y ejecuta la clasificación pulso por pulso.
    """
    global ultimo_analisis_csv
    try:
        fs = 360.0
        model_version = "6.0"

        # 1. Caso archivo subido por formulario multipart
        if 'file' in request.files:
            archivo = request.files['file']
            if archivo.filename == '':
                return jsonify({"error": "No se seleccionó ningún archivo."}), 400

            fs_param = request.form.get("fs", "360").strip()
            try:
                fs = float(fs_param) if fs_param else 360.0
            except ValueError:
                fs = 360.0

            model_version = request.form.get("model", "6.0").strip()
            voltaje, fs_detectada, tiempo = cargar_senal_csv(archivo.stream, fs_por_defecto=fs)

        elif request.is_json:
            # 2. Caso ruta enviada en JSON
            data = request.get_json(silent=True) or {}
            ruta = data.get("file_path", "").strip()
            fs = float(data.get("fs", 360))
            model_version = data.get("model", "6.0").strip()

            if not ruta or not os.path.exists(ruta):
                return jsonify({"error": f"No se encontró el archivo en la ruta '{ruta}'"}), 404

            voltaje, fs_detectada, tiempo = cargar_senal_csv(ruta, fs_por_defecto=fs)
        else:
            return jsonify({"error": "Formato de petición no reconocido. Envía un archivo CSV multipart o JSON."}), 400

        # Ejecutar análisis pulso por pulso
        resultado = analizar_senal_ecg(voltaje, fs=fs_detectada, modelo=model_version)
        if "error" in resultado:
            return jsonify(resultado), 400

        ultimo_analisis_csv = resultado
        return jsonify(resultado)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Error al procesar archivo CSV: {str(e)}"}), 500

@app.route("/api/analyze/export-csv", methods=["GET"])
def api_export_csv():
    """Exporta el reporte pulso por pulso del último análisis como archivo CSV descargable."""
    global ultimo_analisis_csv
    if not ultimo_analisis_csv or "pulsos" not in ultimo_analisis_csv:
        return jsonify({"error": "No hay análisis previo disponible para exportar."}), 404

    try:
        filas = []
        for p in ultimo_analisis_csv["pulsos"]:
            filas.append({
                "Pulso_ID": p["id"],
                "Tiempo_MM_SS": p["tiempo_str"],
                "Tiempo_Segundos": p["tiempo_seg"],
                "Muestra_Pico_R": p["muestra_r"],
                "Intervalo_RR_ms": p["rr_ms"],
                "Frecuencia_Cardiaca_BPM": p["bpm"],
                "Clase_Arritmia": p["clase"],
                "Diagnostico_Clinico": p["nombre_clinico"],
                "Confianza_Porcentaje": p["confianza"],
                "Severidad": p["severidad"]
            })

        df = pd.DataFrame(filas)
        buffer = io.StringIO()
        df.to_csv(buffer, index=False)
        buffer.seek(0)

        from flask import Response
        return Response(
            buffer.getvalue(),
            mimetype="text/csv",
            headers={"Content-Disposition": "attachment;filename=reporte_pulsos_ecg.csv"}
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/database/extract-csv", methods=["POST"])
def api_database_extract_csv():
    """
    Extrae un segmento puro (Voltaje vs Tiempo) de un registro MIT-BIH y lo guarda en CSV.
    Garantiza 100% ausencia de fuga de datos (sin etiquetas oficiales, marcas previas ni diagnósticos).
    """
    try:
        data = request.get_json(silent=True) or {}
        record_id = str(data.get("record", "100")).strip()
        duration_sec = float(data.get("duration_sec", 20))
        start_sec = float(data.get("start_sec", 0))
        channel = int(data.get("channel", 0))

        resultado = extraer_registro_a_csv(
            registro=record_id,
            duracion_seg=duration_sec,
            inicio_seg=start_sec,
            canal=channel,
            ruta_base="Data/Database"
        )
        return jsonify(resultado)
    except Exception as e:
        return jsonify({"error": f"Error al extraer segmento de base de datos: {str(e)}"}), 500

@app.route("/api/database/download-extracted", methods=["GET"])
def api_database_download_extracted():
    """Descarga un archivo CSV crudo extraído desde la base de datos."""
    try:
        file_path = request.args.get("file", "").strip()
        if not file_path:
            return jsonify({"error": "Parámetro 'file' no proporcionado."}), 400

        norm_path = Path(file_path).resolve()
        workspace_path = Path(".").resolve()
        if not str(norm_path).startswith(str(workspace_path)):
            return jsonify({"error": "Ruta no autorizada."}), 403

        if not norm_path.exists():
            return jsonify({"error": f"No se encontró el archivo '{file_path}'."}), 404

        with open(norm_path, 'r', encoding='utf-8') as f:
            contenido = f.read()

        from flask import Response
        return Response(
            contenido,
            mimetype="text/csv",
            headers={"Content-Disposition": f"attachment;filename={norm_path.name}"}
        )
    except Exception as e:
        return jsonify({"error": f"Error al descargar archivo extraído: {str(e)}"}), 500

@app.route("/api/evaluate/raw", methods=["POST"])
def api_evaluate_raw():
    """Ejecuta inferencia directa de extremo a extremo desde el registro crudo."""
    try:
        data = request.get_json(silent=True) or {}
        record_id = str(data.get("record", "100")).strip()
        model_version = str(data.get("model", "6.0")).strip()

        ruta_norm = os.path.join("Data", "Database", record_id)
        if not (os.path.exists(f"{ruta_norm}.hea") or os.path.exists(f"{ruta_norm}.dat")):
            return jsonify({"error": f"No se encontró el archivo del paciente '{record_id}' en Data/Database"}), 404

        record = wfdb.rdrecord(ruta_norm)
        annotation = wfdb.rdann(ruta_norm, 'atr')

        voltaje_ch0 = record.p_signal[:, 0]
        etiquetas_oficiales = list(zip(annotation.sample, annotation.symbol))

        # Preprocesar
        preprocesador = PreprocesadorECG(frecuencia_muestreo=record.fs, frecuencia_corte=40)
        resultado = preprocesador.pre_procesar(voltaje_ch0, etiquetas_bd=etiquetas_oficiales)

        X_test = resultado['X']
        Y_reales = resultado['Y']

        if len(X_test) == 0:
            return jsonify({"error": "No se detectaron latidos válidos en este registro."}), 400

        # Clasificador
        clasificador = ClasificadorArritmiasCNN(version_o_ruta=model_version)
        if clasificador.modelo is None:
            return jsonify({"error": f"No se pudo cargar el modelo '{model_version}'"}), 500

        predicciones_num, probas = clasificador.predecir(X_test)

        # Mapeo a 9 clases (índice 8 = No_Clasificable)
        mapeo_clases = {clase: idx for idx, clase in enumerate(CLASES_VIABLES)}
        Y_reales_num = np.array([mapeo_clases.get(str(e), 8) for e in Y_reales])

        num_salida = clasificador.modelo.output_shape[-1]
        etiquetas_eval = ETIQUETAS_FINALES_9X9 if num_salida >= 9 else CLASES_VIABLES
        num_labels = len(etiquetas_eval)

        if num_salida == 8:
            mascara_validos = Y_reales_num < 8
            Y_reales_num = Y_reales_num[mascara_validos]
            predicciones_num = predicciones_num[mascara_validos]

        matriz = confusion_matrix(Y_reales_num, predicciones_num, labels=np.arange(num_labels))
        accuracy = float(np.sum(Y_reales_num == predicciones_num) / len(Y_reales_num)) if len(Y_reales_num) > 0 else 0

        reporte_dict = classification_report(
            Y_reales_num, predicciones_num,
            target_names=etiquetas_eval,
            labels=np.arange(num_labels),
            output_dict=True,
            zero_division=0
        )

        # Muestra de la señal cruda ECG (primeros 1200 puntos) para graficar en canvas
        preview_ecg = [float(val) for val in voltaje_ch0[:1200]]

        return jsonify({
            "record": record_id,
            "total_beats": int(len(X_test)),
            "fs": int(record.fs),
            "accuracy": round(accuracy * 100, 2),
            "labels": etiquetas_eval,
            "matrix": matriz.tolist(),
            "report": reporte_dict,
            "ecg_preview": preview_ecg
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Error procesando paciente: {str(e)}"}), 500

@app.route("/api/evaluate/npz", methods=["POST"])
def api_evaluate_npz():
    """Evalúa un archivo .npz preprocesado."""
    try:
        data = request.get_json(silent=True) or {}
        file_path = str(data.get("file", "Data/Raw_data/100.npz")).strip()
        model_version = str(data.get("model", "6.0")).strip()

        if not os.path.exists(file_path):
            return jsonify({"error": f"No se encontró el archivo '{file_path}'"}), 404

        datos = np.load(file_path, allow_pickle=True)
        X_test = datos['X']
        Y_reales = datos['Y']

        clasificador = ClasificadorArritmiasCNN(version_o_ruta=model_version)
        if clasificador.modelo is None:
            return jsonify({"error": f"No se pudo cargar el modelo '{model_version}'"}), 500

        predicciones_num, _ = clasificador.predecir(X_test)

        mapeo_clases = {clase: idx for idx, clase in enumerate(CLASES_VIABLES)}
        Y_reales_num = np.array([mapeo_clases.get(str(e), 8) for e in Y_reales])

        num_salida = clasificador.modelo.output_shape[-1]
        etiquetas_eval = ETIQUETAS_FINALES_9X9 if num_salida >= 9 else CLASES_VIABLES
        num_labels = len(etiquetas_eval)

        if num_salida == 8:
            mascara_validos = Y_reales_num < 8
            Y_reales_num = Y_reales_num[mascara_validos]
            predicciones_num = predicciones_num[mascara_validos]

        matriz = confusion_matrix(Y_reales_num, predicciones_num, labels=np.arange(num_labels))
        accuracy = float(np.sum(Y_reales_num == predicciones_num) / len(Y_reales_num)) if len(Y_reales_num) > 0 else 0

        reporte_dict = classification_report(
            Y_reales_num, predicciones_num,
            target_names=etiquetas_eval,
            labels=np.arange(num_labels),
            output_dict=True,
            zero_division=0
        )

        return jsonify({
            "file": Path(file_path).name,
            "total_beats": int(len(X_test)),
            "accuracy": round(accuracy * 100, 2),
            "labels": etiquetas_eval,
            "matrix": matriz.tolist(),
            "report": reporte_dict
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Error evaluando .npz: {str(e)}"}), 500

@app.route("/api/vt-search", methods=["GET", "POST"])
def api_vt_search():
    """Busca eventos de Taquicardia Ventricular en la base de datos."""
    try:
        hallazgos = buscar_taquicardia_ventricular("Data/Database")
        return jsonify({"total": len(hallazgos), "events": hallazgos})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/inspect", methods=["POST"])
def api_inspect():
    """Inspecciona un archivo .npz o compara con anotaciones oficiales."""
    data = request.get_json() or {}
    record_id = data.get("record", "100").strip()

    ruta_npz = os.path.join("Data", "Raw_data", f"{record_id}.npz")
    ruta_atr = os.path.join("Data", "Database", record_id)

    info_npz = {}
    dict_alg = {}
    dict_db = {}

    if os.path.exists(ruta_npz):
        try:
            datos = np.load(ruta_npz, allow_pickle=True)
            X = datos['X']
            Y = datos['Y']
            unicas_alg, conteos_alg = np.unique(Y, return_counts=True)
            dict_alg = dict(zip(unicas_alg, [int(c) for c in conteos_alg]))
            info_npz = {
                "shape_X": list(X.shape),
                "dtype_X": str(X.dtype),
                "min_X": float(np.min(X)),
                "max_X": float(np.max(X)),
                "total_latidos": len(Y)
            }
        except Exception as e:
            info_npz["error"] = str(e)

    if os.path.exists(f"{ruta_atr}.atr"):
        try:
            ann = wfdb.rdann(ruta_atr, 'atr')
            unicas_db, conteos_db = np.unique(ann.symbol, return_counts=True)
            dict_db = dict(zip(unicas_db, [int(c) for c in conteos_db]))
        except Exception as e:
            dict_db["error"] = str(e)

    todas_etiquetas = sorted(list(set(list(dict_alg.keys()) + list(dict_db.keys()))))
    comparativa = []
    for e in todas_etiquetas:
        comparativa.append({
            "label": e,
            "alg": dict_alg.get(e, 0),
            "db": dict_db.get(e, 0)
        })

    return jsonify({
        "record": record_id,
        "info_npz": info_npz,
        "comparativa": comparativa
    })

@app.route("/api/preprocess", methods=["POST"])
def api_preprocess():
    """Ejecuta preprocesamiento de un registro o por lotes."""
    try:
        data = request.get_json(silent=True) or {}
        record_id = str(data.get("record", "100")).strip()

        if record_id.lower() == "all":
            count = procesar_base_datos()
            return jsonify({"message": f"Base de datos procesada con éxito ({count} registros generados)."})
        else:
            ruta = os.path.join("Data", "Database", record_id)
            if not (os.path.exists(f"{ruta}.hea") or os.path.exists(f"{ruta}.dat")):
                return jsonify({"error": f"No se encontró el registro '{record_id}' en Data/Database."}), 404

            ok = procesar_registro_ecg(ruta)
            if ok:
                return jsonify({"message": f"Registro '{record_id}' procesado y guardado en Data/Raw_data."})
            else:
                return jsonify({"error": f"Fallo al procesar registro '{record_id}'."}), 400
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Error en preprocesamiento: {str(e)}"}), 500

@app.route("/api/dataset/unify", methods=["POST"])
def api_unify():
    """Unifica todos los .npz individuales."""
    try:
        salida = unificar_datasets()
        if salida:
            return jsonify({"message": f"Dataset unificado creado con éxito en '{salida}'."})
        else:
            return jsonify({"error": "No se encontraron tensores para unificar en Data/Raw_data."}), 400
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Error al unificar: {str(e)}"}), 500

@app.route("/api/dataset/filter", methods=["POST"])
def api_filter():
    """Filtra y limpia el dataset unificado."""
    try:
        salida = filtrar_dataset()
        if salida:
            return jsonify({"message": f"Dataset limpio generado con éxito en '{salida}'."})
        else:
            return jsonify({"error": "Fallo al filtrar el dataset."}), 400
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Error al filtrar: {str(e)}"}), 500

# Estado global para experimentos de entrenamiento
training_state = {
    "status": "idle",
    "message": "Listo para iniciar entrenamiento experimental.",
    "epochs": 0,
    "last_result": None
}

@app.route("/api/train", methods=["POST"])
def api_train():
    """Inicia entrenamiento experimental de la red neuronal en segundo plano."""
    if training_state["status"] == "running":
        return jsonify({
            "status": "running",
            "message": "Ya existe un experimento de entrenamiento en ejecución."
        }), 400

    data = request.get_json(silent=True) or {}
    epochs = int(data.get("epochs", 15))
    batch_size = int(data.get("batch_size", 256))
    lr = float(data.get("learning_rate", 1e-3))

    ruta_limpio = "Data/Raw_data/dataset_limpio_y_unificado.npz"
    if not os.path.exists(ruta_limpio):
        return jsonify({
            "error": "No se encontró el dataset limpio en Data/Raw_data/dataset_limpio_y_unificado.npz. Por favor ejecuta primero la unificación y filtrado AAMI."
        }), 400

    def _ejecutar_entrenamiento():
        training_state["status"] = "running"
        training_state["message"] = f"Entrenando CNN ({epochs} épocas, Batch: {batch_size}, LR: {lr})..."
        training_state["epochs"] = epochs
        try:
            modelo, hist = entrenar_modelo(
                ruta_dataset=ruta_limpio,
                ruta_salida_modelo="Data/Models/modelo_arritmias_cnn_experimental.keras",
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=lr
            )
            acc_val = float(hist.history.get("val_accuracy", [0])[-1])
            training_state["status"] = "completed"
            training_state["message"] = f"Entrenamiento completado exitosamente. Val Accuracy: {acc_val * 100:.2f}%. Guardado en Data/Models/modelo_arritmias_cnn_experimental.keras"
            training_state["last_result"] = {
                "val_accuracy": round(acc_val * 100, 2),
                "model_file": "modelo_arritmias_cnn_experimental.keras"
            }
        except Exception as e:
            import traceback
            traceback.print_exc()
            training_state["status"] = "error"
            training_state["message"] = f"Error en entrenamiento: {str(e)}"

    threading.Thread(target=_ejecutar_entrenamiento, daemon=True).start()
    return jsonify({
        "status": "running",
        "message": "Entrenamiento iniciado en segundo plano. Monitorea el progreso en esta pestaña."
    })

@app.route("/api/train/status", methods=["GET"])
def api_train_status():
    """Retorna el estado actual del entrenamiento."""
    return jsonify(training_state)

def iniciar_servidor_web(host="127.0.0.1", port=5000, abrir_navegador=True, debug=False):
    """Lanza el servidor Flask y abre el navegador automáticamente."""
    url = f"http://{host}:{port}"
    print(f"\n=======================================================")
    print(f"[*] Iniciando servidor web de la aplicacion...")
    print(f"[>] URL de acceso: {url}")
    print(f"=======================================================\n")

    if abrir_navegador:
        def _abrir():
            import time
            time.sleep(1.2)
            try:
                webbrowser.open_new_tab(url)
            except Exception as e:
                print(f"No se pudo abrir el navegador automáticamente: {e}")

        threading.Thread(target=_abrir, daemon=True).start()

    app.run(host=host, port=port, debug=debug)

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
=============================================================================
CARDIOLAB RESEARCH - ARCHIVO MAESTRO DE TERMINAL (CLI)
=============================================================================
Punto de control maestro por consola para ejecutar y automatizar el workflow:
1. Preprocesamiento de registros crudos (ECG -> Picos R -> STFT -> .npz)
2. Curaduría y unificación de datasets (Filtro AAMI y balanceo)
3. Entrenamiento de la Red Neuronal Convolucional (CNN 2D)
4. Evaluación clínica end-to-end de un paciente crudo (.dat/.hea/.atr)
5. Evaluación sobre datasets preprocesados (Matriz 9x9 con opción a rechazo)
6. Diagnóstico de patologías específicas (Taquicardia Ventricular)
7. Inspección y diagnóstico de archivos de datos (.npz vs .atr)
=============================================================================
"""

import sys
import os
import argparse
from pathlib import Path

if sys.platform == "win32":
    try:
        if hasattr(sys.stdout, 'reconfigure'):
            sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        if hasattr(sys.stderr, 'reconfigure'):
            sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

# Añadir directorio raíz al PYTHONPATH
directorio_raiz = Path(__file__).resolve().parent
if str(directorio_raiz) not in sys.path:
    sys.path.insert(0, str(directorio_raiz))

from src.preprocessing.batch_processor import procesar_base_datos, procesar_registro_ecg
from src.dataset.dataset_manager import unificar_datasets, filtrar_dataset
from src.dataset.inspector import inspeccionar_npz, comparar_registro_con_atr
from src.models.trainer import entrenar_modelo
from src.evaluation.evaluator import evaluar_modelo_desde_npz, evaluar_paciente_crudo
from src.evaluation.clinical_metrics import buscar_taquicardia_ventricular
from src.preprocessing.csv_analyzer import cargar_senal_csv, analizar_senal_ecg, extraer_registro_a_csv
import pandas as pd

BANNER = r"""
==============================================================================
   [CardioAI] - CLASIFICADOR DE ARRITMIAS CARDIACAS - MAESTRO DE WORKFLOW (CLI)
==============================================================================
"""

def menu_interactivo():
    while True:
        print(BANNER)
        print("Selecciona una fase del workflow para ejecutar:\n")
        print("  [1] Preprocesar registros crudos (Data/Database -> Data/Raw_data/*.npz)")
        print("  [2] Unificar y filtrar datasets (dataset_unificado -> dataset_limpio_y_unificado)")
        print("  [3] Entrenar modelo CNN (Balanceo de clases + Checkpoint)")
        print("  [4] Evaluar paciente crudo (End-to-End: .dat/.hea/.atr -> Diagnóstico)")
        print("  [5] Evaluar modelo en archivo .npz (Matriz de confusión 9x9 con rechazo)")
        print("  [6] Detectar episodios de Taquicardia Ventricular (VT) en la base de datos")
        print("  [7] Inspeccionar archivo .npz o comparar con anotaciones oficiales (.atr)")
        print("  [8] Analizar archivo CSV externo (Voltaje vs Tiempo) y clasificar pulso por pulso")
        print("  [9] Extraer segmento de base de datos MIT-BIH a CSV crudo (Cero fuga de datos)")
        print("  [0] Salir")
        print("-" * 78)

        opcion = input("Ingresa tu opción (0-9): ").strip()

        if opcion == "1":
            print("\n--- [1] PREPROCESAMIENTO POR LOTES ---")
            sub_op = input("¿Deseas procesar un único registro (u) o toda la base de datos (t)? [u/t]: ").strip().lower()
            if sub_op == "u":
                rec = input("Ingresa el número de registro (ej. 100): ").strip()
                ruta = os.path.join("Data", "Database", rec)
                procesar_registro_ecg(ruta)
            else:
                procesar_base_datos()
            input("\nPresiona Enter para continuar...")

        elif opcion == "2":
            print("\n--- [2] UNIFICACIÓN Y FILTRADO DE DATASET ---")
            unificar = input("¿Deseas unificar los archivos .npz individuales? [s/n]: ").strip().lower()
            if unificar == 's':
                unificar_datasets()
            filtrar = input("¿Deseas filtrar ruido y generar dataset_limpio_y_unificado.npz? [s/n]: ").strip().lower()
            if filtrar == 's':
                filtrar_dataset()
            input("\nPresiona Enter para continuar...")

        elif opcion == "3":
            print("\n--- [3] ENTRENAMIENTO DE LA CNN ---")
            epochs_str = input("Número de épocas (por defecto 50): ").strip()
            epochs = int(epochs_str) if epochs_str.isdigit() else 50
            entrenar_modelo(epochs=epochs)
            input("\nPresiona Enter para continuar...")

        elif opcion == "4":
            print("\n--- [4] EVALUACIÓN END-TO-END DE PACIENTE CRUDO ---")
            rec = input("Ingresa el número de registro o ruta (por defecto '100'): ").strip()
            if not rec:
                rec = "100"
            ruta = os.path.join("Data", "Database", rec) if not os.path.exists(rec) else rec
            ver = input("Versión del modelo (por defecto '6.0'): ").strip()
            version = ver if ver else "6.0"
            evaluar_paciente_crudo(ruta, version_modelo=version)
            input("\nPresiona Enter para continuar...")

        elif opcion == "5":
            print("\n--- [5] EVALUACIÓN DE MODELO DESDE ARCHIVO .NPZ ---")
            archivo = input("Ruta al archivo .npz (por defecto 'Data/Raw_data/100.npz'): ").strip()
            if not archivo:
                archivo = os.path.join("Data", "Raw_data", "100.npz")
            ver = input("Versión del modelo (por defecto '6.0'): ").strip()
            version = ver if ver else "6.0"
            evaluar_modelo_desde_npz(archivo, version_modelo=version)
            input("\nPresiona Enter para continuar...")

        elif opcion == "6":
            print("\n--- [6] BÚSQUEDA DE TAQUICARDIA VENTRICULAR ---")
            buscar_taquicardia_ventricular()
            input("\nPresiona Enter para continuar...")

        elif opcion == "7":
            print("\n--- [7] INSPECCIÓN DE DATOS ---")
            sub_op = input("¿Inspeccionar estructura de .npz (i) o comparar con .atr (c)? [i/c]: ").strip().lower()
            if sub_op == "c":
                rec = input("Número de registro a comparar (ej. 103): ").strip()
                if not rec:
                    rec = "103"
                comparar_registro_con_atr(rec)
            else:
                archivo = input("Ruta al archivo .npz a inspeccionar (ej. 'Data/Raw_data/100.npz'): ").strip()
                if not archivo:
                    archivo = os.path.join("Data", "Raw_data", "100.npz")
                inspeccionar_npz(archivo)
            input("\nPresiona Enter para continuar...")

        elif opcion == "8":
            print("\n--- [8] ANALIZAR ARCHIVO CSV (VOLTAJE VS TIEMPO) ---")
            ruta_csv = input("Ruta al archivo CSV [default: Data/ejemplo_ecg_voltaje_tiempo.csv]: ").strip()
            if not ruta_csv:
                ruta_csv = "Data/ejemplo_ecg_voltaje_tiempo.csv"

            if not os.path.exists(ruta_csv):
                print(f"Error: No se encontró el archivo en '{ruta_csv}'.")
                input("\nPresiona Enter para continuar...")
                continue

            fs_input = input("Frecuencia de muestreo (Hz) [default: auto-detectar o 360]: ").strip()
            fs = float(fs_input) if fs_input else 360.0

            modelo_sel = input("Versión del modelo CNN [default: 6.0]: ").strip() or "6.0"

            print(f"\nLeyendo archivo '{ruta_csv}'...")
            try:
                voltaje, fs_est, tiempo = cargar_senal_csv(ruta_csv, fs_por_defecto=fs)
                print(f"Señal cargada: {len(voltaje)} muestras | Fs: {fs_est} Hz | Duración: {len(voltaje)/fs_est:.2f} s")
                print("Ejecutando filtrado Butterworth, detección R y clasificación CNN...")

                resultado = analizar_senal_ecg(voltaje, fs=fs_est, modelo=modelo_sel)
                if "error" in resultado:
                    print(f"Error en análisis: {resultado['error']}")
                else:
                    resumen = resultado["resumen"]
                    print("\n" + "=" * 80)
                    print(f"  RESUMEN DEL ANÁLISIS CLÍNICO: {len(resultado['pulsos'])} PULSOS DETECTADOS")
                    print("=" * 80)
                    print(f"  - Frecuencia Cardíaca Media: {resumen['bpm_medio']} BPM (Min: {resumen['bpm_min']} | Max: {resumen['bpm_max']})")
                    print(f"  - Latidos Normales (N):      {resumen['normales']}")
                    print(f"  - Latidos Anormales:         {resumen['anormales']} ({resumen['tasa_arritmias_pct']}%)")
                    print(f"  - Distribución por Clase:    {resumen['distribucion_clases']}")
                    if resumen['alertas_taquicardia']:
                        print(f"  - ¡ALERTA! Se detectaron {len(resumen['alertas_taquicardia'])} episodios de ráfagas ventriculares (VT).")
                    print("-" * 80)
                    print(f"{'#':<4} | {'Tiempo':<10} | {'Muestra':<9} | {'BPM':<6} | {'Clase':<5} | {'Diagnóstico Clínico':<32} | {'Confianza':<9} | {'Severidad'}")
                    print("-" * 80)
                    for p in resultado["pulsos"]:
                        print(f"{p['id']:<4} | {p['tiempo_str']:<10} | {p['muestra_r']:<9} | {p['bpm']:<6.1f} | {p['clase']:<5} | {p['nombre_clinico'][:32]:<32} | {p['confianza']:>7.1f}% | {p['severidad']}")
                    print("=" * 80)

                    guardar = input("\n¿Deseas exportar este reporte a un archivo CSV? [s/n]: ").strip().lower()
                    if guardar == 's':
                        salida = input("Nombre del archivo de salida [default: reporte_pulsos.csv]: ").strip() or "reporte_pulsos.csv"
                        df_exp = pd.DataFrame([{
                            "Pulso_ID": p["id"],
                            "Tiempo_MM_SS": p["tiempo_str"],
                            "Tiempo_Segundos": p["tiempo_seg"],
                            "Muestra_Pico_R": p["muestra_r"],
                            "Intervalo_RR_ms": p["rr_ms"],
                            "BPM": p["bpm"],
                            "Clase": p["clase"],
                            "Diagnostico": p["nombre_clinico"],
                            "Confianza": p["confianza"],
                            "Severidad": p["severidad"]
                        } for p in resultado["pulsos"]])
                        df_exp.to_csv(salida, index=False)
                        print(f"Reporte exportado con éxito a '{salida}'.")
            except Exception as e:
                print(f"Error procesando CSV: {e}")

            input("\nPresiona Enter para continuar...")

        elif opcion == "9":
            print("\n--- [9] EXTRAER SEGMENTO A CSV CRUDO (CERO FUGA DE DATOS) ---")
            rec = input("Número de registro MIT-BIH [default: 100]: ").strip() or "100"
            dur_in = input("Duración en segundos a extraer [default: 20]: ").strip()
            dur = float(dur_in) if dur_in else 20.0
            start_in = input("Segundo de inicio / offset [default: 0]: ").strip()
            start = float(start_in) if start_in else 0.0

            salida = input("Ruta destino del CSV [default: Data/Extraidos/ecg_registro_<id>_<dur>s.csv]: ").strip() or None

            try:
                res = extraer_registro_a_csv(rec, duracion_seg=dur, inicio_seg=start, ruta_salida=salida)
                print("\n" + "=" * 78)
                print(f"  EXTRACCIÓN EXITOSA (GARANTÍA ANTI-FUGA DE DATOS)")
                print("=" * 78)
                print(f"  - Registro fuente:   {res['registro']} (Data/Database/{res['registro']}.dat/.hea)")
                print(f"  - Ventana temporal:  {res['inicio_seg']}s a {res['inicio_seg'] + res['duracion_seg']}s ({res['duracion_seg']}s totales)")
                print(f"  - Muestras extraídas:{res['puntos']} puntos @ {res['fs']} Hz")
                print(f"  - Columnas físicas:  tiempo, voltaje (exclusivamente)")
                print(f"  - Archivo guardado:  {res['ruta_archivo']}")
                print("=" * 78)
                print("-> Ahora puedes analizar este archivo con la opción [8] o desde la aplicación web.")
            except Exception as e:
                print(f"Error al extraer registro: {e}")

            input("\nPresiona Enter para continuar...")

        elif opcion == "0":
            print("\n¡Hasta luego!")
            break
        else:
            print("\nOpción no válida. Intenta de nuevo.")

def parsear_argumentos():
    parser = argparse.ArgumentParser(
        description="Clasificador de Arritmias Cardíacas - Orquestador Maestro de Workflow",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--mode",
        choices=["interactive", "preprocess", "unify", "filter", "train", "eval-raw", "eval-npz", "vt-search", "inspect", "predict-csv", "extract-csv"],
        default="interactive",
        help="Modo de ejecución del pipeline"
    )
    parser.add_argument("--record", type=str, default="100", help="Número de registro de paciente (ej. 100)")
    parser.add_argument("--file", type=str, default=None, help="Ruta a archivo .npz o .csv para evaluar o clasificar")
    parser.add_argument("--duration", type=float, default=20.0, help="Duración en segundos a extraer (para extract-csv)")
    parser.add_argument("--start", type=float, default=0.0, help="Segundo de inicio / offset (para extract-csv)")
    parser.add_argument("--fs", type=float, default=360.0, help="Frecuencia de muestreo en Hz (para modo predict-csv)")
    parser.add_argument("--output", type=str, default=None, help="Ruta para exportar reporte CSV o destino de extracción")
    parser.add_argument("--model", type=str, default="6.0", help="Versión del modelo o ruta .keras (ej. 6.0)")
    parser.add_argument("--epochs", type=int, default=50, help="Número de épocas de entrenamiento")
    parser.add_argument("--batch-size", type=int, default=256, help="Tamaño de lote para entrenamiento")
    parser.add_argument("--no-plot", action="store_true", help="Desactivar visualización gráfica de la matriz")
    parser.add_argument("--save-plot", type=str, default=None, help="Ruta para guardar imagen de la matriz de confusión")
    return parser.parse_args()

def main():
    args = parsear_argumentos()

    if args.mode == "interactive" and len(sys.argv) == 1:
        menu_interactivo()
        return

    mostrar_grafica = not args.no_plot

    if args.mode == "predict-csv":
        archivo_csv = args.file if args.file else "Data/ejemplo_ecg_voltaje_tiempo.csv"
        if not os.path.exists(archivo_csv):
            print(f"Error: No se encontró el archivo '{archivo_csv}'.")
            return

        print(f"Leyendo '{archivo_csv}'...")
        voltaje, fs_est, tiempo = cargar_senal_csv(archivo_csv, fs_por_defecto=args.fs)
        print(f"Señal cargada: {len(voltaje)} puntos a {fs_est} Hz ({len(voltaje)/fs_est:.2f} s). Procesando...")
        resultado = analizar_senal_ecg(voltaje, fs=fs_est, modelo=args.model)

        if "error" in resultado:
            print(f"Error: {resultado['error']}")
            return

        resumen = resultado["resumen"]
        print("\n" + "=" * 80)
        print(f"  CLASIFICACIÓN PULSO POR PULSO: {len(resultado['pulsos'])} LATIDOS ENCONTRADOS")
        print("=" * 80)
        print(f"  - BPM Medio: {resumen['bpm_medio']} (Rango: {resumen['bpm_min']} - {resumen['bpm_max']})")
        print(f"  - Normales: {resumen['normales']} | Arritmias: {resumen['anormales']} ({resumen['tasa_arritmias_pct']}%)")
        print(f"  - Distribución: {resumen['distribucion_clases']}")
        print("-" * 80)
        print(f"{'#':<4} | {'Tiempo':<10} | {'Muestra':<9} | {'BPM':<6} | {'Clase':<5} | {'Diagnóstico':<28} | {'Confianza':<9} | {'Severidad'}")
        print("-" * 80)
        for p in resultado["pulsos"]:
            print(f"{p['id']:<4} | {p['tiempo_str']:<10} | {p['muestra_r']:<9} | {p['bpm']:<6.1f} | {p['clase']:<5} | {p['nombre_clinico'][:28]:<28} | {p['confianza']:>7.1f}% | {p['severidad']}")
        print("=" * 80)

        if args.output:
            df_out = pd.DataFrame([{
                "Pulso_ID": p["id"],
                "Tiempo_MM_SS": p["tiempo_str"],
                "Muestra_Pico_R": p["muestra_r"],
                "BPM": p["bpm"],
                "Clase": p["clase"],
                "Diagnostico": p["nombre_clinico"],
                "Confianza": p["confianza"],
                "Severidad": p["severidad"]
            } for p in resultado["pulsos"]])
            df_out.to_csv(args.output, index=False)
            print(f"Reporte exportado exitosamente a '{args.output}'.")

    elif args.mode == "preprocess":
        if args.record and args.record != "all":
            ruta = os.path.join("Data", "Database", args.record)
            procesar_registro_ecg(ruta)
        else:
            procesar_base_datos()

    elif args.mode == "unify":
        unificar_datasets()

    elif args.mode == "filter":
        filtrar_dataset()

    elif args.mode == "train":
        entrenar_modelo(epochs=args.epochs, batch_size=args.batch_size)

    elif args.mode == "eval-raw":
        ruta = os.path.join("Data", "Database", args.record) if not os.path.exists(args.record) else args.record
        evaluar_paciente_crudo(
            ruta,
            version_modelo=args.model,
            mostrar_grafica=mostrar_grafica,
            ruta_guardar_grafica=args.save_plot
        )

    elif args.mode == "eval-npz":
        archivo = args.file if args.file else os.path.join("Data", "Raw_data", f"{args.record}.npz")
        evaluar_modelo_desde_npz(
            archivo,
            version_modelo=args.model,
            mostrar_grafica=mostrar_grafica,
            ruta_guardar_grafica=args.save_plot
        )

    elif args.mode == "vt-search":
        buscar_taquicardia_ventricular()

    elif args.mode == "inspect":
        if args.file:
            inspeccionar_npz(args.file)
        else:
            comparar_registro_con_atr(args.record)

    elif args.mode == "extract-csv":
        try:
            res = extraer_registro_a_csv(
                registro=args.record,
                duracion_seg=args.duration,
                inicio_seg=args.start,
                ruta_salida=args.output
            )
            print("\n" + "=" * 80)
            print("  EXTRACCIÓN DE SEÑAL PURA A CSV (CERO FUGA DE DATOS)")
            print("=" * 80)
            print(f"  - Registro fuente:   {res['registro']} (Data/Database/{res['registro']})")
            print(f"  - Ventana temporal:  {res['inicio_seg']}s a {res['inicio_seg'] + res['duracion_seg']}s ({res['duracion_seg']}s)")
            print(f"  - Puntos extraídos:  {res['puntos']} muestras @ {res['fs']} Hz")
            print(f"  - Columnas:          tiempo, voltaje (sin anotaciones ni marcas previas)")
            print(f"  - Archivo guardado:  {res['ruta_archivo']}")
            print("=" * 80)
            print(f"Para clasificar este archivo pulso a pulso, ejecuta:")
            print(f"  python maestro.py --mode predict-csv --file \"{res['ruta_archivo']}\"")
        except Exception as e:
            print(f"Error al extraer registro a CSV: {e}")

if __name__ == "__main__":
    main()

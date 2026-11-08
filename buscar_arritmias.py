#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
=============================================================================
CARDIOLAB - EXPLORADOR DE REGISTROS CLÍNICOS Y CATÁLOGO DE ARRITMIAS (MIT-BIH)
=============================================================================
Script rápido e independiente para consultar qué registros de la base de datos
MIT-BIH contienen arritmias específicas, su frecuencia de aparición y el segundo
exacto en el que aparecen por primera vez.

Uso rápido:
  python buscar_arritmias.py                    # Catálogo curado de registros clave
  python buscar_arritmias.py --tipo V           # Solo contracciones ventriculares (PVCs)
  python buscar_arritmias.py --tipo L           # Bloqueo de rama izquierda (LBBB)
  python buscar_arritmias.py --registro 208     # Desglose clínico completo del registro 208
=============================================================================
"""

import os
import sys
import glob
import argparse
from collections import Counter
from pathlib import Path

# Ajustar codificación para consola Windows
if sys.platform == "win32":
    try:
        if hasattr(sys.stdout, 'reconfigure'):
            sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        if hasattr(sys.stderr, 'reconfigure'):
            sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

try:
    import wfdb
except ImportError:
    print("Error: Se requiere 'wfdb'. Instálalo con: pip install wfdb")
    sys.exit(1)

# Metadatos clínicos según el estándar AAMI EC57
NOMBRES_ARRITMIAS = {
    'V': ('Contracción Ventricular Prematura (PVC)', 'Alta', 'Ventrículos'),
    'A': ('Contracción Auricular Prematura (PAC)', 'Moderada', 'Aurículas'),
    'L': ('Bloqueo de Rama Izquierda (LBBB)', 'Moderada/Alta', 'Conducción Intraventricular'),
    'R': ('Bloqueo de Rama Derecha (RBBB)', 'Moderada', 'Conducción Intraventricular'),
    '/': ('Latido de Marcapasos Artificial (Paced)', 'Especial', 'Estimulación Eléctrica'),
    'F': ('Fusión Ventricular y Normal', 'Moderada', 'Ventrículos / Sinusal'),
    'f': ('Fusión de Marcapasos y Normal', 'Especial', 'Marcapasos / Sinusal'),
    'E': ('Latido de Escape Ventricular', 'Alta', 'Ventrículos'),
    'j': ('Latido de Escape Nodal / Unión', 'Moderada', 'Nodo AV'),
    'J': ('Contracción Prematura Nodal', 'Baja', 'Nodo AV'),
    'N': ('Latido Normal (Sinus Beat)', 'Normal', 'Nodo Sinusal')
}

NOTAS_CLINICAS = {
    '208': 'Bigeminismo y pares ventriculares frecuentes, alternando con latidos de fusión (F).',
    '233': 'Salvas rápidas de taquicardia ventricular no sostenida (VT) y PVCs polimórficos.',
    '200': 'Arritmias ventriculares complejas, bigeminismo y episodios de taquicardia ventricular.',
    '106': 'Episodios clásicos de taquicardia ventricular paroxística a partir del minuto 1:30.',
    '207': 'Flutter Ventricular extremo, arritmias ventriculares polimórficas y bloqueos de conducción.',
    '214': 'Bloqueo de rama izquierda (LBBB) con frecuentes contracciones ventriculares prematuras.',
    '109': 'Morfología pura de bloqueo de rama izquierda (LBBB) en más del 98% del trazado.',
    '111': 'Bloqueo de rama izquierda (LBBB) persistente de alta estabilidad morfológica.',
    '118': 'Bloqueo de rama derecha (RBBB) clásico con morfología rSR\' casi omnipresente.',
    '124': 'Bloqueo de rama derecha (RBBB) puro y constante.',
    '212': 'Bloqueo de rama derecha (RBBB) con cambios de eje eléctrico.',
    '232': 'Alta densidad de contracciones auriculares prematuras (PACs) y ritmo ectópico.',
    '209': 'Contracciones auriculares prematuras (PACs) frecuentes y episodios de taquicardia auricular.',
    '107': 'Ritmo de marcapasos unipolar puro (espigas prominentes y complejos anchos).',
    '102': 'Ritmo de marcapasos artificial con abundantes latidos de fusión (f).',
    '217': 'Marcapasos a demanda alternando con complejos sinusales conducidos.',
    '213': 'Excelente registro para observar latidos de fusión ventricular (F).'
}

def obtener_registros_disponibles(ruta_db="Data/Database"):
    """Lista todos los registros válidos de la base de datos MIT-BIH."""
    if not os.path.exists(ruta_db):
        return []
    archivos_atr = glob.glob(os.path.join(ruta_db, "*.atr"))
    registros = []
    for f in archivos_atr:
        nombre = os.path.splitext(os.path.basename(f))[0]
        if not nombre.endswith("-0"):  # Ignorar variantes secundarias
            registros.append(nombre)
    return sorted(list(set(registros)))

def analizar_todos_los_registros(ruta_db="Data/Database"):
    """Escanea los archivos de anotaciones oficiales y extrae estadísticas de arritmias."""
    registros = obtener_registros_disponibles(ruta_db)
    if not registros:
        print(f"No se encontraron registros en '{ruta_db}'.")
        return {}

    catalogo = {}
    for r in registros:
        ruta_base = os.path.join(ruta_db, r)
        try:
            ann = wfdb.rdann(ruta_base, 'atr')
        except Exception:
            continue

        fs = 360.0
        total_latidos = len(ann.symbol)
        conteo = Counter(ann.symbol)

        # Ubicación del primer latido de cada tipo
        primeros_seg = {}
        for sample, sym in zip(ann.sample, ann.symbol):
            if sym not in primeros_seg:
                primeros_seg[sym] = round(sample / fs, 2)

        catalogo[r] = {
            "total": total_latidos,
            "conteo": conteo,
            "primeros_seg": primeros_seg
        }

    return catalogo

def mostrar_desglose_registro(registro_id, datos):
    """Muestra el perfil completo de un paciente específico."""
    total = datos["total"]
    conteo = datos["conteo"]
    primeros = datos["primeros_seg"]

    print("=" * 80)
    print(f"  PERFIL CLÍNICO DEL REGISTRO MIT-BIH: {registro_id}")
    print("=" * 80)
    if registro_id in NOTAS_CLINICAS:
        print(f"  Observación clínica: {NOTAS_CLINICAS[registro_id]}")
        print("-" * 80)

    print(f"  Total de latidos anotados: {total}")
    print("-" * 80)
    print(f"  {'Símbolo':<8} | {'Cantidad':<10} | {'Porcentaje':<12} | {'Primer Latido':<15} | {'Nombre Clínico'}")
    print("-" * 80)

    for sym, cnt in conteo.most_common():
        pct = (cnt / total) * 100
        t_primero = f"{primeros.get(sym, 0.0)} s"
        nombre_clinico = NOMBRES_ARRITMIAS.get(sym, (f"Anotación '{sym}'", '', ''))[0]
        print(f"  {sym:<8} | {cnt:<10} | {pct:>8.2f} %   | {t_primero:<15} | {nombre_clinico}")

    print("=" * 80)
    print(f"  Comando sugerido para extraer y clasificar los primeros 30s de este paciente:")
    print(f"    python maestro.py --mode extract-csv --record {registro_id} --duration 30 --start 0")
    print(f"    python maestro.py --mode predict-csv --file \"Data/Extraidos/ecg_registro_{registro_id}_30s.csv\"")
    print("=" * 80)

def mostrar_top_por_arritmia(catalogo, tipo_arritmia, top_n=5):
    """Muestra los mejores registros para una arritmia en particular."""
    info_arr = NOMBRES_ARRITMIAS.get(tipo_arritmia, (f"Arritmia '{tipo_arritmia}'", "Variable", "General"))
    candidatos = []

    for reg, d in catalogo.items():
        cnt = d["conteo"].get(tipo_arritmia, 0)
        if cnt > 0:
            pct = (cnt / d["total"]) * 100
            t_primero = d["primeros_seg"].get(tipo_arritmia, 0.0)
            candidatos.append((reg, cnt, pct, t_primero, d["total"]))

    candidatos.sort(key=lambda x: x[1], reverse=True)

    print("\n" + "=" * 85)
    print(f"  ARRITMIA: [{tipo_arritmia}] {info_arr[0].upper()}")
    print(f"  Severidad: {info_arr[1]} | Origen: {info_arr[2]}")
    print("=" * 85)

    if not candidatos:
        print(f"  No se encontraron registros con la arritmia '{tipo_arritmia}'.")
        return

    print(f"  {'#':<3} | {'Registro':<9} | {'Latidos':<9} | {'% Total':<10} | {'1er Latido en':<14} | {'Recomendación / Nota Clínica'}")
    print("-" * 85)

    for i, (reg, cnt, pct, t_primero, tot) in enumerate(candidatos[:top_n], 1):
        nota = NOTAS_CLINICAS.get(reg, "Presencia significativa de esta arritmia.")
        if len(nota) > 36:
            nota = nota[:33] + "..."
        print(f"  {i:<3} | {reg:<9} | {cnt:<9} | {pct:>6.2f} %  | {t_primero:>7.1f} s     | {nota}")

    print("-" * 85)
    mejor_reg = candidatos[0][0]
    t_start = max(0, int(candidatos[0][3] - 2))
    print(f"  -> Registro más destacado: {mejor_reg} ({candidatos[0][1]} latidos, {candidatos[0][2]:.1f}%)")
    print(f"  -> Prueba rápida sugerida (30s):")
    print(f"     python maestro.py --mode extract-csv --record {mejor_reg} --duration 30 --start {t_start}")
    print("=" * 85)

def mostrar_guia_general(catalogo):
    """Muestra la tabla de las arritmias más importantes y los pacientes clave para cada una."""
    print("=" * 85)
    print("   GUÍA RÁPIDA DE REGISTROS DESTACADOS POR ARRITMIA (MIT-BIH DATABASE)")
    print("=" * 85)
    print(" Registros ideales para evaluar el comportamiento del modelo ante cada condición:\n")

    secciones = [
        ('V', 'Contracciones Ventriculares Prematuras (PVCs) y Salvas de VT', ['208', '233', '200', '106', '207']),
        ('L', 'Bloqueo de Rama Izquierda (LBBB) - Complejos Anchos', ['109', '111', '214', '207']),
        ('R', 'Bloqueo de Rama Derecha (RBBB) - Morfología rSR\'', ['118', '212', '124', '231']),
        ('A', 'Contracciones Auriculares Prematuras (PACs)', ['232', '209', '222']),
        ('/', 'Ritmo de Marcapasos Artificial (Paced Beats)', ['107', '102', '217', '104']),
        ('F', 'Latidos de Fusión Ventricular (Normal + PVC)', ['208', '213'])
    ]

    for sym, titulo, lista_top in secciones:
        print(f" [Clase {sym}] {titulo}")
        print(f" {'Registro':<10} | {'Latidos':<10} | {'Incidencia':<12} | {'Aparición':<12} | {'Comentario'}")
        print(" " + "-" * 80)

        for reg in lista_top:
            if reg in catalogo:
                cnt = catalogo[reg]["conteo"].get(sym, 0)
                tot = catalogo[reg]["total"]
                pct = (cnt / tot * 100) if tot > 0 else 0
                t_first = catalogo[reg]["primeros_seg"].get(sym, 0.0)
                nota = NOTAS_CLINICAS.get(reg, "")
                if len(nota) > 34:
                    nota = nota[:31] + "..."
                print(f" {reg:<10} | {cnt:<10} | {pct:>7.2f} %    | {t_first:>6.1f} s     | {nota}")
        print()

    print("=" * 85)
    print(" TIP CLÍNICO:")
    print(" - Para probar Ventriculares severas: usa el Registro 200 (desde 0s) o 208 (desde 0s).")
    print(" - Para probar Bloqueo Izquierdo:     usa el Registro 109 (desde 0s).")
    print(" - Para probar Bloqueo Derecho:       usa el Registro 118 (desde 0s).")
    print(" - Para probar Marcapasos:            usa el Registro 107 (desde 0s).")
    print(" - Para probar Auriculares:           usa el Registro 232 (desde 0s).")
    print("=" * 85)

def exportar_catalogo_csv(catalogo, ruta_salida="Data/catalogo_arritmias_mitbih.csv"):
    """Exporta un resumen tabular completo a CSV para consulta externa."""
    try:
        import pandas as pd
    except ImportError:
        print("Pandas no disponible, no se exportó a CSV.")
        return

    filas = []
    for reg, d in catalogo.items():
        tot = d["total"]
        fila = {
            "Registro": reg,
            "Total_Latidos": tot,
            "Latidos_N": d["conteo"].get("N", 0),
            "Latidos_V_Ventricular": d["conteo"].get("V", 0),
            "Latidos_A_Auricular": d["conteo"].get("A", 0),
            "Latidos_L_BloqueoIzq": d["conteo"].get("L", 0),
            "Latidos_R_BloqueoDer": d["conteo"].get("R", 0),
            "Latidos_Paced_Marcapasos": d["conteo"].get("/", 0),
            "Latidos_F_FusionVent": d["conteo"].get("F", 0),
            "Latidos_f_FusionMarcapasos": d["conteo"].get("f", 0),
            "Primer_V_seg": d["primeros_seg"].get("V", -1),
            "Primer_A_seg": d["primeros_seg"].get("A", -1),
            "Primer_L_seg": d["primeros_seg"].get("L", -1),
            "Primer_R_seg": d["primeros_seg"].get("R", -1),
            "Primer_Paced_seg": d["primeros_seg"].get("/", -1),
            "Observacion": NOTAS_CLINICAS.get(reg, "")
        }
        filas.append(fila)

    df = pd.DataFrame(filas)
    os.makedirs(os.path.dirname(ruta_salida), exist_ok=True)
    df.to_csv(ruta_salida, index=False)
    print(f"\n[OK] Catálogo completo exportado a: {ruta_salida}")

def main():
    parser = argparse.ArgumentParser(
        description="Explorador de registros con arritmias específicas (MIT-BIH Database)"
    )
    parser.add_argument(
        "--tipo", "-t",
        type=str,
        default=None,
        help="Símbolo de la arritmia a consultar (V: Ventricular, A: Auricular, L: Bloqueo Izq, R: Bloqueo Der, /: Marcapasos, F: Fusión, todas)"
    )
    parser.add_argument(
        "--registro", "-r",
        type=str,
        default=None,
        help="Número de registro específico a inspeccionar (ej: 200, 208, 109)"
    )
    parser.add_argument(
        "--top", "-n",
        type=int,
        default=5,
        help="Cantidad de registros a mostrar por arritmia (default: 5)"
    )
    parser.add_argument(
        "--csv",
        action="store_true",
        help="Exportar el catálogo consolidado de todos los pacientes a un archivo CSV"
    )

    args = parser.parse_args()

    # Cargar y analizar anotaciones oficiales
    catalogo = analizar_todos_los_registros("Data/Database")
    if not catalogo:
        return

    if args.csv:
        exportar_catalogo_csv(catalogo)

    if args.registro:
        reg_id = args.registro.strip()
        if reg_id in catalogo:
            mostrar_desglose_registro(reg_id, catalogo[reg_id])
        else:
            print(f"Error: El registro '{reg_id}' no se encontró en la base de datos.")
        return

    if args.tipo:
        tipo = args.tipo.strip().upper()
        if tipo in NOMBRES_ARRITMIAS:
            mostrar_top_por_arritmia(catalogo, tipo, top_n=args.top)
        elif tipo in ["TODAS", "ALL"]:
            for t in ['V', 'L', 'R', 'A', '/', 'F']:
                mostrar_top_por_arritmia(catalogo, t, top_n=args.top)
        else:
            print(f"Arritmia '{tipo}' no reconocida. Opciones comunes: V, A, L, R, /, F, f")
        return

    # Si no se pasó argumento específico, mostrar la guía curada general
    mostrar_guia_general(catalogo)

if __name__ == "__main__":
    main()

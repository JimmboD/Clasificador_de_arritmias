#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
=============================================================================
CARDIOLAB RESEARCH - LANZADOR DE APLICACION WEB EXPERIMENTAL
=============================================================================
Al ejecutar este archivo (`python main.py`):
1. Se inicia el servidor web local interactivo.
2. Se abre automáticamente tu navegador predeterminado en http://127.0.0.1:5000.
3. Puedes operar todas las fases del clasificador: diagnóstico end-to-end,
   evaluación de tensores, escaneo de Taquicardia Ventricular, inspección de
   datos, preprocesamiento y gestión de datasets.

Para ejecutar en modo consola o por comandos de terminal (CLI), utiliza:
  python maestro.py --help
=============================================================================
"""

import sys
import argparse
from pathlib import Path

# Asegurar que el directorio raíz esté en sys.path
directorio_raiz = Path(__file__).resolve().parent
if str(directorio_raiz) not in sys.path:
    sys.path.insert(0, str(directorio_raiz))

from src.web.server import iniciar_servidor_web

# Configurar salida UTF-8 en Windows
if sys.platform == "win32":
    try:
        if hasattr(sys.stdout, 'reconfigure'):
            sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        if hasattr(sys.stderr, 'reconfigure'):
            sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

def parsear_argumentos():
    parser = argparse.ArgumentParser(
        description="Lanzador de CardioAI - Interfaz Web Local para Clasificación de Arritmias",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Dirección IP del servidor local (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=5000, help="Puerto del servidor local (default: 5000)")
    parser.add_argument("--no-browser", action="store_true", help="No abrir el navegador automáticamente al iniciar")
    parser.add_argument("--debug", action="store_true", help="Iniciar en modo debug de Flask")
    parser.add_argument("--cli", action="store_true", help="Delegar al menú maestro por consola (maestro.py)")
    return parser.parse_args()

def main():
    args = parsear_argumentos()

    if args.cli:
        from maestro import main as maestro_main
        maestro_main()
        return

    print("=" * 70)
    print("   [CardioAI] - CLASIFICADOR DE ARRITMIAS CARDIACAS (CNN & STFT)")
    print("=" * 70)
    print(f"-> Servidor local en http://{args.host}:{args.port}")
    print(f"-> Abriendo navegador automaticamente...")
    print(f"-> Para modo consola por comandos, utiliza: python maestro.py")
    print("=" * 70)

    iniciar_servidor_web(
        host=args.host,
        port=args.port,
        abrir_navegador=not args.no_browser,
        debug=args.debug
    )

if __name__ == "__main__":
    main()

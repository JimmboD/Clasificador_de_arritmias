import numpy as np
from scipy import signal
from scipy.signal import butter, filtfilt

class PreprocesadorECG:
    """
    Clase encargada de acondicionar la señal cruda del ECG,
    aplicar filtros digitales, segmentar ciclos cardíacos y 
    generar los tensores espectrogramas (29, 112) para la CNN.
    """

    def __init__(self, frecuencia_muestreo=360, frecuencia_corte=40):
        self.fs = frecuencia_muestreo
        self.fc = frecuencia_corte

    # --- 1. Filtros digitales ---
    def filtros_digitales(self, senal, orden=4):
        """
        Aplica un filtro pasa-bajas Butterworth de fase cero para eliminar
        ruidos de alta frecuencia por encima de la frecuencia de corte.
        """
        nyquist = 0.5 * self.fs
        corte_normalizado = self.fc / nyquist
        b, a = butter(orden, corte_normalizado, btype='low')
        senal_filtrada = filtfilt(b, a, senal)
        return senal_filtrada

    # --- 2. Segmentar picos R ---
    def segmentar_pulsos(self, senal, umbral_pendiente=None):
        """
        Detecta los picos R mediante análisis de pendiente y período refractario.
        """
        if umbral_pendiente is None:
            pendiente = 50 * (1 / self.fs)
        else:
            pendiente = umbral_pendiente

        pulsos = []
        salto = int(0.2 * self.fs)  # 200 ms de período refractario
        i = 0

        while i < len(senal) - 1:
            valor_siguiente = senal[i + 1]
            valor_actual = senal[i]

            if (valor_siguiente - valor_actual) >= pendiente:
                while i < len(senal) - 1 and senal[i + 1] >= senal[i]:
                    i += 1
                pulsos.append(i)
                i += salto
            else:
                i += 1

        return pulsos

    # --- 3. Extraer la ventana asimétrica de cada latido ---
    def generar_marcadores_pulsos(self, senal, lista_picos, tiempo_ventana=0.75):
        """
        Calcula los límites de ventana para cada latido (35% antes del pico R, 65% después).
        Retorna: [[inicio1, fin1], [inicio2, fin2], ...]
        """
        muestras_totales = int(tiempo_ventana * self.fs)
        muestras_izq = int(muestras_totales * 0.35)
        muestras_der = muestras_totales - muestras_izq

        marcadores = []
        for pico_r in lista_picos:
            centro_r = int(pico_r)
            inicio = centro_r - muestras_izq
            fin = centro_r + muestras_der

            if inicio >= 0 and fin <= len(senal):
                marcadores.append([inicio, fin])

        return marcadores

    # --- 4. Emparejar picos con anotaciones del cardiólogo ---
    def emparejar_etiquetas(self, lista_picos, anotaciones_oficiales, tolerancia_segundos=0.222):
        """
        Asocia cada pico detectado con una etiqueta oficial dentro del margen de tolerancia.
        anotaciones_oficiales: [(muestra, 'simbolo'), ...]
        """
        tolerancia_muestras = int(tolerancia_segundos * self.fs)
        picos_etiquetados = []

        for pico_detectado in lista_picos:
            etiqueta_asignada = "Desconocido"
            for muestra_oficial, simbolo in anotaciones_oficiales:
                if abs(muestra_oficial - pico_detectado) <= tolerancia_muestras:
                    etiqueta_asignada = simbolo
                    break
            picos_etiquetados.append((pico_detectado, etiqueta_asignada))

        return picos_etiquetados

    # --- 5. Estandarización / Remuestreo ---
    def estandarizar_ventana(self, ventana_pulso, muestras_esperadas=270):
        """
        Aplica remuestreo basado en Fourier para fijar la longitud a 270 muestras.
        """
        return signal.resample(ventana_pulso, muestras_esperadas)

    # --- 6. Transformada de Fourier de Tiempo Corto (STFT / Espectrograma) ---
    def obtener_tensor_espectrograma(self, ventana_procesada, nperseg=48, etiqueta=None):
        """
        Genera el espectrograma 2D recortado (f <= 40 Hz), en escala logarítmica y normalizado Min-Max.
        Produce exactamente una matriz de tamaño (29, 112).
        """
        x = np.asarray(ventana_procesada)
        f, t, Sxx = signal.spectrogram(x, fs=self.fs, nperseg=nperseg, noverlap=46, nfft=256)

        indice_limite = np.where(f <= self.fc)[0][-1] + 1
        Sxx_recortado = Sxx[:indice_limite, :]

        Sxx_log = 10 * np.log10(Sxx_recortado + 1e-10)
        Sxx_norm = (Sxx_log - np.min(Sxx_log)) / (np.max(Sxx_log) - np.min(Sxx_log) + 1e-10)

        tensor_final = Sxx_norm

        if etiqueta is not None:
            return {"accion": "empaquetar", "datos": tensor_final, "etiqueta": etiqueta}
        else:
            return {"accion": "predecir", "datos": tensor_final}

    # --- 7. Pipeline orquestador maestro ---
    def pre_procesar(self, senal, etiquetas_bd=None):
        """
        Orquesta todo el pipeline: filtra la señal, detecta picos, empareja etiquetas,
        recorta ventanas asimétricas, estandariza a 270 muestras y genera espectrogramas.
        """
        senal_limpia = self.filtros_digitales(senal)
        picos_r = self.segmentar_pulsos(senal_limpia)

        if etiquetas_bd is not None:
            picos_etiquetados = self.emparejar_etiquetas(picos_r, etiquetas_bd)
        else:
            picos_etiquetados = [(pico, None) for pico in picos_r]

        solo_picos = [pico[0] for pico in picos_etiquetados]
        marcadores = self.generar_marcadores_pulsos(senal_limpia, solo_picos)

        X_datos = []
        Y_etiquetas = []

        for (inicio, fin), (centro_r, etiqueta) in zip(marcadores, picos_etiquetados):
            ventana_pulso = senal_limpia[inicio:fin]
            ventana_estandarizada = self.estandarizar_ventana(ventana_pulso, muestras_esperadas=270)
            paquete = self.obtener_tensor_espectrograma(ventana_estandarizada, etiqueta=etiqueta)

            if paquete["accion"] == "empaquetar" and paquete["etiqueta"] != "Desconocido":
                X_datos.append(paquete["datos"])
                Y_etiquetas.append(paquete["etiqueta"])
            elif paquete["accion"] == "predecir":
                X_datos.append(paquete["datos"])

        if etiquetas_bd is not None:
            return {"X": np.array(X_datos), "Y": np.array(Y_etiquetas)}
        else:
            return np.array(X_datos)


# Alias de retrocompatibilidad
Pre_procesador_EMG = PreprocesadorECG

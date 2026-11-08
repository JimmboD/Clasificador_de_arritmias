/**
 * CardioLab Research - Controlador Frontend (Vanilla JavaScript)
 * Plataforma Experimental de Análisis Electrocardiográfico y Detección de Arritmias
 */

document.addEventListener('DOMContentLoaded', () => {
    // --- ESTADO GLOBAL ---
    let currentInfo = null;
    let ecgAnimationId = null;
    let ecgPoints = [];
    let ecgOffset = 0;
    let isEcgPaused = false;

    // Estado del análisis CSV
    let csvSelectedFile = null;
    let csvAnalysisResult = null;
    let currentPulseIndex = 0;
    let activeFilter = 'all';

    // --- ELEMENTOS DOM: NAVEGACIÓN ---
    const navItems = document.querySelectorAll('.nav-item');
    const tabPanes = document.querySelectorAll('.tab-pane');
    const sectionTitle = document.getElementById('section-title');
    const sectionDesc = document.getElementById('section-desc');

    // Elementos Tab CSV (Workflow Principal)
    const csvDropzone = document.getElementById('csv-dropzone');
    const inputCsvFile = document.getElementById('input-csv-file');
    const csvFileName = document.getElementById('csv-file-name');
    const btnBrowseCsv = document.getElementById('btn-browse-csv');
    const btnLoadSampleCsv = document.getElementById('btn-load-sample-csv');
    const inputCsvFs = document.getElementById('input-csv-fs');
    const selectCsvModel = document.getElementById('select-csv-model');
    const btnRunCsv = document.getElementById('btn-run-csv');
    const csvResultsWrapper = document.getElementById('csv-results-wrapper');
    const canvasCsvOverview = document.getElementById('csv-canvas-overview');
    const tbodyCsvPulses = document.getElementById('tbody-csv-pulses');
    const btnExportCsvReport = document.getElementById('btn-export-csv-report');
    const pillButtons = document.querySelectorAll('.pill-btn');

    // Elementos Extracción Pura desde Base Natural MIT-BIH
    const selectExtractRecord = document.getElementById('select-extract-record');
    const inputExtractDuration = document.getElementById('input-extract-duration');
    const inputExtractStart = document.getElementById('input-extract-start');
    const btnExtractLoad = document.getElementById('btn-extract-load');
    const btnExtractDownload = document.getElementById('btn-extract-download');
    const extractStatusBanner = document.getElementById('extract-status-banner');
    const chipBtns = document.querySelectorAll('.chip-btn');

    // Elementos Visor de Pulso Clínico (Gráficas Reciclables)
    const canvasPulse1D = document.getElementById('canvas-pulse-1d');
    const canvasPulseSTFT = document.getElementById('canvas-pulse-stft');
    const viewerTitle = document.getElementById('viewer-title');
    const viewerBadgeSeverity = document.getElementById('viewer-badge-severity');
    const viewerPulseCounter = document.getElementById('viewer-pulse-counter');
    const btnPrevPulse = document.getElementById('btn-prev-pulse');
    const btnNextPulse = document.getElementById('btn-next-pulse');
    const pulseProbBars = document.getElementById('pulse-prob-bars');

    // Elementos Tabs Secundarios
    const selectPatientRaw = document.getElementById('select-patient-raw');
    const selectModelRaw = document.getElementById('select-model-raw');
    const btnRunRaw = document.getElementById('btn-run-raw');

    const selectNpzFile = document.getElementById('select-npz-file');
    const selectModelNpz = document.getElementById('select-model-npz');
    const btnRunNpz = document.getElementById('btn-run-npz');

    const btnScanVt = document.getElementById('btn-scan-vt');
    const selectInspectPatient = document.getElementById('select-inspect-patient');
    const btnRunInspect = document.getElementById('btn-run-inspect');

    const btnPreprocessSingle = document.getElementById('btn-preprocess-single');
    const btnPreprocessAll = document.getElementById('btn-preprocess-all');
    const btnUnifyDataset = document.getElementById('btn-unify-dataset');
    const btnFilterDataset = document.getElementById('btn-filter-dataset');
    const btnToggleEcg = document.getElementById('btn-toggle-ecg');

    const ecgCanvas = document.getElementById('ecg-canvas');
    const ecgCtx = ecgCanvas ? ecgCanvas.getContext('2d') : null;

    // Mapeo clínico riguroso (AAMI EC57)
    const NOMBRES_CLINICOS = {
        'N': 'Latido Normal (Sinus Beat)',
        '/': 'Latido Estimulado por Marcapasos (Paced Beat)',
        'R': 'Bloqueo de Rama Derecha (RBBB)',
        'L': 'Bloqueo de Rama Izquierda (LBBB)',
        'V': 'Contracción Ventricular Prematura (PVC)',
        'A': 'Contracción Auricular Prematura (PAC)',
        'f': 'Fusión de Latido Marcapasos y Normal',
        'F': 'Fusión de Latido Ventricular y Normal',
        'No_Clasificable': 'Rechazo Clínico / Incertidumbre Diagnóstica'
    };

    // Metadata descriptiva de pestañas
    const TAB_INFO = {
        'tab-csv': {
            title: 'Clasificación de Señal Externa (CSV)',
            desc: 'Carga cualquier archivo con datos de voltaje vs tiempo. El sistema filtrará la señal, detectará cada latido, calculará su espectrograma y clasificará cada pulso con gráficas reciclables en memoria.'
        },
        'tab-eval-raw': {
            title: 'Inferencia Clínica MIT-BIH End-to-End',
            desc: 'Procesamiento digital de señal cruda ECG (360 Hz) desde registros MIT-BIH (.dat/.hea/.atr) con CNN con opción a rechazo.'
        },
        'tab-eval-npz': {
            title: 'Evaluación de Tensores Preprocesados (.npz)',
            desc: 'Evaluación rápida de lotes de tensores espectrogramas (29 × 112) ya extraídos y normalizados.'
        },
        'tab-vt': {
            title: 'Detector de Taquicardia Ventricular (VT)',
            desc: 'Búsqueda exhaustiva por etiquetas de ritmo y ráfagas consecutivas de 3 o más latidos ventriculares.'
        },
        'tab-inspect': {
            title: 'Auditoría de Anotaciones Oficiales (.atr)',
            desc: 'Auditoría de consistencia y cotejo cuantitativo entre latidos detectados por el algoritmo y anotaciones del cardiólogo.'
        },
        'tab-data': {
            title: 'Curaduría & Pipeline de Datos',
            desc: 'Filtrado Butterworth pasa-bajas (40 Hz), extracción de espectrogramas, unificación y depuración estándar AAMI.'
        }
    };

    // --- TAB NAVIGATION ---
    navItems.forEach(item => {
        item.addEventListener('click', () => {
            const targetTab = item.getAttribute('data-tab');

            navItems.forEach(n => n.classList.remove('active'));
            tabPanes.forEach(p => p.classList.remove('active'));

            item.classList.add('active');
            const targetPane = document.getElementById(targetTab);
            if (targetPane) targetPane.classList.add('active');

            if (TAB_INFO[targetTab]) {
                sectionTitle.textContent = TAB_INFO[targetTab].title;
                sectionDesc.textContent = TAB_INFO[targetTab].desc;
            }
        });
    });

    // --- CARGAR INFORMACIÓN DEL SISTEMA ---
    async function cargarInfoInicial() {
        try {
            const res = await fetch('/api/info');
            if (!res.ok) throw new Error(`HTTP error ${res.status}`);
            const data = await res.json();
            currentInfo = data;

            // Stat model
            const statModel = document.getElementById('stat-model');
            if (statModel && data.available_models && data.available_models.length > 0) {
                const tieneV6 = data.available_models.some(m => m.includes('6.0'));
                statModel.textContent = tieneV6 ? 'V6.0 (Rechazo)' : data.available_models[0];
            }

            // Llenar pacientes
            if (selectPatientRaw && selectInspectPatient) {
                selectPatientRaw.innerHTML = '';
                selectInspectPatient.innerHTML = '';
                (data.records || []).forEach(rec => {
                    selectPatientRaw.add(new Option(`Paciente ${rec}`, rec));
                    selectInspectPatient.add(new Option(`Registro ${rec}`, rec));
                });
            }

            // Llenar registros para extracción pura
            if (selectExtractRecord) {
                selectExtractRecord.innerHTML = '';
                (data.records || []).forEach(rec => {
                    const opt = new Option(`Registro ${rec}`, rec);
                    if (rec === '200') opt.selected = true;
                    selectExtractRecord.add(opt);
                });
            }

            // Llenar modelos
            const modelSelects = [selectModelRaw, selectModelNpz, selectCsvModel];
            const modelos = (data.available_models && data.available_models.length > 0)
                ? data.available_models
                : ['6.0', '5.0', '4.0', '2.0', '1.0'];

            modelSelects.forEach(sel => {
                if (!sel) return;
                sel.innerHTML = '';
                modelos.forEach(m => {
                    const opt = new Option(m, m);
                    if (m.includes('6.0') || m === '6.0') opt.selected = true;
                    sel.add(opt);
                });
            });

            // Llenar archivos .npz
            if (selectNpzFile) {
                selectNpzFile.innerHTML = '';
                if (data.npz_files && data.npz_files.length > 0) {
                    data.npz_files.forEach(f => {
                        selectNpzFile.add(new Option(f, `Data/Raw_data/${f}`));
                    });
                } else {
                    selectNpzFile.add(new Option('100.npz', 'Data/Raw_data/100.npz'));
                }
            }

            // Iniciar simulador de ECG para pestaña MIT-BIH
            generarECGSimulado();
            if (!ecgAnimationId) animarECG();

        } catch (err) {
            mostrarToast('Error al conectar con la API de CardioLab: ' + err.message, 'error');
        }
    }

    // =========================================================================
    // WORKFLOW PRINCIPAL: SUBIDA Y ANÁLISIS DE SEÑAL CSV (PULSO POR PULSO)
    // =========================================================================

    // Drag and Drop & File Picker
    if (csvDropzone && inputCsvFile) {
        csvDropzone.addEventListener('click', (e) => {
            if (e.target !== btnLoadSampleCsv && !btnLoadSampleCsv.contains(e.target)) {
                inputCsvFile.click();
            }
        });

        if (btnBrowseCsv) {
            btnBrowseCsv.addEventListener('click', (e) => {
                e.stopPropagation();
                inputCsvFile.click();
            });
        }

        inputCsvFile.addEventListener('change', () => {
            if (inputCsvFile.files.length > 0) {
                csvSelectedFile = inputCsvFile.files[0];
                csvFileName.textContent = `Archivo seleccionado: ${csvSelectedFile.name} (${(csvSelectedFile.size / 1024).toFixed(1)} KB)`;
                mostrarToast(`Archivo '${csvSelectedFile.name}' cargado listo para análisis`, 'info');
            }
        });

        csvDropzone.addEventListener('dragover', (e) => {
            e.preventDefault();
            csvDropzone.classList.add('dragover');
        });

        csvDropzone.addEventListener('dragleave', () => {
            csvDropzone.classList.remove('dragover');
        });

        csvDropzone.addEventListener('drop', (e) => {
            e.preventDefault();
            csvDropzone.classList.remove('dragover');
            if (e.dataTransfer.files.length > 0) {
                csvSelectedFile = e.dataTransfer.files[0];
                csvFileName.textContent = `Archivo seleccionado: ${csvSelectedFile.name} (${(csvSelectedFile.size / 1024).toFixed(1)} KB)`;
                mostrarToast(`Archivo '${csvSelectedFile.name}' soltado correctamente`, 'info');
            }
        });
    }

    // Botón para cargar señal de prueba (20s)
    if (btnLoadSampleCsv) {
        btnLoadSampleCsv.addEventListener('click', (e) => {
            e.stopPropagation();
            csvSelectedFile = 'SAMPLE_DATA';
            csvFileName.textContent = "Archivo de prueba cargado: Data/ejemplo_ecg_voltaje_tiempo.csv (20 segundos a 360 Hz)";
            mostrarToast("Señal de prueba cargada lista para clasificar", 'info');
        });
    }

    // Ejecutar análisis CSV
    if (btnRunCsv) {
        btnRunCsv.addEventListener('click', async () => {
            if (!csvSelectedFile) {
                mostrarToast('Por favor selecciona un archivo CSV o haz clic en "Cargar Señal de Prueba".', 'warning');
                return;
            }

            const fsVal = inputCsvFs ? inputCsvFs.value : '360';
            const modelVal = selectCsvModel ? selectCsvModel.value : '6.0';

            setLoading(btnRunCsv, true, 'Filtrando, segmentando y clasificando...');

            try {
                let res;
                if (typeof csvSelectedFile === 'string') {
                    const filePath = (csvSelectedFile === 'SAMPLE_DATA') ? 'Data/ejemplo_ecg_voltaje_tiempo.csv' : csvSelectedFile;
                    res = await fetch('/api/analyze/csv', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                            file_path: filePath,
                            fs: fsVal,
                            model: modelVal
                        })
                    });
                } else {
                    const formData = new FormData();
                    formData.append('file', csvSelectedFile);
                    formData.append('fs', fsVal);
                    formData.append('model', modelVal);

                    res = await fetch('/api/analyze/csv', {
                        method: 'POST',
                        body: formData
                    });
                }

                const data = await res.json();
                if (data.error) throw new Error(data.error);

                csvAnalysisResult = data;
                renderizarResultadosCsv(data);

                mostrarToast(`Análisis finalizado con éxito: ${data.resumen.total_pulsos} pulsos clasificados`, 'success');
            } catch (err) {
                mostrarToast(err.message, 'error');
            } finally {
                setLoading(btnRunCsv, false, 'Filtrar, Segmentar y Clasificar Señal');
            }
        });
    }

    // Chips de duración rápida (10s, 20s, 30s, 60s)
    chipBtns.forEach(chip => {
        chip.addEventListener('click', () => {
            chipBtns.forEach(c => c.classList.remove('active'));
            chip.classList.add('active');
            if (inputExtractDuration) {
                inputExtractDuration.value = chip.dataset.sec;
            }
        });
    });

    if (inputExtractDuration) {
        inputExtractDuration.addEventListener('input', () => {
            const val = inputExtractDuration.value.trim();
            chipBtns.forEach(c => {
                if (c.dataset.sec === val) {
                    c.classList.add('active');
                } else {
                    c.classList.remove('active');
                }
            });
        });
    }

    // Botón para Extraer Segmento Puro de Base de Datos y Cargar al Workflow
    if (btnExtractLoad) {
        btnExtractLoad.addEventListener('click', async () => {
            const record = selectExtractRecord ? selectExtractRecord.value : '200';
            const duration = inputExtractDuration ? parseFloat(inputExtractDuration.value) || 20 : 20;
            const start = inputExtractStart ? parseFloat(inputExtractStart.value) || 0 : 0;

            setLoading(btnExtractLoad, true, 'Extrayendo señal pura...');
            if (extractStatusBanner) extractStatusBanner.style.display = 'none';

            try {
                const res = await fetch('/api/database/extract-csv', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        record: record,
                        duration_sec: duration,
                        start_sec: start
                    })
                });

                const data = await res.json();
                if (data.error) throw new Error(data.error);

                // Asignar al workflow CSV
                csvSelectedFile = data.ruta_archivo;
                if (inputCsvFs) inputCsvFs.value = data.fs;
                if (csvFileName) {
                    csvFileName.textContent = `Muestra extraída: ${data.nombre_archivo} (${data.puntos} puntos @ ${data.fs} Hz, ${data.duracion_seg}s) - Cero Fuga de Datos`;
                }

                if (extractStatusBanner) {
                    extractStatusBanner.style.display = 'block';
                    extractStatusBanner.style.background = 'rgba(16, 185, 129, 0.1)';
                    extractStatusBanner.style.borderColor = 'rgba(16, 185, 129, 0.3)';
                    extractStatusBanner.style.color = '#34D399';
                    extractStatusBanner.innerHTML = `<strong>Muestra extraída con éxito:</strong> <code>${data.nombre_archivo}</code> (${data.duracion_seg} segundos, ${data.puntos} muestras a ${data.fs} Hz). Señal física continua pura sin marcas ni etiquetas previas. Lista para clasificar.`;
                }

                mostrarToast(`Muestra de ${data.duracion_seg}s (Registro ${data.registro}) cargada al workflow`, 'success');

                if (btnRunCsv) {
                    btnRunCsv.scrollIntoView({ behavior: 'smooth', block: 'center' });
                }
            } catch (err) {
                mostrarToast('Error al extraer registro: ' + err.message, 'error');
                if (extractStatusBanner) {
                    extractStatusBanner.style.display = 'block';
                    extractStatusBanner.style.background = 'rgba(239, 68, 68, 0.1)';
                    extractStatusBanner.style.borderColor = 'rgba(239, 68, 68, 0.3)';
                    extractStatusBanner.style.color = '#F87171';
                    extractStatusBanner.textContent = 'Error: ' + err.message;
                }
            } finally {
                setLoading(btnExtractLoad, false, 'Extraer y Cargar al Workflow');
            }
        });
    }

    // Botón para Descargar CSV Crudo
    if (btnExtractDownload) {
        btnExtractDownload.addEventListener('click', async () => {
            const record = selectExtractRecord ? selectExtractRecord.value : '200';
            const duration = inputExtractDuration ? parseFloat(inputExtractDuration.value) || 20 : 20;
            const start = inputExtractStart ? parseFloat(inputExtractStart.value) || 0 : 0;

            setLoading(btnExtractDownload, true, 'Generando CSV...');

            try {
                const res = await fetch('/api/database/extract-csv', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        record: record,
                        duration_sec: duration,
                        start_sec: start
                    })
                });

                const data = await res.json();
                if (data.error) throw new Error(data.error);

                mostrarToast(`Descargando ${data.nombre_archivo}...`, 'info');
                window.location.href = `/api/database/download-extracted?file=${encodeURIComponent(data.ruta_archivo)}`;
            } catch (err) {
                mostrarToast('Error al generar CSV para descarga: ' + err.message, 'error');
            } finally {
                setLoading(btnExtractDownload, false, 'Descargar CSV');
            }
        });
    }

    function renderizarResultadosCsv(data) {
        if (!csvResultsWrapper) return;
        csvResultsWrapper.style.display = 'flex';

        const resumen = data.resumen;
        document.getElementById('csv-stat-total').textContent = resumen.total_pulsos;
        document.getElementById('csv-stat-duration').textContent = `Duración: ${resumen.duracion_seg}s (${resumen.fs} Hz)`;

        document.getElementById('csv-stat-bpm').textContent = `${resumen.bpm_medio} BPM`;
        document.getElementById('csv-stat-bpm-range').textContent = `Rango: ${resumen.bpm_min} - ${resumen.bpm_max} BPM`;

        document.getElementById('csv-stat-normal').textContent = resumen.normales;
        const normPct = resumen.total_pulsos > 0 ? (resumen.normales / resumen.total_pulsos * 100).toFixed(1) : 0;
        document.getElementById('csv-stat-normal-pct').textContent = `${normPct}% de la señal`;

        document.getElementById('csv-stat-arrhythmias').textContent = resumen.anormales;
        document.getElementById('csv-stat-arrhythmias-pct').textContent = `${resumen.tasa_arritmias_pct}% incidencia`;

        // Contadores en píldoras de filtro
        document.getElementById('pill-count-all').textContent = resumen.total_pulsos;
        document.getElementById('pill-count-arr').textContent = resumen.anormales;
        document.getElementById('pill-count-norm').textContent = resumen.normales;

        // Renderizar monitor global de señal continua
        renderizarMonitorOverview(data.preview_senal, data.picos_preview, data.pulsos);

        // Renderizar tabla de pulsos
        renderizarTablaPulsos(data.pulsos, activeFilter);

        // Seleccionar y visualizar el primer pulso
        if (data.pulsos.length > 0) {
            seleccionarPulso(0);
        }
    }

    // Monitor Continuo de la Señal Filtrada con Picos R
    function renderizarMonitorOverview(senal, picos, pulsos) {
        if (!canvasCsvOverview || !senal || senal.length === 0) return;
        const ctx = canvasCsvOverview.getContext('2d');
        const width = canvasCsvOverview.width;
        const height = canvasCsvOverview.height;
        const midY = height / 2;

        ctx.fillStyle = '#060B12';
        ctx.fillRect(0, 0, width, height);

        // Cuadrícula médica fina
        ctx.strokeStyle = 'rgba(16, 185, 129, 0.05)';
        ctx.lineWidth = 1;
        for (let x = 0; x < width; x += 20) {
            ctx.beginPath();
            ctx.moveTo(x, 0);
            ctx.lineTo(x, height);
            ctx.stroke();
        }
        for (let y = 0; y < height; y += 20) {
            ctx.beginPath();
            ctx.moveTo(0, y);
            ctx.lineTo(width, y);
            ctx.stroke();
        }

        // Trazado de señal continua
        ctx.strokeStyle = '#0EA5E9';
        ctx.lineWidth = 1.6;
        ctx.beginPath();

        const stepX = width / senal.length;
        const maxAmp = Math.max(...senal.map(Math.abs), 0.5);
        const scaleY = (height * 0.42) / maxAmp;

        for (let i = 0; i < senal.length; i++) {
            const x = i * stepX;
            const y = midY - (senal[i] * scaleY);
            if (i === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
        }
        ctx.stroke();

        // Marcadores de Picos R clasificados
        const picosSet = new Map();
        pulsos.forEach(p => picosSet.set(p.muestra_r, p));

        picos.forEach(picoIdx => {
            const x = picoIdx * stepX;
            const pulsoInfo = picosSet.get(picoIdx);
            const esArritmia = pulsoInfo && pulsoInfo.clase !== 'N';

            ctx.strokeStyle = esArritmia ? '#EF4444' : '#10B981';
            ctx.lineWidth = 1.2;
            ctx.setLineDash([3, 3]);
            ctx.beginPath();
            ctx.moveTo(x, 4);
            ctx.lineTo(x, height - 4);
            ctx.stroke();
            ctx.setLineDash([]);

            if (pulsoInfo) {
                ctx.fillStyle = esArritmia ? '#EF4444' : '#10B981';
                ctx.font = 'bold 9px monospace';
                ctx.fillText(pulsoInfo.clase, x + 3, 14);
            }
        });
    }

    // Tabla de Pulsos
    function renderizarTablaPulsos(pulsos, filtro = 'all') {
        if (!tbodyCsvPulses) return;
        tbodyCsvPulses.innerHTML = '';

        pulsos.forEach((p, idx) => {
            const esNormal = p.clase === 'N';
            if (filtro === 'normal' && !esNormal) return;
            if (filtro === 'arrhythmias' && esNormal) return;

            const tr = document.createElement('tr');
            tr.dataset.index = idx;
            if (idx === currentPulseIndex) tr.classList.add('table-row-selected');

            tr.innerHTML = `
                <td><strong>#${p.id}</strong></td>
                <td><code>${p.tiempo_str}</code></td>
                <td>${p.rr_ms} ms</td>
                <td>${p.bpm}</td>
                <td><span class="badge ${p.badge_class}">${p.clase}</span></td>
                <td>${p.nombre_clinico}</td>
                <td><strong>${p.confianza}%</strong></td>
            `;

            tr.addEventListener('click', () => {
                seleccionarPulso(idx);
            });

            tbodyCsvPulses.appendChild(tr);
        });
    }

    // Píldoras de Filtro
    pillButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            pillButtons.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            activeFilter = btn.dataset.filter;
            if (csvAnalysisResult) {
                renderizarTablaPulsos(csvAnalysisResult.pulsos, activeFilter);
            }
        });
    });

    // =========================================================================
    // VISOR DE PULSO CLÍNICO: GRÁFICAS RECICLABLES EN CANVAS (0 MB EN DISCO)
    // =========================================================================

    function seleccionarPulso(index) {
        if (!csvAnalysisResult || !csvAnalysisResult.pulsos[index]) return;
        currentPulseIndex = index;
        const pulso = csvAnalysisResult.pulsos[index];
        const total = csvAnalysisResult.pulsos.length;

        // Actualizar header del visor
        if (viewerTitle) viewerTitle.textContent = `Latido #${pulso.id} (${pulso.tiempo_str})`;
        if (viewerBadgeSeverity) {
            viewerBadgeSeverity.className = `lead-badge ${pulso.badge_class}`;
            viewerBadgeSeverity.textContent = `${pulso.clase} - ${pulso.nombre_clinico}`;
        }
        if (viewerPulseCounter) viewerPulseCounter.textContent = `${index + 1} / ${total}`;

        // Destacar fila seleccionada en tabla
        const filas = tbodyCsvPulses.querySelectorAll('tr');
        filas.forEach(f => {
            f.classList.toggle('table-row-selected', parseInt(f.dataset.index) === index);
        });

        // 1. Gráfica Reciclable 1: Corte del Latido 1D (270 muestras)
        dibujarCorte1D(pulso.corte_1d, pulso.clase);

        // 2. Gráfica Reciclable 2: Transformada STFT 2D (Espectrograma 29x112)
        dibujarEspectrogramaSTFT(pulso.espectrograma_2d);

        // 3. Gráfica 3: Barras de probabilidad CNN
        dibujarBarrasProbabilidad(pulso.distribucion_probas, pulso.clase);
    }

    // Gráfica 1: Corte 1D en Canvas
    function dibujarCorte1D(puntos270, clase) {
        if (!canvasPulse1D || !puntos270 || puntos270.length === 0) return;
        const ctx = canvasPulse1D.getContext('2d');
        const width = canvasPulse1D.width;
        const height = canvasPulse1D.height;
        const midY = height / 2;

        ctx.fillStyle = '#060B12';
        ctx.fillRect(0, 0, width, height);

        // Rejilla milimétrica médica
        ctx.strokeStyle = 'rgba(16, 185, 129, 0.08)';
        ctx.lineWidth = 1;
        for (let x = 0; x < width; x += 20) {
            ctx.beginPath();
            ctx.moveTo(x, 0);
            ctx.lineTo(x, height);
            ctx.stroke();
        }
        for (let y = 0; y < height; y += 20) {
            ctx.beginPath();
            ctx.moveTo(0, y);
            ctx.lineTo(width, y);
            ctx.stroke();
        }

        // Línea base isoelectrica (0V)
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.15)';
        ctx.setLineDash([4, 4]);
        ctx.beginPath();
        ctx.moveTo(0, midY);
        ctx.lineTo(width, midY);
        ctx.stroke();
        ctx.setLineDash([]);

        // Marcador del Pico R en muestra 94 (35% de 270)
        const xPicoR = (94 / 270) * width;
        ctx.strokeStyle = 'rgba(239, 68, 68, 0.7)';
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.moveTo(xPicoR, 0);
        ctx.lineTo(xPicoR, height);
        ctx.stroke();

        ctx.fillStyle = '#EF4444';
        ctx.font = 'bold 9px monospace';
        ctx.fillText('Pico R', xPicoR + 4, 12);

        // Dibujar forma de onda
        const esArritmia = clase !== 'N';
        ctx.strokeStyle = esArritmia ? '#EF4444' : '#10B981';
        ctx.lineWidth = 2.2;
        ctx.shadowColor = esArritmia ? '#EF4444' : '#10B981';
        ctx.shadowBlur = 6;
        ctx.beginPath();

        const stepX = width / puntos270.length;
        const maxAmp = Math.max(...puntos270.map(Math.abs), 0.5);
        const scaleY = (height * 0.42) / maxAmp;

        for (let i = 0; i < puntos270.length; i++) {
            const x = i * stepX;
            const y = midY - (puntos270[i] * scaleY);
            if (i === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
        }
        ctx.stroke();
        ctx.shadowBlur = 0;
    }

    // Gráfica 2: Espectrograma STFT 2D en Canvas con Paleta Plasma/Viridis
    function dibujarEspectrogramaSTFT(matriz29x112) {
        if (!canvasPulseSTFT || !matriz29x112 || matriz29x112.length === 0) return;
        const ctx = canvasPulseSTFT.getContext('2d');
        const width = canvasPulseSTFT.width;
        const height = canvasPulseSTFT.height;

        const filas = matriz29x112.length;    // 29 frecuencias
        const cols = matriz29x112[0].length;  // 112 bins temporales

        const cellW = width / cols;
        const cellH = height / filas;

        ctx.fillStyle = '#060B12';
        ctx.fillRect(0, 0, width, height);

        // Paleta de color espectral médica: Azul marino -> Cyan -> Amarillo -> Blanco
        for (let f = 0; f < filas; f++) {
            // f=0 es frecuencia baja (dibujamos abajo para escala cartesiana convencional)
            const y = height - (f + 1) * cellH;
            for (let t = 0; t < cols; t++) {
                const val = matriz29x112[f][t]; // Normalizado [0, 1]
                const x = t * cellW;

                ctx.fillStyle = interpolarColorEspectral(val);
                ctx.fillRect(x, y, cellW + 0.5, cellH + 0.5);
            }
        }
    }

    function interpolarColorEspectral(val) {
        // Mapa térmico para espectrograma médico
        const v = Math.max(0, Math.min(1, val));
        let r, g, b;

        if (v < 0.25) {
            const f = v / 0.25;
            r = Math.round(15 + f * 10);
            g = Math.round(23 + f * 50);
            b = Math.round(60 + f * 120);
        } else if (v < 0.5) {
            const f = (v - 0.25) / 0.25;
            r = Math.round(25 + f * 10);
            g = Math.round(73 + f * 120);
            b = Math.round(180 + f * 30);
        } else if (v < 0.75) {
            const f = (v - 0.5) / 0.25;
            r = Math.round(35 + f * 210);
            g = Math.round(193 + f * 30);
            b = Math.round(210 - f * 170);
        } else {
            const f = (v - 0.75) / 0.25;
            r = 245 + Math.round(f * 10);
            g = 223 + Math.round(f * 32);
            b = 40 + Math.round(f * 215);
        }

        return `rgb(${r}, ${g}, ${b})`;
    }

    // Gráfica 3: Barras de Probabilidades CNN
    function dibujarBarrasProbabilidad(probasDict, clasePredicha) {
        if (!pulseProbBars || !probasDict) return;
        pulseProbBars.innerHTML = '';

        const ordenClases = ['N', 'V', 'A', 'R', 'L', '/', 'f', 'F', 'No_Clasificable'];

        ordenClases.forEach(c => {
            const p = probasDict[c] || 0.0;
            const esGanadora = c === clasePredicha;

            let barColor = 'rgba(255, 255, 255, 0.2)';
            if (esGanadora) {
                if (c === 'N') barColor = '#10B981';
                else if (c === 'V' || c === 'F') barColor = '#EF4444';
                else if (c === 'A' || c === 'R' || c === 'L') barColor = '#F59E0B';
                else barColor = '#0EA5E9';
            }

            const row = document.createElement('div');
            row.className = 'prob-bar-row';
            row.innerHTML = `
                <span class="prob-bar-label ${esGanadora ? 'text-primary' : ''}">${c}</span>
                <div class="prob-bar-track">
                    <div class="prob-bar-fill" style="width: ${p}%; background: ${barColor};"></div>
                </div>
                <span class="prob-bar-val ${esGanadora ? 'text-primary' : ''}">${p.toFixed(1)}%</span>
            `;
            pulseProbBars.appendChild(row);
        });
    }

    // Navegación de pulsos Anterior / Siguiente
    if (btnPrevPulse) {
        btnPrevPulse.addEventListener('click', () => {
            if (csvAnalysisResult && currentPulseIndex > 0) {
                seleccionarPulso(currentPulseIndex - 1);
            }
        });
    }

    if (btnNextPulse) {
        btnNextPulse.addEventListener('click', () => {
            if (csvAnalysisResult && currentPulseIndex < csvAnalysisResult.pulsos.length - 1) {
                seleccionarPulso(currentPulseIndex + 1);
            }
        });
    }

    // Exportar Reporte CSV
    if (btnExportCsvReport) {
        btnExportCsvReport.addEventListener('click', () => {
            window.location.href = '/api/analyze/export-csv';
            mostrarToast('Descargando reporte pulso por pulso en CSV...', 'info');
        });
    }

    // =========================================================================
    // RESTO DE FUNCIONALIDADES: INFERENCIA CRUDO, .NPZ, VT, INSPECTOR, PIPELINE
    // =========================================================================

    // Monitor ECG animado (Pestaña 2: MIT-BIH)
    function generarECGSimulado() {
        ecgPoints = [];
        const cicloNormal = [
            0.0, 0.02, 0.05, 0.10, 0.06, 0.01, 0.0, -0.04,
            0.88, -0.22, 0.0, 0.04, 0.12, 0.20, 0.16, 0.06, 0.01, 0.0, 0.0, 0.0, 0.0
        ];
        for (let i = 0; i < 35; i++) {
            cicloNormal.forEach(p => ecgPoints.push(p + (Math.random() - 0.5) * 0.015));
            for (let j = 0; j < 14; j++) ecgPoints.push((Math.random() - 0.5) * 0.008);
        }
    }

    function animarECG() {
        if (!ecgCanvas || !ecgCtx) return;
        const width = ecgCanvas.width;
        const height = ecgCanvas.height;
        const midY = height / 2;

        if (!isEcgPaused) {
            ecgCtx.fillStyle = '#060B12';
            ecgCtx.fillRect(0, 0, width, height);

            ecgCtx.strokeStyle = 'rgba(16, 185, 129, 0.08)';
            ecgCtx.lineWidth = 1;
            for (let x = 0; x < width; x += 25) {
                ecgCtx.beginPath();
                ecgCtx.moveTo(x, 0);
                ecgCtx.lineTo(x, height);
                ecgCtx.stroke();
            }
            for (let y = 0; y < height; y += 25) {
                ecgCtx.beginPath();
                ecgCtx.moveTo(0, y);
                ecgCtx.lineTo(width, y);
                ecgCtx.stroke();
            }

            ecgCtx.strokeStyle = '#10B981';
            ecgCtx.lineWidth = 2.0;
            ecgCtx.shadowColor = '#10B981';
            ecgCtx.shadowBlur = 6;
            ecgCtx.beginPath();

            const step = 2.2;
            const totalVisible = Math.floor(width / step);

            for (let i = 0; i < totalVisible; i++) {
                const dataIndex = (ecgOffset + i) % ecgPoints.length;
                const val = ecgPoints[dataIndex] || 0;
                const x = i * step;
                const y = midY - (val * 85);

                if (i === 0) ecgCtx.moveTo(x, y);
                else ecgCtx.lineTo(x, y);
            }
            ecgCtx.stroke();
            ecgCtx.shadowBlur = 0;

            ecgOffset = (ecgOffset + 1) % ecgPoints.length;
        }

        ecgAnimationId = requestAnimationFrame(animarECG);
    }

    if (btnToggleEcg) {
        btnToggleEcg.addEventListener('click', () => {
            isEcgPaused = !isEcgPaused;
            const textEl = btnToggleEcg.querySelector('.btn-text');
            if (textEl) textEl.textContent = isEcgPaused ? 'Reanudar' : 'Pausar';
        });
    }

    // ACCIÓN: Evaluar Paciente Crudo MIT-BIH
    if (btnRunRaw) {
        btnRunRaw.addEventListener('click', async () => {
            const record = selectPatientRaw ? selectPatientRaw.value : '100';
            const model = selectModelRaw ? selectModelRaw.value : '6.0';

            setLoading(btnRunRaw, true, 'Procesando señal ECG...');
            try {
                const res = await fetch('/api/evaluate/raw', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ record, model })
                });
                const data = await res.json();
                if (data.error) throw new Error(data.error);

                const resultsCont = document.getElementById('raw-results-container');
                if (resultsCont) resultsCont.style.display = 'grid';

                const badgeAcc = document.getElementById('raw-accuracy-badge');
                if (badgeAcc) badgeAcc.textContent = `Exactitud Global: ${data.accuracy}%`;

                const totalBeatsEl = document.getElementById('raw-total-beats');
                if (totalBeatsEl) totalBeatsEl.textContent = `Latidos Analizados: ${data.total_beats}`;

                const sampleInfo = document.getElementById('ecg-sample-info');
                if (sampleInfo) sampleInfo.textContent = `Paciente: ${data.record} • ${data.fs} Hz • Latidos: ${data.total_beats} • Modelo: ${model}`;

                if (data.ecg_preview && data.ecg_preview.length > 0) {
                    ecgPoints = data.ecg_preview;
                    ecgOffset = 0;
                }

                renderizarMatriz('raw-matrix-wrapper', data.matrix, data.labels);
                renderizarTablaMetricas('tbody-raw-metrics', data.report, data.labels);

                mostrarToast(`Diagnóstico del Paciente ${record} completado: Exactitud ${data.accuracy}%`, 'success');
            } catch (err) {
                mostrarToast(err.message, 'error');
            } finally {
                setLoading(btnRunRaw, false, 'Ejecutar Diagnóstico End-to-End');
            }
        });
    }

    // ACCIÓN: Evaluar .NPZ
    if (btnRunNpz) {
        btnRunNpz.addEventListener('click', async () => {
            const file = selectNpzFile ? selectNpzFile.value : 'Data/Raw_data/100.npz';
            const model = selectModelNpz ? selectModelNpz.value : '6.0';

            setLoading(btnRunNpz, true, 'Evaluando tensores .npz...');
            try {
                const res = await fetch('/api/evaluate/npz', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ file, model })
                });
                const data = await res.json();
                if (data.error) throw new Error(data.error);

                const resultsCont = document.getElementById('npz-results-container');
                if (resultsCont) resultsCont.style.display = 'grid';

                const badgeAcc = document.getElementById('npz-accuracy-badge');
                if (badgeAcc) badgeAcc.textContent = `Exactitud: ${data.accuracy}%`;

                const totalBeatsEl = document.getElementById('npz-total-beats');
                if (totalBeatsEl) totalBeatsEl.textContent = `Latidos: ${data.total_beats}`;

                renderizarMatriz('npz-matrix-wrapper', data.matrix, data.labels);
                renderizarTablaMetricas('tbody-npz-metrics', data.report, data.labels);

                mostrarToast(`Evaluación de ${data.file} finalizada: Exactitud ${data.accuracy}%`, 'success');
            } catch (err) {
                mostrarToast(err.message, 'error');
            } finally {
                setLoading(btnRunNpz, false, 'Evaluar Tensor .npz');
            }
        });
    }

    // ACCIÓN: Taquicardia Ventricular
    if (btnScanVt) {
        btnScanVt.addEventListener('click', async () => {
            setLoading(btnScanVt, true, 'Analizando 48 registros MIT-BIH...');
            const tbody = document.getElementById('tbody-vt-results');
            if (tbody) tbody.innerHTML = '<tr><td colspan="5" class="text-center">Analizando episodios ventriculares en la base de datos...</td></tr>';

            try {
                const res = await fetch('/api/vt-search', { method: 'POST' });
                const data = await res.json();
                if (data.error) throw new Error(data.error);

                const summaryBar = document.getElementById('vt-summary-bar');
                if (summaryBar) summaryBar.style.display = 'flex';

                const countBadge = document.getElementById('vt-count-badge');
                if (countBadge) countBadge.textContent = `Total Episodios Detectados: ${data.total}`;

                if (tbody) {
                    tbody.innerHTML = '';
                    if (!data.events || data.events.length === 0) {
                        tbody.innerHTML = '<tr><td colspan="5" class="text-center text-muted">No se detectaron episodios de Taquicardia Ventricular.</td></tr>';
                    } else {
                        data.events.forEach(ev => {
                            let badgeClass = 'badge-warning';
                            let severidad = 'Moderada';

                            if (ev.tipo_deteccion.includes('Ráfaga') || ev.tipo_deteccion.includes('Rafaga')) {
                                const match = ev.tipo_deteccion.match(/\d+/);
                                const cant = match ? parseInt(match[0]) : 3;
                                if (cant >= 10) { badgeClass = 'badge-danger'; severidad = `Crítica (${cant} latidos)`; }
                                else if (cant >= 5) { badgeClass = 'badge-warning'; severidad = `Alta (${cant} latidos)`; }
                                else { badgeClass = 'badge-info'; severidad = `Moderada (${cant} latidos)`; }
                            } else {
                                badgeClass = 'badge-danger';
                                severidad = 'Ritmo VT Anotado';
                            }

                            const tr = document.createElement('tr');
                            tr.innerHTML = `
                                <td><strong>Paciente ${ev.registro}</strong></td>
                                <td><code>${ev.tiempo}</code></td>
                                <td>${ev.muestra}</td>
                                <td>${ev.tipo_deteccion}</td>
                                <td><span class="badge ${badgeClass}">${severidad}</span></td>
                            `;
                            tbody.appendChild(tr);
                        });
                    }
                }

                mostrarToast(`Escaneo de Taquicardia Ventricular finalizado: ${data.total} episodios localizados`, 'success');
            } catch (err) {
                mostrarToast(err.message, 'error');
            } finally {
                setLoading(btnScanVt, false, 'Iniciar Escaneo Global en MIT-BIH');
            }
        });
    }

    // ACCIÓN: Inspeccionar Paciente
    if (btnRunInspect) {
        btnRunInspect.addEventListener('click', async () => {
            const record = selectInspectPatient ? selectInspectPatient.value : '100';
            setLoading(btnRunInspect, true, 'Auditando registro...');

            try {
                const res = await fetch('/api/inspect', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ record })
                });
                const data = await res.json();
                if (data.error) throw new Error(data.error);

                const inspectCont = document.getElementById('inspect-container');
                if (inspectCont) inspectCont.style.display = 'grid';

                const infoList = document.getElementById('inspect-tensor-info');
                if (infoList && data.info_npz) {
                    infoList.innerHTML = `
                        <div class="info-row"><span class="label">Registro:</span><span class="val">${data.record}</span></div>
                        <div class="info-row"><span class="label">Dimensiones Tensor X:</span><span class="val">${JSON.stringify(data.info_npz.shape_X || 'N/A')}</span></div>
                        <div class="info-row"><span class="label">Tipo de Dato (dtype):</span><span class="val">${data.info_npz.dtype_X || 'N/A'}</span></div>
                        <div class="info-row"><span class="label">Rango Normalizado:</span><span class="val">[${data.info_npz.min_X || 0} a ${data.info_npz.max_X || 1}]</span></div>
                        <div class="info-row"><span class="label">Latidos Guardados en .npz:</span><span class="val">${data.info_npz.total_latidos || 0}</span></div>
                    `;
                }

                const tbody = document.getElementById('tbody-inspect-comp');
                if (tbody && data.comparativa) {
                    tbody.innerHTML = '';
                    data.comparativa.forEach(row => {
                        const diff = row.alg - row.db;
                        const diffColor = diff === 0 ? 'text-success' : 'text-warning';
                        const tr = document.createElement('tr');
                        tr.innerHTML = `
                            <td><strong>${row.label}</strong> <span class="text-muted small">(${NOMBRES_CLINICOS[row.label] || 'Otros'})</span></td>
                            <td>${row.alg}</td>
                            <td>${row.db}</td>
                            <td class="${diffColor}"><strong>${diff > 0 ? '+' + diff : diff}</strong></td>
                        `;
                        tbody.appendChild(tr);
                    });
                }

                mostrarToast(`Auditoría del Paciente ${record} completada exitosamente`, 'info');
            } catch (err) {
                mostrarToast(err.message, 'error');
            } finally {
                setLoading(btnRunInspect, false, 'Inspeccionar & Comparar vs .atr Oficial');
            }
        });
    }

    // ACCIONES: Pipeline de Datos
    if (btnPreprocessSingle) {
        btnPreprocessSingle.addEventListener('click', async () => {
            const inputRec = document.getElementById('input-preprocess-rec');
            const rec = inputRec ? inputRec.value.trim() : '100';
            setLoading(btnPreprocessSingle, true, 'Procesando...');
            try {
                const res = await fetch('/api/preprocess', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ record: rec })
                });
                const data = await res.json();
                if (data.error) throw new Error(data.error);
                mostrarToast(data.message, 'success');
                cargarInfoInicial();
            } catch (e) { mostrarToast(e.message, 'error'); }
            finally { setLoading(btnPreprocessSingle, false, 'Procesar Registro'); }
        });
    }

    if (btnPreprocessAll) {
        btnPreprocessAll.addEventListener('click', async () => {
            if (!confirm('¿Deseas iniciar el procesamiento masivo de la base MIT-BIH (100 al 234)?')) return;
            setLoading(btnPreprocessAll, true, 'Procesando 48 registros...');
            try {
                const res = await fetch('/api/preprocess', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ record: 'all' })
                });
                const data = await res.json();
                if (data.error) throw new Error(data.error);
                mostrarToast(data.message, 'success');
                cargarInfoInicial();
            } catch (e) { mostrarToast(e.message, 'error'); }
            finally { setLoading(btnPreprocessAll, false, 'Procesar Base Completa (100-234)'); }
        });
    }

    if (btnUnifyDataset) {
        btnUnifyDataset.addEventListener('click', async () => {
            setLoading(btnUnifyDataset, true, 'Concatenando tensores...');
            try {
                const res = await fetch('/api/dataset/unify', { method: 'POST' });
                const data = await res.json();
                if (data.error) throw new Error(data.error);
                mostrarToast(data.message, 'success');
                cargarInfoInicial();
            } catch (e) { mostrarToast(e.message, 'error'); }
            finally { setLoading(btnUnifyDataset, false, 'Unificar Tensores (.npz)'); }
        });
    }

    if (btnFilterDataset) {
        btnFilterDataset.addEventListener('click', async () => {
            setLoading(btnFilterDataset, true, 'Filtrando clases AAMI...');
            try {
                const res = await fetch('/api/dataset/filter', { method: 'POST' });
                const data = await res.json();
                if (data.error) throw new Error(data.error);
                mostrarToast(data.message, 'success');
                cargarInfoInicial();
            } catch (e) { mostrarToast(e.message, 'error'); }
            finally { setLoading(btnFilterDataset, false, 'Generar Dataset Limpio AAMI'); }
        });
    }

    // Auxiliares de matriz y tablas
    function renderizarMatriz(containerId, matrix, labels) {
        const container = document.getElementById(containerId);
        if (!container || !matrix) return;

        let maxVal = 1;
        for (let i = 0; i < matrix.length; i++) {
            for (let j = 0; j < matrix[i].length; j++) {
                if (matrix[i][j] > maxVal) maxVal = matrix[i][j];
            }
        }

        let html = '<table class="matrix-table"><thead><tr><th>Real \\ Pred</th>';
        labels.forEach(l => { html += `<th>${l}</th>`; });
        html += '</tr></thead><tbody>';

        for (let i = 0; i < matrix.length; i++) {
            html += `<tr><th>${labels[i] || i}</th>`;
            for (let j = 0; j < matrix[i].length; j++) {
                const val = matrix[i][j];
                const intensity = val > 0 ? Math.min(1, Math.max(0.12, Math.pow(val / maxVal, 0.4))) : 0;
                let bgColor = 'rgba(255, 255, 255, 0.02)';
                let textColor = 'rgba(255, 255, 255, 0.25)';

                if (val > 0) {
                    if (i === j) {
                        bgColor = `rgba(14, 165, 233, ${intensity * 0.85})`;
                        textColor = '#ffffff';
                    } else {
                        bgColor = `rgba(239, 68, 68, ${intensity * 0.75})`;
                        textColor = '#ffffff';
                    }
                }

                html += `<td class="matrix-cell" style="background-color: ${bgColor}; color: ${textColor};" title="Real: ${labels[i]} | Inferencia: ${labels[j]} = ${val} latidos">${val}</td>`;
            }
            html += '</tr>';
        }

        html += '</tbody></table>';
        container.innerHTML = html;
    }

    function renderizarTablaMetricas(tbodyId, report, labels) {
        const tbody = document.getElementById(tbodyId);
        if (!tbody || !report) return;

        tbody.innerHTML = '';
        labels.forEach(label => {
            const metric = report[label];
            if (!metric) return;

            const prec = (metric.precision * 100).toFixed(1);
            const rec = (metric.recall * 100).toFixed(1);
            const f1 = (metric['f1-score'] * 100).toFixed(1);
            const sup = metric.support;

            const tr = document.createElement('tr');
            tr.innerHTML = `
                <td><strong>${label}</strong></td>
                <td>${NOMBRES_CLINICOS[label] || label}</td>
                <td><span class="badge ${rec >= 85 ? 'badge-success' : (rec >= 60 ? 'badge-warning' : 'badge-danger')}">${rec}%</span></td>
                <td><span class="badge ${prec >= 85 ? 'badge-success' : (prec >= 60 ? 'badge-warning' : 'badge-danger')}">${prec}%</span></td>
                <td><strong>${f1}%</strong></td>
                <td><code>${sup}</code></td>
            `;
            tbody.appendChild(tr);
        });
    }

    // UI Helper defensivo
    function setLoading(btn, isLoading, loadingText) {
        if (!btn) return;
        btn.disabled = isLoading;
        btn.classList.toggle('btn-loading', isLoading);

        const textEl = btn.querySelector('.btn-text');
        const spinnerEl = btn.querySelector('.btn-spinner');

        if (isLoading) {
            if (!btn.dataset.originalText && textEl) {
                btn.dataset.originalText = textEl.textContent;
            }
            if (textEl && loadingText) textEl.textContent = loadingText;
            else if (!textEl && loadingText) btn.textContent = loadingText;

            if (spinnerEl) spinnerEl.style.display = 'inline-flex';
        } else {
            if (textEl && (loadingText || btn.dataset.originalText)) {
                textEl.textContent = loadingText || btn.dataset.originalText;
            } else if (!textEl && (loadingText || btn.dataset.originalText)) {
                btn.textContent = loadingText || btn.dataset.originalText;
            }

            if (spinnerEl) spinnerEl.style.display = 'none';
        }
    }

    function mostrarToast(mensaje, tipo = 'info') {
        const container = document.getElementById('toast-container');
        if (!container) return;

        const toast = document.createElement('div');
        toast.className = `toast ${tipo}`;

        let iconSvg = '';
        if (tipo === 'success') {
            iconSvg = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="20 6 9 17 4 12"></polyline></svg>`;
        } else if (tipo === 'error') {
            iconSvg = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"></circle><line x1="15" y1="9" x2="9" y2="15"></line><line x1="9" y1="9" x2="15" y2="15"></line></svg>`;
        } else if (tipo === 'warning') {
            iconSvg = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"></path><line x1="12" y1="9" x2="12" y2="13"></line><line x1="12" y1="17" x2="12.01" y2="17"></line></svg>`;
        } else {
            iconSvg = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="16" x2="12" y2="12"></line><line x1="12" y1="8" x2="12.01" y2="8"></line></svg>`;
        }

        toast.innerHTML = `<span class="toast-icon">${iconSvg}</span><span class="toast-msg">${mensaje}</span>`;
        container.appendChild(toast);

        setTimeout(() => {
            toast.style.opacity = '0';
            toast.style.transform = 'translateY(10px)';
            setTimeout(() => toast.remove(), 350);
        }, 4500);
    }

    // Inicializar aplicación
    cargarInfoInicial();
});

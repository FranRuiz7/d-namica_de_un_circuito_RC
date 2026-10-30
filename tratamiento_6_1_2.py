import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import OptimizeWarning, curve_fit

# =============================================================================
# CONFIGURACIÓN DE ARCHIVOS Y PARÁMETROS EXPERIMENTALES
# =============================================================================
ARCHIVOS = {
    33: r"C:\Users\franr\Documents\Facultad\Experimental 2\Informes\Lab 2 - Capacitores\tercera_vuelta\datos_33ohm.csv",
    100: r"C:\Users\franr\Documents\Facultad\Experimental 2\Informes\Lab 2 - Capacitores\tercera_vuelta\datos_100ohm.csv",
    220: r"C:\Users\franr\Documents\Facultad\Experimental 2\Informes\Lab 2 - Capacitores\tercera_vuelta\datos_220ohm.csv",
}

C_NOMINAL_UF = 330.0  # Capacitancia nominal en uF
N_CICLOS = 5  # 5 ciclos completos = 5 cargas + 5 descargas por R

# Ventanas temporales para estimar Vi, Vinf y rango de ajuste (en segundos)
T_PRE_VI_INI = (
    0.025  # Inicio de ventana para Vi (25 ms antes de t0)
)
T_PRE_VI_FIN = (
    0.001  # Fin de ventana para Vi (1 ms antes de t0, evita la transición)
)
T_FIN_VINF_INI = (
    0.045  # Inicio de ventana para Vinf (45 ms antes del siguiente flanco)
)
T_FIN_VINF_FIN = (
    0.002  # Fin de ventana para Vinf (2 ms antes del siguiente flanco)
)
T_MUERTO_INICIO = (
    0.0004  # Salto tras t0 (0.4 ms = 2 muestras a 5 kHz) para evitar transición del generador
)


# =============================================================================
# 1. FUNCIÓN DE LECTURA ROBUSTA PARA CSV DE PASCO CAPSTONE
# =============================================================================
def cargar_csv_pasco(ruta_archivo):
  """Lee un CSV exportado por PASCO Capstone (con coma decimal) y estandariza

  las tres columnas: 'tiempo', 'vs' (Ch A: generador) y 'vc' (Ch C: capacitor).
  """
  # Detectar separador automáticamente (; o , o tabulador)
  with open(ruta_archivo, 'r', encoding='utf-8', errors='ignore') as f:
    primera_linea = f.readline()
  sep = (
      ';'
      if ';' in primera_linea
      else ('\t' if '\t' in primera_linea else ',')
  )

  df = pd.read_csv(ruta_archivo, sep=sep, decimal=',')

  # Quedarse con las primeras 3 columnas relevantes y renombrarlas
  df = df.iloc[:, :3].copy()
  df.columns = ['tiempo', 'ch_a', 'ch_c']

  # Asegurar conversión a float por si alguna columna quedó como texto
  for col in ['tiempo', 'ch_a', 'ch_c']:
    if df[col].dtype == object:
      df[col] = (
          df[col]
          .astype(str)
          .str.replace(',', '.', regex=False)
          .str.strip()
      )
      df[col] = pd.to_numeric(df[col], errors='coerce')

  df = df.dropna().reset_index(drop=True)

  # Verificación automática de qué canal es el generador (Vs) y cuál el capacitor (Vc):
  # El generador (onda cuadrada) tiene derivadas máximas mucho más abruptas que el capacitor.
  max_diff_a = np.max(np.abs(np.diff(df['ch_a'])))
  max_diff_c = np.max(np.abs(np.diff(df['ch_c'])))

  if max_diff_a >= max_diff_c:
    df['vs'] = df['ch_a']
    df['vc'] = df['ch_c']
  else:
    df['vs'] = df['ch_c']
    df['vc'] = df['ch_a']

  return df


# =============================================================================
# 2. DETECCIÓN DE FLANCOS Y SELECCIÓN DE 5 CICLOS COMPLETOS
# =============================================================================
def detectar_ciclos_completos(df, n_ciclos=5):
  """Detecta los flancos ascendentes y descendentes de Vs(t) y selecciona

  exactamente n_ciclos completos consecutivos (comenzando en una Carga) que
  posean margen previo suficiente para calcular Vi y un flanco final que cierre
  el último semiperíodo.
  """
  t = df['tiempo'].values
  vs = df['vs'].values

  # Umbral a mitad de excursión de la onda cuadrada (aprox. 2.0 V)
  umbral = 0.5 * (np.max(vs) + np.min(vs))

  # Índices donde Vs cruza el umbral
  cruces_subida = (
      np.where((vs[:-1] < umbral) & (vs[1:] >= umbral))[0] + 1
  )
  cruces_bajada = (
      np.where((vs[:-1] >= umbral) & (vs[1:] < umbral))[0] + 1
  )

  # Unificar todos los flancos ordenados cronológicamente
  flancos = []
  for idx in cruces_subida:
    flancos.append((idx, 'Carga'))
  for idx in cruces_bajada:
    flancos.append((idx, 'Descarga'))
  flancos.sort(key=lambda x: x[0])

  # Filtrar flancos demasiado cercanos al inicio (necesitamos al menos T_PRE_VI_INI antes de t0)
  flancos_validos = [
      f for f in flancos if t[f[0]] >= (t[0] + T_PRE_VI_INI + 0.005)
  ]

  # Buscar el primer flanco de 'Carga' para que el Ciclo 1 empiece en Carga
  idx_inicio = None
  for i, (_, tipo) in enumerate(flancos_validos):
    if tipo == 'Carga':
      idx_inicio = i
      break

  if idx_inicio is None or (idx_inicio + 2 * n_ciclos) >= len(
      flancos_validos
  ):
    raise ValueError(
        'No se encontraron suficientes ciclos completos en el archivo.'
        ' Verifique que haya al menos 5 ciclos enteros.'
    )

  # Seleccionamos 2*n_ciclos flancos + 1 flanco extra de cierre
  seleccion = flancos_validos[
      idx_inicio : idx_inicio + 2 * n_ciclos + 1
  ]
  return seleccion


# =============================================================================
# 3. AJUSTE EXPONENCIAL DE UN ÚNICO PARÁMETRO (Ecuación 9)
# =============================================================================
def ajustar_flanco_unico_parametro(
    t, vc, idx_ini, idx_sig, R_nom
):
  """Determina t0, Vi y Vinf experimentalmente y ajusta únicamente tau."""
  t0 = t[idx_ini]
  t_fin_semiperiodo = t[idx_sig]

  # 1. Estimación de Vi (promedio inmediatamente antes de t0)
  mask_vi = (t >= (t0 - T_PRE_VI_INI)) & (
      t <= (t0 - T_PRE_VI_FIN)
  )
  Vi = np.mean(vc[mask_vi])

  # 2. Estimación de Vinf (promedio hacia el final del semiperíodo)
  mask_vinf = (t >= (t_fin_semiperiodo - T_FIN_VINF_INI)) & (
      t <= (t_fin_semiperiodo - T_FIN_VINF_FIN)
  )
  Vinf = np.mean(vc[mask_vinf])

  # 3. Selección del intervalo de ajuste del transitorio
  mask_fit = (t >= (t0 + T_MUERTO_INICIO)) & (
      t <= (t_fin_semiperiodo - T_FIN_VINF_FIN)
  )
  t_fit = t[mask_fit]
  vc_fit = vc[mask_fit]

  # Modelo exponencial con t0, Vi y Vinf fijos (Ecuación 9 de la guía)
  def modelo_vc(t_val, tau):
    return Vinf + (Vi - Vinf) * np.exp(-(t_val - t0) / tau)

  # Semilla inicial basada en el valor nominal RC
  tau_semilla = (R_nom * C_NOMINAL_UF) * 1e-6
  advertencia = 'OK'

  with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter('always', OptimizeWarning)
    try:
      popt, pcov = curve_fit(
          modelo_vc,
          t_fit,
          vc_fit,
          p0=[tau_semilla],
          bounds=(1e-5, 1.0),
      )
      tau_opt = popt[0]
      u_tau = np.sqrt(pcov[0, 0])
      if len(w) > 0:
        advertencia = str(w[-1].message)
    except Exception as e:
      tau_opt = np.nan
      u_tau = np.nan
      advertencia = f'Fallo: {str(e)}'

  # Cálculo de curva ajustada y residuos r_i = V_{C,i} - V_{C,modelo}(t_i)
  vc_modelo = modelo_vc(t_fit, tau_opt)
  residuos = vc_fit - vc_modelo
  rmse_mv = np.sqrt(np.mean(residuos**2)) * 1000.0  # en mV

  # R^2 (solo como referencia complementaria; el diagnóstico principal son los residuos)
  ss_res = np.sum(residuos**2)
  ss_tot = np.sum((vc_fit - np.mean(vc_fit)) ** 2)
  r2 = 1.0 - (ss_res / ss_tot)

  return {
      't0': t0,
      'Vi': Vi,
      'Vinf': Vinf,
      'tau_s': tau_opt,
      'u_tau_s': u_tau,
      'tau_ms': tau_opt * 1000.0,
      'u_tau_ms': u_tau * 1000.0,
      'rmse_mv': rmse_mv,
      'r2': r2,
      'advertencia': advertencia,
      't_local_ms': (t_fit - t0) * 1000.0,
      'vc_fit': vc_fit,
      'vc_modelo': vc_modelo,
      'residuos_mv': residuos * 1000.0,
  }


# =============================================================================
# 4. PROCESAMIENTO PRINCIPAL Y GRÁFICAS (ESTILO FIGURA 3 + RESIDUOS)
# =============================================================================
lista_resultados = []

for R_nom, ruta in ARCHIVOS.items():
  print(
      f'\nProcesando R_nominal = {R_nom} Ohm ({ruta})...'
  )
  df = cargar_csv_pasco(ruta)
  flancos = detectar_ciclos_completos(df, n_ciclos=N_CICLOS)

  t = df['tiempo'].values
  vs = df['vs'].values
  vc = df['vc'].values

  detalles_carga_ciclo1 = None
  detalles_descarga_ciclo1 = None

  for k in range(2 * N_CICLOS):
    idx_ini, tipo = flancos[k]
    idx_sig, _ = flancos[k + 1]
    ciclo_actual = (k // 2) + 1

    res = ajustar_flanco_unico_parametro(
        t, vc, idx_ini, idx_sig, R_nom
    )

    # Guardar el Ciclo 1 para los paneles detallados de la gráfica
    if ciclo_actual == 1 and tipo == 'Carga':
      detalles_carga_ciclo1 = res
    elif ciclo_actual == 1 and tipo == 'Descarga':
      detalles_descarga_ciclo1 = res

    lista_resultados.append({
        'R_nom_ohm': R_nom,
        'Ciclo': ciclo_actual,
        'Proceso': tipo,
        't0_s': round(res['t0'], 5),
        'Vi_V': round(res['Vi'], 4),
        'Vinf_V': round(res['Vinf'], 4),
        'tau_ms': round(res['tau_ms'], 4),
        'u_tau_ms': round(res['u_tau_ms'], 5),
        'RMSE_residuos_mV': round(res['rmse_mv'], 3),
        'R2': round(res['r2'], 6),
        'Convergencia': res['advertencia'],
    })

  # -------------------------------------------------------------------------
  # CREACIÓN DE LA FIGURA DIAGNÓSTICA PARA CADA RESISTENCIA
  # -------------------------------------------------------------------------
  fig = plt.figure(figsize=(13, 9))
  gs = fig.add_gridspec(3, 2, height_ratios=[1.2, 1.3, 0.8], hspace=0.42, wspace=0.22)

  # Panel superior: Los 5 ciclos completos seleccionados
  ax_top = fig.add_subplot(gs[0, :])
  t_ini_5c = t[flancos[0][0]] - 0.05
  t_fin_5c = t[flancos[-1][0]] + 0.05
  mask_5c = (t >= t_ini_5c) & (t <= t_fin_5c)

  ax_top.plot(
      t[mask_5c],
      vs[mask_5c],
      '--',
      color='gray',
      alpha=0.7,
      label='Generador $V_s(t)$',
  )
  ax_top.plot(
      t[mask_5c],
      vc[mask_5c],
      color='#1f77b4',
      lw=1.5,
      label='Capacitor $V_C(t)$',
  )
  ax_top.axvspan(
      t[flancos[0][0]],
      t[flancos[2][0]],
      color='gold',
      alpha=0.2,
      label='Ciclo 1 (detallado abajo)',
  )
  ax_top.set_title(
      f'1. Registro de 5 Ciclos Estacionarios ($R_{{nom}} = {R_nom}\\,\\Omega$)',
      fontsize=11,
      fontweight='bold',
  )
  ax_top.set_xlabel('Tiempo global $t$ (s)')
  ax_top.set_ylabel('Tensión (V)')
  ax_top.legend(loc='upper right', fontsize=9)
  ax_top.grid(True, alpha=0.3)

  # Paneles centrales e inferiores: Carga (izquierda) y Descarga (derecha)
  for col_idx, (titulo, det) in enumerate([
      ('2. Transitorio de Carga (Ciclo 1)', detalles_carga_ciclo1),
      ('3. Transitorio de Descarga (Ciclo 1)', detalles_descarga_ciclo1),
  ]):
    ax_fit = fig.add_subplot(gs[1, col_idx])
    ax_res = fig.add_subplot(gs[2, col_idx], sharex=ax_fit)

    # Datos y curva ajustada
    ax_fit.plot(
        det['t_local_ms'],
        det['vc_fit'],
        '.',
        color='#1f77b4',
        markersize=2,
        alpha=0.5,
        label='Datos $V_C(t)$',
    )
    ax_fit.plot(
        det['t_local_ms'],
        det['vc_modelo'],
        '-',
        color='#d62728',
        lw=1.8,
        label=(
            f"Ajuste ($\\tau = {det['tau_ms']:.2f} \\pm"
            f" {det['u_tau_ms']:.3f}$ ms)"
        ),
    )
    ax_fit.axhline(
        det['Vi'],
        color='gray',
        ls=':',
        lw=1,
        label=f"$V_i = {det['Vi']:.3f}$ V",
    )
    ax_fit.axhline(
        det['Vinf'],
        color='black',
        ls='--',
        lw=1,
        label=f"$V_\\infty = {det['Vinf']:.3f}$ V",
    )
    ax_fit.set_title(titulo, fontsize=10, fontweight='bold')
    ax_fit.set_ylabel('Tensión $V_C$ (V)')
    ax_fit.legend(loc='best', fontsize=8)
    ax_fit.grid(True, alpha=0.3)

    # Gráfico de residuos
    ax_res.plot(
        det['t_local_ms'],
        det['residuos_mv'],
        '.',
        color='#2ca02c',
        markersize=2,
        alpha=0.5,
    )
    ax_res.axhline(0, color='black', lw=1, ls='--')
    ax_res.set_xlabel("Tiempo local $t' = t - t_0$ (ms)")
    ax_res.set_ylabel('Residuo $r_i$ (mV)')
    ax_res.set_title(
        f"Residuos $r_i = V_{{C,i}} - V_{{C,modelo}}$ (RMSE = {det['rmse_mv']:.2f} mV)",
        fontsize=9,
    )
    ax_res.grid(True, alpha=0.3)

  plt.savefig(
      f'ajuste_6_1_{R_nom}ohm.png', dpi=300, bbox_inches='tight'
  )
  plt.show()

# =============================================================================
# 5. EXPORTACIÓN DE TABLA COMPLETA DE LOS 30 AJUSTES
# =============================================================================
df_res = pd.DataFrame(lista_resultados)
df_res.to_csv(r"C:\Users\franr\Documents\Facultad\Experimental 2\Informes\Lab 2 - Capacitores\tercera_vuelta\resultados_30_flancos_6_1.csv", index=False)

print('\n=== RESUMEN DE LOS 30 AJUSTES EXPONENCIALES (SECCIÓN 6.1) ===')
print(df_res.to_string(index=False))

print('\n=== MEDIANAS PRELIMINARES POR CONFIGURACIÓN (en ms) ===')
resumen_med = (
    df_res.groupby(['R_nom_ohm', 'Proceso'])['tau_ms']
    .median()
    .unstack()
)
print(resumen_med)
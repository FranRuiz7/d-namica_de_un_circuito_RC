import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# =============================================================================
# CONFIGURACIÓN DE RUTAS Y PARÁMETROS
# =============================================================================
CARPETA = r'C:\Users\franr\Documents\Facultad\Experimental 2\Informes\Lab 2 - Capacitores\tercera_vuelta'

ARCHIVOS_CRUDOS = {
    33: os.path.join(CARPETA, 'datos_33ohm.csv'),
    100: os.path.join(CARPETA, 'datos_100ohm.csv'),
    220: os.path.join(CARPETA, 'datos_220ohm.csv'),
}

ARCHIVO_RES_6_1 = os.path.join(CARPETA, 'resultados_30_flancos_6_1.csv')

# Parámetros de ventana (idénticos a 6.1 para consistencia)
T_FIN_VINF_INI = 0.045  # 45 ms antes del fin del semiperíodo
T_FIN_VINF_FIN = 0.002  # 2 ms antes del fin del semiperíodo
T_MUERTO_INICIO = 0.0004  # 0.4 ms después de t0

# Criterio de corte de ruido para la linealización:
# Excluimos puntos donde |Vc - Vinf| sea menor a N veces el ruido sigma_V (con un piso mínimo de 30 mV)
FACTOR_RUIDO = 10.0
UMBRAL_MINIMO_V = 0.030  # 30 mV (aprox. 5 a 10 escalones de cuantización del ADC)


# =============================================================================
# 1. FUNCIÓN DE LECTURA DE CSV DE PASCO
# =============================================================================
def cargar_csv_pasco(ruta_archivo):
  with open(ruta_archivo, 'r', encoding='utf-8', errors='ignore') as f:
    primera_linea = f.readline()
  sep = (
      ';'
      if ';' in primera_linea
      else ('\t' if '\t' in primera_linea else ',')
  )

  df = pd.read_csv(ruta_archivo, sep=sep, decimal=',')
  df = df.iloc[:, :3].copy()
  df.columns = ['tiempo', 'ch_a', 'ch_c']

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

  max_diff_a = np.max(np.abs(np.diff(df['ch_a'])))
  max_diff_c = np.max(np.abs(np.diff(df['ch_c'])))
  if max_diff_a >= max_diff_c:
    df['vs'], df['vc'] = df['ch_a'], df['ch_c']
  else:
    df['vs'], df['vc'] = df['ch_c'], df['ch_a']

  return df


# =============================================================================
# 2. AJUSTE LINEALIZADO LOGARÍTMICO (Ecuación 10 de la guía)
# =============================================================================
def analizar_linealizacion_descarga(t, vc, t0, Vi, Vinf, t_semiperiodo=0.5):
  """Aplica la transformación Y = ln|Vc(t) - Vinf| y ajusta la recta

  Y = a + m * (t - t0), comparando el ajuste truncado (sin ruido de cola)
  frente al ajuste sin truncar.
  """
  t_fin_flanco = t0 + t_semiperiodo

  # 1. Medir el nivel de ruido real sigma_V en la meseta final (donde se midió Vinf)
  mask_ruido = (t >= (t_fin_flanco - T_FIN_VINF_INI)) & (
      t <= (t_fin_flanco - T_FIN_VINF_FIN)
  )
  sigma_v = np.std(vc[mask_ruido], ddof=1)
  if np.isnan(sigma_v) or sigma_v < 0.001:
    sigma_v = 0.0015  # Piso típico de cuantización de PASCO (~1.5 mV)

  # Definir umbral de exclusión: |Vc - Vinf| > umbral_corte
  umbral_corte = max(FACTOR_RUIDO * sigma_v, UMBRAL_MINIMO_V)

  # 2. Seleccionar todos los puntos del semiperíodo de descarga
  mask_flanco = (t >= (t0 + T_MUERTO_INICIO)) & (
      t <= (t_fin_flanco - T_FIN_VINF_FIN)
  )
  t_local = t[mask_flanco] - t0
  vc_flanco = vc[mask_flanco]
  delta_v = np.abs(vc_flanco - Vinf)

  # Evitar ln(0) en los puntos de la cola que clavan exactamente en Vinf
  mask_positivo = delta_v > 1e-6
  t_local_pos = t_local[mask_positivo]
  delta_v_pos = delta_v[mask_positivo]
  y_ln_pos = np.log(delta_v_pos)

  # 3. Separar puntos válidos (señal > umbral de ruido) vs. puntos dominados por ruido
  mask_valido = delta_v_pos >= umbral_corte
  t_val = t_local_pos[mask_valido]
  y_val = y_ln_pos[mask_valido]

  # 4. Ajuste lineal CORRECTO (excluyendo puntos donde |Vc - Vinf| ~ ruido)
  # Modelo: y = m * t' + a  -->  m = -1/tau  -->  tau = -1/m
  popt_val, pcov_val = np.polyfit(t_val, y_val, 1, cov=True)
  m_val, a_val = popt_val[0], popt_val[1]
  u_m_val = np.sqrt(pcov_val[0, 0])

  tau_lin_s = -1.0 / m_val
  # Propagación de incertidumbre: u(tau) = |d(-1/m)/dm| * u(m) = (1/m^2) * u(m)
  u_tau_lin_s = (u_m_val) / (m_val**2)

  residuos_ln = y_val - (m_val * t_val + a_val)
  ss_res = np.sum(residuos_ln**2)
  ss_tot = np.sum((y_val - np.mean(y_val)) ** 2)
  r2_lin = 1.0 - (ss_res / ss_tot)

  # 5. Ajuste lineal SIN EXCLUIR RUIDO (para demostrar el sesgo en la gráfica)
  popt_todos, _ = np.polyfit(t_local_pos, y_ln_pos, 1, cov=True)
  m_todos, a_todos = popt_todos[0], popt_todos[1]
  tau_sin_corte_ms = (-1.0 / m_todos) * 1000.0

  return {
      'sigma_v_mv': sigma_v * 1000.0,
      'umbral_corte_mv': umbral_corte * 1000.0,
      'n_puntos_usados': len(t_val),
      'n_puntos_excluidos': len(t_local_pos) - len(t_val),
      'tau_lin_ms': tau_lin_s * 1000.0,
      'u_tau_lin_ms': u_tau_lin_s * 1000.0,
      'a_intercepto': a_val,
      'ln_Vi_Vinf_teorico': np.log(np.abs(Vi - Vinf)),
      'r2_lin': r2_lin,
      'tau_sin_corte_ms': tau_sin_corte_ms,
      # Vectores para graficar
      't_local_pos_ms': t_local_pos * 1000.0,
      'y_ln_pos': y_ln_pos,
      'mask_valido': mask_valido,
      't_val_ms': t_val * 1000.0,
      'y_val': y_val,
      'residuos_ln': residuos_ln,
      'sigma_y_teorico': sigma_v / delta_v_pos,
      'm_val': m_val,
      'a_val': a_val,
      'm_todos': m_todos,
      'a_todos': a_todos,
  }


# =============================================================================
# 3. PROCESAMIENTO DE LAS DESCARGAS Y TABLA COMPARATIVA
# =============================================================================
df_6_1 = pd.read_csv(ARCHIVO_RES_6_1)
df_descargas_6_1 = df_6_1[df_6_1['Proceso'] == 'Descarga'].copy()

resultados_comparacion = []
datos_grafica_ciclo1 = {}

for R_nom, ruta_csv in ARCHIVOS_CRUDOS.items():
  df_raw = cargar_csv_pasco(ruta_csv)
  t = df_raw['tiempo'].values
  vc = df_raw['vc'].values

  sub_df = df_descargas_6_1[df_descargas_6_1['R_nom_ohm'] == R_nom]

  for _, fila in sub_df.iterrows():
    ciclo = int(fila['Ciclo'])
    t0 = float(fila['t0_s'])
    Vi = float(fila['Vi_V'])
    Vinf = float(fila['Vinf_V'])
    tau_dir_ms = float(fila['tau_ms'])
    u_tau_dir_ms = float(fila['u_tau_ms'])

    res_lin = analizar_linealizacion_descarga(t, vc, t0, Vi, Vinf)

    dif_ms = res_lin['tau_lin_ms'] - tau_dir_ms
    dif_porcentual = 100.0 * dif_ms / tau_dir_ms

    resultados_comparacion.append({
        'R_nom_ohm': R_nom,
        'Ciclo': ciclo,
        'tau_directo_ms': round(tau_dir_ms, 4),
        'u_tau_directo_ms': round(u_tau_dir_ms, 5),
        'tau_lineal_trunc_ms': round(res_lin['tau_lin_ms'], 4),
        'u_tau_lineal_ms': round(res_lin['u_tau_lin_ms'], 5),
        'tau_lineal_SIN_truncar_ms': round(res_lin['tau_sin_corte_ms'], 2),
        'Diferencia_ms': round(dif_ms, 4),
        'Diferencia_%': round(dif_porcentual, 2),
        'Umbral_corte_mV': round(res_lin['umbral_corte_mv'], 1),
        'Puntos_usados': res_lin['n_puntos_usados'],
        'R2_lineal': round(res_lin['r2_lin'], 5),
    })

    if ciclo == 1:
      res_lin['tau_dir_ms'] = tau_dir_ms
      res_lin['u_tau_dir_ms'] = u_tau_dir_ms
      datos_grafica_ciclo1[R_nom] = res_lin

# Guardar tabla comparativa en CSV
df_comp = pd.DataFrame(resultados_comparacion)
ruta_salida_csv = os.path.join(CARPETA, 'comparacion_linealizacion_6_2.csv')
df_comp.to_csv(ruta_salida_csv, index=False)

print('\n=== COMPARACIÓN AJUSTE DIRECTO (6.1) VS LINEALIZACIÓN LOGARÍTMICA (6.2) ===')
print(df_comp.to_string(index=False))

print('\n=== RESUMEN DE MEDIANAS POR RESISTENCIA (DESCARGAS) ===')
resumen = df_comp.groupby('R_nom_ohm')[[
    'tau_directo_ms',
    'u_tau_directo_ms',
    'tau_lineal_trunc_ms',
    'u_tau_lineal_ms',
    'tau_lineal_SIN_truncar_ms',
    'Diferencia_%',
]].median()
print(resumen)

# =============================================================================
# 4. FIGURA DIAGNÓSTICA DE LA SECCIÓN 6.2 (EL EFECTO DEL LOGARITMO EN EL RUIDO)
# =============================================================================
fig, axes = plt.subplots(2, 3, figsize=(15, 8.5), sharex='col')
plt.subplots_adjust(hspace=0.28, wspace=0.25)

for col_idx, R_nom in enumerate([33, 100, 220]):
  d = datos_grafica_ciclo1[R_nom]
  ax_top = axes[0, col_idx]
  ax_bot = axes[1, col_idx]

  t_all = d['t_local_pos_ms']
  y_all = d['y_ln_pos']
  mask_val = d['mask_valido']

  # Panel superior: Transformación logarítmica ln|Vc - Vinf| vs t'
  ax_top.plot(
      t_all[~mask_val],
      y_all[~mask_val],
      '.',
      color='lightgray',
      markersize=3,
      label='Excluidos ($|V_C - V_\\infty| <$ umbral)',
  )
  ax_top.plot(
      t_all[mask_val],
      y_all[mask_val],
      '.',
      color='#1f77b4',
      markersize=3,
      alpha=0.7,
      label='Incluidos en ajuste lineal',
  )

  # Recta ajustada con corte de ruido (correcta)
  y_recta_val = d['m_val'] * (t_all / 1000.0) + d['a_val']
  ax_top.plot(
      t_all,
      y_recta_val,
      '-',
      color='#d62728',
      lw=2,
      label=f"Lineal truncado ($\\tau = {d['tau_lin_ms']:.2f}$ ms)",
  )

  # Recta si NO se excluye el ruido (sesgada por la cola)
  y_recta_todos = d['m_todos'] * (t_all / 1000.0) + d['a_todos']
  ax_top.plot(
      t_all,
      y_recta_todos,
      '--',
      color='purple',
      lw=1.5,
      label=f"Sin truncar ($\\tau = {d['tau_sin_corte_ms']:.1f}$ ms)",
  )

  # Línea horizontal del umbral de corte
  ax_top.axhline(
      np.log(d['umbral_corte_mv'] / 1000.0),
      color='black',
      ls=':',
      lw=1,
      label=f"Umbral ({d['umbral_corte_mv']:.0f} mV)",
  )

  ax_top.set_title(
      f'Descarga Ciclo 1 ($R_{{nom}} = {R_nom}\\,\\Omega$)\n'
      f"Directo 6.1: $\\tau = {d['tau_dir_ms']:.2f} \\pm {d['u_tau_dir_ms']:.3f}$ ms",
      fontsize=10,
      fontweight='bold',
  )
  ax_top.set_ylabel('$\\ln|V_C(t) - V_\\infty|$ (ln V)')
  ax_top.set_ylim(-8.5, 2.0)
  ax_top.legend(loc='lower left', fontsize=7.5)
  ax_top.grid(True, alpha=0.3)

  # Panel inferior: Residuos en el espacio logarítmico + envolvente +-2*sigma_y
  res_all = y_all - y_recta_val
  ax_bot.plot(
      t_all[~mask_val],
      res_all[~mask_val],
      '.',
      color='lightgray',
      markersize=2.5,
  )
  ax_bot.plot(
      d['t_val_ms'],
      d['residuos_ln'],
      '.',
      color='#2ca02c',
      markersize=3,
      alpha=0.6,
      label='Residuos (zona válida)',
  )

  # Banda teórica de propagación del ruido: +- 2 * sigma_V / |Vc - Vinf|
  orden = np.argsort(t_all)
  t_ord = t_all[orden]
  banda_2sigma = 2.0 * d['sigma_y_teorico'][orden]
  ax_bot.plot(
      t_ord,
      banda_2sigma,
      '--',
      color='darkorange',
      lw=1.2,
      label='Envolvente $\\pm 2\\sigma_V / |V_C - V_\\infty|$',
  )
  ax_bot.plot(t_ord, -banda_2sigma, '--', color='darkorange', lw=1.2)

  ax_bot.axhline(0, color='black', ls='--', lw=1)
  ax_bot.set_xlabel("Tiempo local $t' = t - t_0$ (ms)")
  ax_bot.set_ylabel('Residuo logarítmico')
  ax_bot.set_title('Explosión heterocedástica del ruido', fontsize=9)
  ax_bot.set_ylim(-1.5, 1.5)
  ax_bot.legend(loc='upper left', fontsize=7.5)
  ax_bot.grid(True, alpha=0.3)

ruta_fig = os.path.join(CARPETA, 'linealizacion_6_2_comparacion.png')
plt.savefig(ruta_fig, dpi=300, bbox_inches='tight')
plt.show()
print(f'\nGráfica guardada en: {ruta_fig}')
print(f'Tabla guardada en:   {ruta_salida_csv}')
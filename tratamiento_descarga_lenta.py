import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

# =============================================================================
# CONFIGURACIÓN DE RUTAS Y PARÁMETROS DE LA SECCIÓN 6.4
# =============================================================================
CARPETA = r'C:\Users\franr\Documents\Facultad\Experimental 2\Informes\Lab 2 - Capacitores\tercera_vuelta'

ARCHIVOS_DESCARGA = {
    'Sin multímetro (solo PASCO)': os.path.join(
        CARPETA, 'datos_sin_multimetro.csv'
    ),
    'Con multímetro en paralelo': os.path.join(
        CARPETA, 'datos_con_multimetro.csv'
    ),
}

ARCHIVO_SALIDA_TABLA = os.path.join(CARPETA, 'Resultados_Parte_C_5_3.csv')
ARCHIVO_FIGURA = os.path.join(CARPETA, 'descarga_lenta_5_3.png')

# Capacitancia inferida en la Sección 6.4 (Mediana = 370.7 uF, IC 95% = [354.3, 388.6] uF)
# Para propagar por Montecarlo: sigma_C ~ (388.6 - 354.3) / (2 * 1.96) = 8.75 uF
C_MED_UF = 370.7
C_STD_UF = 8.75
M_MC = 5000
rng = np.random.default_rng(202610)


# =============================================================================
# 1. LECTURA INTELIGENTE DE CSV DE DESCARGA LENTA (2 O MÁS COLUMNAS)
# =============================================================================
def cargar_csv_descarga_lenta(ruta_archivo):
  """Lee el CSV de PASCO Capstone, convierte comas decimales y selecciona

  automáticamente la columna de tiempo y la columna del canal que mide V_C(t).
  """
  with open(ruta_archivo, 'r', encoding='utf-8', errors='ignore') as f:
    primera_linea = f.readline()
  sep = (
      ';'
      if ';' in primera_linea
      else ('\t' if '\t' in primera_linea else ',')
  )

  df = pd.read_csv(ruta_archivo, sep=sep, decimal=',')

  # Convertir todas las columnas a numérico
  for col in df.columns:
    if df[col].dtype == object:
      df[col] = (
          df[col]
          .astype(str)
          .str.replace(',', '.', regex=False)
          .str.strip()
      )
      df[col] = pd.to_numeric(df[col], errors='coerce')

  # Eliminar columnas completamente vacías y filas con NaN
  df = df.dropna(axis=1, how='all').dropna().reset_index(drop=True)

  # La primera columna es Tiempo (s)
  t = df.iloc[:, 0].values

  # Si se exportaron más de 2 columnas (ej. Ch A y Ch C), elegir la que tiene
  # voltaje inicial cercano a 3-4 V (el capacitor cargado)
  mejor_col = 1
  max_v_ini = -1.0
  for c_idx in range(1, df.shape[1]):
    v_col = df.iloc[:, c_idx].values
    v_ini_prom = np.mean(v_col[: min(20, len(v_col))])
    if v_ini_prom > max_v_ini:
      max_v_ini = v_ini_prom
      mejor_col = c_idx

  vc = df.iloc[:, mejor_col].values
  return t, vc


# =============================================================================
# 2. DETECCIÓN DEL INICIO DE LA DESCARGA Y AJUSTE EXPONENCIAL
# =============================================================================
def ajustar_descarga_lenta(t_raw, vc_raw):
  """Detecta si hubo unos segundos iniciales con la fuente aún conectada en ~4 V,

  recorta desde el instante exacto de desconexión t0 y realiza:
    A) Ajuste exponencial con offset libre (punto 5.3.2 de la guía):
       V_C(t) = V_inf + (V_0 - V_inf) * exp(-(t - t0) / tau_abierto)
    B) Ajuste exponencial con asíntota física en reposo V_inf = 0 V:
       V_C(t) = V_0 * exp(-(t - t0) / tau_abierto)
  """
  # Detectar el instante de desconexión (por si se empezó a grabar antes de abrir el interruptor)
  v_max = np.max(vc_raw[: min(200, len(vc_raw))])
  # Buscar el último punto al inicio que sigue pegado al techo (dentro de 5 mV de v_max)
  idx_techo = np.where(vc_raw[: min(300, len(vc_raw))] >= (v_max - 0.005))[0]
  idx_ini = idx_techo[-1] if len(idx_techo) > 0 else 0

  t0 = t_raw[idx_ini]
  t = t_raw[idx_ini:] - t0  # Tiempo local desde el inicio de la descarga (s)
  vc = vc_raw[idx_ini:]

  # Estimación de semilla inicial para tau usando los extremos del registro
  v_ini = vc[0]
  v_fin = vc[-1]
  duracion = t[-1]
  tau_semilla = (
      duracion / np.log(v_ini / max(v_fin, 0.1))
      if v_ini > v_fin > 0
      else 300.0
  )

  # --- MODELO A: Exponencial con offset V_inf libre (pedido en 5.3.2) ---
  def modelo_con_offset(t_val, v_inf, delta_v, tau):
    return v_inf + delta_v * np.exp(-t_val / tau)

  popt_off, pcov_off = curve_fit(
      modelo_con_offset,
      t,
      vc,
      p0=[0.0, v_ini, tau_semilla],
      bounds=([-1.0, 0.5, 10.0], [v_fin * 0.99, 5.0, 10000.0]),
      maxfev=10000,
  )
  v_inf_opt, delta_v_opt, tau_off_s = popt_off
  u_v_inf, u_delta_v, u_tau_off_s = np.sqrt(np.diag(pcov_off))
  vc_mod_off = modelo_con_offset(t, *popt_off)
  res_off_mv = (vc - vc_mod_off) * 1000.0
  rmse_off_mv = np.sqrt(np.mean(res_off_mv**2))

  # --- MODELO B: Exponencial con V_inf = 0 V (descarga completa hacia masa) ---
  def modelo_sin_offset(t_val, v0, tau):
    return v0 * np.exp(-t_val / tau)

  popt_cero, pcov_cero = curve_fit(
      modelo_sin_offset,
      t,
      vc,
      p0=[v_ini, tau_semilla],
      bounds=([0.5, 10.0], [5.0, 10000.0]),
  )
  v0_cero, tau_cero_s = popt_cero
  u_v0_cero, u_tau_cero_s = np.sqrt(np.diag(pcov_cero))
  vc_mod_cero = modelo_sin_offset(t, *popt_cero)
  res_cero_mv = (vc - vc_mod_cero) * 1000.0
  rmse_cero_mv = np.sqrt(np.mean(res_cero_mv**2))

  return {
      't_s': t,
      'vc_v': vc,
      'duracion_s': duracion,
      'v_inicial_v': v_ini,
      'v_final_v': v_fin,
      'caida_v': v_ini - v_fin,
      # Resultados Modelo con Offset (5.3.2)
      'tau_off_s': tau_off_s,
      'u_tau_off_s': u_tau_off_s,
      'v_inf_v': v_inf_opt,
      'u_v_inf_v': u_v_inf,
      'v0_off_v': v_inf_opt + delta_v_opt,
      'rmse_off_mv': rmse_off_mv,
      'vc_mod_off': vc_mod_off,
      'res_off_mv': res_off_mv,
      # Resultados Modelo V_inf = 0 V
      'tau_cero_s': tau_cero_s,
      'u_tau_cero_s': u_tau_cero_s,
      'v0_cero_v': v0_cero,
      'rmse_cero_mv': rmse_cero_mv,
      'vc_mod_cero': vc_mod_cero,
      'res_cero_mv': res_cero_mv,
  }


# =============================================================================
# 3. PROCESAMIENTO DE AMBOS ARCHIVOS Y PROPAGACIÓN MONTECARLO HACIA R_{||,ef}
# =============================================================================
resultados_dict = {}
filas_tabla = []

# Muestras de C (en Faradios) desde la Sección 6.4
C_mc_F = rng.normal(C_MED_UF, C_STD_UF, size=M_MC) * 1e-6

for nombre_config, ruta in ARCHIVOS_DESCARGA.items():
  print(f'\nProcesando: {nombre_config} ({ruta})...')
  t_raw, vc_raw = cargar_csv_descarga_lenta(ruta)
  res = ajustar_descarga_lenta(t_raw, vc_raw)

  # Propagación por Montecarlo hacia R_{||,ef} = tau_abierto / C (en MOhm)
  # 1) Usando tau del modelo con offset libre
  tau_off_mc = rng.normal(res['tau_off_s'], res['u_tau_off_s'], size=M_MC)
  R_par_off_MOhm = (tau_off_mc / C_mc_F) / 1e6  # en MOhm
  R_off_med = np.median(R_par_off_MOhm)
  R_off_ic = np.percentile(R_par_off_MOhm, [2.5, 97.5])

  # 2) Usando tau del modelo con V_inf = 0 V
  tau_cero_mc = rng.normal(res['tau_cero_s'], res['u_tau_cero_s'], size=M_MC)
  R_par_cero_MOhm = (tau_cero_mc / C_mc_F) / 1e6  # en MOhm
  R_cero_med = np.median(R_par_cero_MOhm)
  R_cero_ic = np.percentile(R_par_cero_MOhm, [2.5, 97.5])

  res['R_par_off_mc_MOhm'] = R_par_off_MOhm
  res['R_par_cero_mc_MOhm'] = R_par_cero_MOhm
  resultados_dict[nombre_config] = res

  filas_tabla.append({
      'Configuracion': nombre_config,
      'Duracion_s': round(res['duracion_s'], 1),
      'V_inicial_V': round(res['v_inicial_v'], 3),
      'V_final_V': round(res['v_final_v'], 3),
      'Delta_V_resoluble_V': round(res['caida_v'], 3),
      # Modelo con offset (5.3.2)
      'tau_con_offset_s': f"{res['tau_off_s']:.2f} ± {res['u_tau_off_s']:.2f}",
      'V_inf_offset_V': f"{res['v_inf_v']:.3f} ± {res['u_v_inf_v']:.3f}",
      'R_par_ef_offset_MOhm': round(R_off_med, 3),
      'IC95_R_par_offset_MOhm': f'[{R_off_ic[0]:.3f}, {R_off_ic[1]:.3f}]',
      'RMSE_offset_mV': round(res['rmse_off_mv'], 2),
      # Modelo con V_inf = 0 V
      'tau_Vinf0_s': f"{res['tau_cero_s']:.2f} ± {res['u_tau_cero_s']:.2f}",
      'R_par_ef_Vinf0_MOhm': round(R_cero_med, 3),
      'IC95_R_par_Vinf0_MOhm': f'[{R_cero_ic[0]:.3f}, {R_cero_ic[1]:.3f}]',
      'RMSE_Vinf0_mV': round(res['rmse_cero_mv'], 2),
  })

df_res_5_3 = pd.DataFrame(filas_tabla)
df_res_5_3.to_csv(ARCHIVO_SALIDA_TABLA, index=False)

print('\n=== RESULTADOS DE LA PARTE C / SECCIÓN 5.3 (DESCARGA LENTA) ===')
print(df_res_5_3.to_string(index=False))

# =============================================================================
# 4. EVALUACIÓN CUANTITATIVA DEL EFECTO DE CARGA DEL MULTÍMETRO (Pregunta 8)
# =============================================================================
res_sin = resultados_dict['Sin multímetro (solo PASCO)']
res_con = resultados_dict['Con multímetro en paralelo']

for etiqueta_mod, clave_r, clave_tau in [
    ('Modelo con offset (5.3.2)', 'R_par_off_mc_MOhm', 'tau_off_s'),
    ('Modelo asintótico (V_inf = 0 V)', 'R_par_cero_mc_MOhm', 'tau_cero_s'),
]:
  R_sin_mc = res_sin[clave_r]
  R_con_mc = res_con[clave_r]
  tau_sin = res_sin[clave_tau]
  tau_con = res_con[clave_tau]

  cambio_tau_pct = 100.0 * (tau_con - tau_sin) / tau_sin

  # Si R_con < R_sin, podemos despejar la resistencia interna del multímetro:
  # 1 / R_con = 1 / R_sin + 1 / R_mult  ==>  R_mult = 1 / (1/R_con - 1/R_sin)
  diff_inv = (1.0 / R_con_mc) - (1.0 / R_sin_mc)
  validos = diff_inv > 0
  print(f'\n--- Comparación ({etiqueta_mod}) ---')
  print(
      f'  tau_abierto (Sin multímetro): {tau_sin:.2f} s  |  R_||,ef ='
      f' {np.median(R_sin_mc):.3f} MOhm'
  )
  print(
      f'  tau_abierto (Con multímetro): {tau_con:.2f} s  |  R_||,ef ='
      f' {np.median(R_con_mc):.3f} MOhm'
  )
  print(
      '  Variación relativa de la constante de tiempo al conectar el'
      f' multímetro: {cambio_tau_pct:+.2f}%'
  )
  if np.mean(validos) > 0.5:
    R_mult_mc = 1.0 / diff_inv[validos]
    R_mult_med = np.median(R_mult_mc)
    R_mult_ic = np.percentile(R_mult_mc, [2.5, 97.5])
    print(
        '  Resistencia de entrada inferida del multímetro R_in,mult ='
        f' {R_mult_med:.2f} MOhm (IC 95%: [{R_mult_ic[0]:.2f},'
        f' {R_mult_ic[1]:.2f}] MOhm)'
    )

# =============================================================================
# 5. GRÁFICA COMPARATIVA DE LA PARTE C (CURVAS, AJUSTES Y RESIDUOS)
# =============================================================================
fig, axes = plt.subplots(
    2,
    2,
    figsize=(14, 8.5),
    gridspec_kw={'height_ratios': [1.4, 0.85], 'hspace': 0.32, 'wspace': 0.24},
)

# Panel Superior Izquierdo: Comparación directa de ambas curvas de descarga
ax_comp = axes[0, 0]
colores = {
    'Sin multímetro (solo PASCO)': '#1f77b4',
    'Con multímetro en paralelo': '#d62728',
}

for nombre_config, res in resultados_dict.items():
  c = colores[nombre_config]
  R_med = np.median(res['R_par_off_mc_MOhm'])
  ax_comp.plot(
      res['t_s'],
      res['vc_v'],
      '.',
      color=c,
      markersize=2,
      alpha=0.35,
  )
  ax_comp.plot(
      res['t_s'],
      res['vc_mod_off'],
      '-',
      color=c,
      lw=2.0,
      label=(
          f"{nombre_config}\n$\\tau_{{abierto}} = {res['tau_off_s']:.1f}$ s |"
          f' $R_{{\\parallel,ef}} = {R_med:.2f}$ M$\\Omega$'
      ),
  )

ax_comp.set_title(
    '1. Comparación de Descarga Lenta: Efecto de Carga Instrumental',
    fontsize=10.5,
    fontweight='bold',
)
ax_comp.set_xlabel('Tiempo desde desconexión $t - t_0$ (s)')
ax_comp.set_ylabel('Tensión en el capacitor $V_C(t)$ (V)')
ax_comp.legend(loc='upper right', fontsize=8.5)
ax_comp.grid(True, alpha=0.3)

# Panel Inferior Izquierdo: Residuos del ajuste con offset para ambas configuraciones
ax_res_off = axes[1, 0]
for nombre_config, res in resultados_dict.items():
  c = colores[nombre_config]
  ax_res_off.plot(
      res['t_s'],
      res['res_off_mv'],
      '.',
      color=c,
      markersize=2,
      alpha=0.5,
      label=f"{nombre_config} (RMSE = {res['rmse_off_mv']:.1f} mV)",
  )
ax_res_off.axhline(0, color='black', ls='--', lw=1.2)
ax_res_off.set_title(
    'Residuos del modelo exponencial con offset ($V_\\infty$ libre)',
    fontsize=9.5,
    fontweight='bold',
)
ax_res_off.set_xlabel('Tiempo $t - t_0$ (s)')
ax_res_off.set_ylabel('Residuo $r_i$ (mV)')
ax_res_off.legend(loc='best', fontsize=8)
ax_res_off.grid(True, alpha=0.3)

# Panel Superior Derecho: Escala semilogarítmica ln(V_C) para visualizar cambios de pendiente
ax_log = axes[0, 1]
for nombre_config, res in resultados_dict.items():
  c = colores[nombre_config]
  R_med_0 = np.median(res['R_par_cero_mc_MOhm'])
  ax_log.plot(
      res['t_s'],
      np.log(np.maximum(res['vc_v'], 1e-4)),
      '.',
      color=c,
      markersize=2,
      alpha=0.35,
  )
  ax_log.plot(
      res['t_s'],
      np.log(res['vc_mod_cero']),
      '--',
      color=c,
      lw=1.8,
      label=(
          f"{nombre_config} ($V_\\infty=0$)\n$\\tau = {res['tau_cero_s']:.1f}$"
          f' s | $R_{{\\parallel,ef}} = {R_med_0:.2f}$ M$\\Omega$'
      ),
  )

ax_log.set_title(
    '2. Vista Semilogarítmica $\\ln(V_C)$ y Ajuste con $V_\\infty = 0$ V',
    fontsize=10.5,
    fontweight='bold',
)
ax_log.set_xlabel('Tiempo desde desconexión $t - t_0$ (s)')
ax_log.set_ylabel('$\\ln[V_C(t) / 1\\,\\mathrm{V}]$')
ax_log.legend(loc='upper right', fontsize=8.5)
ax_log.grid(True, alpha=0.3)

# Panel Inferior Derecho: Histogramas Montecarlo de R_{||,ef} (MOhm)
ax_hist = axes[1, 1]
for nombre_config, res in resultados_dict.items():
  c = colores[nombre_config]
  dist_r = res['R_par_off_mc_MOhm']
  med_r = np.median(dist_r)
  ic_r = np.percentile(dist_r, [2.5, 97.5])
  ax_hist.hist(
      dist_r,
      bins=35,
      density=True,
      color=c,
      alpha=0.55,
      edgecolor='white',
      label=(
          f'{nombre_config}: {med_r:.2f} M$\\Omega$ [{ic_r[0]:.2f},'
          f' {ic_r[1]:.2f}]'
      ),
  )
  ax_hist.axvline(med_r, color=c, ls='--', lw=1.8)

ax_hist.set_title(
    '3. Distribución Montecarlo de $R_{\\parallel,ef} = \\tau_{abierto} / C$',
    fontsize=9.5,
    fontweight='bold',
)
ax_hist.set_xlabel('Resistencia paralela efectiva $R_{\\parallel,ef}$ (M$\\Omega$)')
ax_hist.set_ylabel('Densidad de probabilidad')
ax_hist.legend(loc='best', fontsize=8)
ax_hist.grid(True, alpha=0.3)

plt.savefig(ARCHIVO_FIGURA, dpi=300, bbox_inches='tight')
plt.show()

print(f'\nTabla guardada en:  {ARCHIVO_SALIDA_TABLA}')
print(f'Figura guardada en: {ARCHIVO_FIGURA}')
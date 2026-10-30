import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# =============================================================================
# CONFIGURACIÓN DE RUTAS
# =============================================================================
CARPETA = r'C:\Users\franr\Documents\Facultad\Experimental 2\Informes\Lab 2 - Capacitores\tercera_vuelta'
ARCHIVO_ENTRADA = os.path.join(CARPETA, 'resultados_30_flancos_6_1.csv')
ARCHIVO_SALIDA_CICLOS = os.path.join(CARPETA, 'compatibilidad_ciclos_6_3.csv')
ARCHIVO_SALIDA_CUADRO3 = os.path.join(CARPETA, 'resumen_cuadro_3_6_3.csv')
ARCHIVO_FIGURA = os.path.join(CARPETA, 'compatibilidad_6_3.png')

# =============================================================================
# 1. CARGA Y EMPAREJAMIENTO DE CARGA Y DESCARGA POR CICLO
# =============================================================================
df_6_1 = pd.read_csv(ARCHIVO_ENTRADA)

# Separar ramas de Carga y Descarga
df_carga = df_6_1[df_6_1['Proceso'] == 'Carga'].copy()
df_descarga = df_6_1[df_6_1['Proceso'] == 'Descarga'].copy()

# Unir por resistencia (R_nom_ohm) y número de ciclo (Ciclo j = 1..5)
df_ciclos = pd.merge(
    df_carga[['R_nom_ohm', 'Ciclo', 'tau_ms', 'u_tau_ms', 'RMSE_residuos_mV']],
    df_descarga[
        ['R_nom_ohm', 'Ciclo', 'tau_ms', 'u_tau_ms', 'RMSE_residuos_mV']
    ],
    on=['R_nom_ohm', 'Ciclo'],
    suffixes=('_c', '_d'),
).sort_values(['R_nom_ohm', 'Ciclo']).reset_index(drop=True)

# =============================================================================
# 2. EVALUACIÓN DE COMPATIBILIDAD (Ecuaciones 11, 12 y 13)
# =============================================================================
# Ecuación (11): Delta_kj = tau_c,kj - tau_d,kj  y  u(Delta_kj) = sqrt(u^2(tau_c) + u^2(tau_d))
df_ciclos['Delta_kj_ms'] = df_ciclos['tau_ms_c'] - df_ciclos['tau_ms_d']
df_ciclos['u_Delta_kj_ms'] = np.sqrt(
    df_ciclos['u_tau_ms_c'] ** 2 + df_ciclos['u_tau_ms_d'] ** 2
)

# Criterio operativo: |Delta_kj| <= 2 * u(Delta_kj)
df_ciclos['umbral_2u_ms'] = 2.0 * df_ciclos['u_Delta_kj_ms']
df_ciclos['z_score'] = (
    np.abs(df_ciclos['Delta_kj_ms']) / df_ciclos['u_Delta_kj_ms']
)
df_ciclos['Compatible_2u'] = (
    np.abs(df_ciclos['Delta_kj_ms']) <= df_ciclos['umbral_2u_ms']
)

# Diferencia relativa porcentual entre carga y descarga
df_ciclos['Dif_relativa_%'] = (
    100.0
    * df_ciclos['Delta_kj_ms']
    / (0.5 * (df_ciclos['tau_ms_c'] + df_ciclos['tau_ms_d']))
)

# Ecuación (12): Promedio por ciclo tau_k,j = (tau_c,kj + tau_d,kj) / 2
# Incertidumbre propagada del promedio: u(tau_k,j) = 0.5 * sqrt(u^2(tau_c) + u^2(tau_d))
df_ciclos['tau_kj_prom_ms'] = 0.5 * (
    df_ciclos['tau_ms_c'] + df_ciclos['tau_ms_d']
)
df_ciclos['u_tau_kj_prom_ms'] = 0.5 * df_ciclos['u_Delta_kj_ms']

# =============================================================================
# 3. RESUMEN POR RESISTENCIA (Ecuación 13 y Cuadro 3 de la guía)
# =============================================================================
filas_cuadro3 = []

for R_nom, grupo in df_ciclos.groupby('R_nom_ohm'):
  med_c = np.median(grupo['tau_ms_c'])
  med_d = np.median(grupo['tau_ms_d'])
  med_prom = np.median(grupo['tau_kj_prom_ms'])

  # Dispersión entre los 5 ciclos (desviación estándar muestral y rango intercuartílico)
  sd_c = np.std(grupo['tau_ms_c'], ddof=1)
  sd_d = np.std(grupo['tau_ms_d'], ddof=1)
  sd_prom = np.std(grupo['tau_kj_prom_ms'], ddof=1)

  n_compatibles = int(np.sum(grupo['Compatible_2u']))
  dif_med_ms = med_c - med_d
  dif_med_pct = 100.0 * dif_med_ms / med_prom

  # Decisión según criterio de la Sección 6.3
  if n_compatibles == len(grupo):
    decision = 'Compatibles (5/5): usar tau_kj promediado'
  elif n_compatibles >= 3:
    decision = (
        f'Parcial ({n_compatibles}/5): evaluar si domina subestimación de u(tau)'
    )
  else:
    decision = (
        f'No compatibles ({n_compatibles}/5): analizar tau_c(R) y tau_d(R)'
        ' por separado'
    )

  filas_cuadro3.append({
      'Config': f'R_{R_nom}',
      'R_nom_ohm': R_nom,
      'med_tau_c_ms': round(med_c, 4),
      'sd_ciclos_c_ms': round(sd_c, 4),
      'med_tau_d_ms': round(med_d, 4),
      'sd_ciclos_d_ms': round(sd_d, 4),
      'tau_hat_k_prom_ms': round(med_prom, 4),
      'sd_ciclos_prom_ms': round(sd_prom, 4),
      'med(tau_c)-med(tau_d) [ms]': round(dif_med_ms, 4),
      'Dif_medianas_%': round(dif_med_pct, 2),
      'Ciclos_compatibles': f'{n_compatibles}/5',
      'Decision_6_3': decision,
  })

df_cuadro3 = pd.DataFrame(filas_cuadro3)

# Guardar ambos CSV
df_ciclos_export = df_ciclos.round({
    'tau_ms_c': 4,
    'u_tau_ms_c': 5,
    'tau_ms_d': 4,
    'u_tau_ms_d': 5,
    'Delta_kj_ms': 4,
    'u_Delta_kj_ms': 5,
    'umbral_2u_ms': 5,
    'z_score': 2,
    'Dif_relativa_%': 2,
    'tau_kj_prom_ms': 4,
    'u_tau_kj_prom_ms': 5,
})
df_ciclos_export.to_csv(ARCHIVO_SALIDA_CICLOS, index=False)
df_cuadro3.to_csv(ARCHIVO_SALIDA_CUADRO3, index=False)

print('\n=== EVALUACIÓN CICLO A CICLO: ECUACIÓN (11) Y (12) ===')
cols_mostrar = [
    'R_nom_ohm',
    'Ciclo',
    'tau_ms_c',
    'u_tau_ms_c',
    'tau_ms_d',
    'u_tau_ms_d',
    'Delta_kj_ms',
    'umbral_2u_ms',
    'z_score',
    'Compatible_2u',
    'tau_kj_prom_ms',
]
print(df_ciclos_export[cols_mostrar].to_string(index=False))

print('\n=== RESUMEN PARA COMPLETAR EL CUADRO 3 DE LA GUÍA (ECUACIÓN 13) ===')
print(df_cuadro3.to_string(index=False))

# =============================================================================
# 4. GRÁFICA DE DIAGNÓSTICO DE COMPATIBILIDAD Y REPETIBILIDAD CICLO A CICLO
# =============================================================================
fig, axes = plt.subplots(2, 3, figsize=(15, 8.5), sharex='col')
plt.subplots_adjust(hspace=0.30, wspace=0.25)

for col_idx, R_nom in enumerate([33, 100, 220]):
  sub = df_ciclos[df_ciclos['R_nom_ohm'] == R_nom]
  res_r = df_cuadro3[df_cuadro3['R_nom_ohm'] == R_nom].iloc[0]

  ciclos = sub['Ciclo'].values
  tau_c = sub['tau_ms_c'].values
  u_c = sub['u_tau_ms_c'].values
  tau_d = sub['tau_ms_d'].values
  u_d = sub['u_tau_ms_d'].values
  tau_p = sub['tau_kj_prom_ms'].values

  delta = sub['Delta_kj_ms'].values
  u_delta_2 = sub['umbral_2u_ms'].values

  ax_top = axes[0, col_idx]
  ax_bot = axes[1, col_idx]

  # -------------------------------------------------------------------------
  # Panel superior: tau_c,kj y tau_d,kj en función del ciclo j (barras +-2u)
  # -------------------------------------------------------------------------
  ax_top.errorbar(
      ciclos - 0.06,
      tau_c,
      yerr=2.0 * u_c,
      fmt='o-',
      color='#d62728',
      capsize=4,
      lw=1.2,
      label=f"Carga $\\tau_c$ (med = {res_r['med_tau_c_ms']:.2f} ms)",
  )
  ax_top.errorbar(
      ciclos + 0.06,
      tau_d,
      yerr=2.0 * u_d,
      fmt='s-',
      color='#1f77b4',
      capsize=4,
      lw=1.2,
      label=f"Descarga $\\tau_d$ (med = {res_r['med_tau_d_ms']:.2f} ms)",
  )
  ax_top.plot(
      ciclos,
      tau_p,
      'd--',
      color='black',
      alpha=0.7,
      markersize=5,
      label=f"Promedio $\\tau_{{k,j}}$ ($\\hat{{\\tau}}_k$ = {res_r['tau_hat_k_prom_ms']:.2f} ms)",
  )

  ax_top.set_title(
      f'Evolución por Ciclo ($R_{{nom}} = {R_nom}\\,\\Omega$)\n'
      f"Compatibles: {res_r['Ciclos_compatibles']} ciclos",
      fontsize=10,
      fontweight='bold',
  )
  ax_top.set_ylabel('Constante de tiempo $\\tau$ (ms)')
  ax_top.set_xticks([1, 2, 3, 4, 5])
  ax_top.legend(loc='best', fontsize=8)
  ax_top.grid(True, alpha=0.3)

  # -------------------------------------------------------------------------
  # Panel inferior: Prueba de compatibilidad Delta_kj +- 2*u(Delta_kj) vs 0
  # -------------------------------------------------------------------------
  ax_bot.axhline(
      0.0, color='black', ls='--', lw=1.5, label='Compatibilidad exacta ($\Delta = 0$)'
  )

  # Colorear cada punto según cumpla o no |Delta_kj| <= 2*u(Delta_kj)
  for j_idx in range(len(ciclos)):
    color_pto = '#2ca02c' if sub['Compatible_2u'].iloc[j_idx] else '#ff7f0e'
    ax_bot.errorbar(
        ciclos[j_idx],
        delta[j_idx],
        yerr=u_delta_2[j_idx],
        fmt='o',
        color=color_pto,
        ecolor=color_pto,
        capsize=5,
        lw=1.8,
        markersize=6,
    )

  # Conectar puntos con línea suave para ver tendencia temporal
  ax_bot.plot(ciclos, delta, ':', color='gray', lw=1)

  # Banda sombreada del promedio de +-2*u(Delta_kj) como referencia visual
  u2_media = np.mean(u_delta_2)
  ax_bot.axhspan(
      -u2_media,
      u2_media,
      color='green',
      alpha=0.12,
      label=f'Zona $\\pm 2u(\\Delta)$ media ($\\pm {u2_media:.3f}$ ms)',
  )

  ax_bot.set_title(
      'Diferencia $\\Delta_{kj} = \\tau_{c,kj} - \\tau_{d,kj} \\pm 2u(\\Delta_{kj})$',
      fontsize=9.5,
  )
  ax_bot.set_xlabel('Número de ciclo $j$')
  ax_bot.set_ylabel('$\\Delta_{kj}$ (ms)')
  ax_bot.set_xticks([1, 2, 3, 4, 5])
  ax_bot.legend(loc='best', fontsize=8)
  ax_bot.grid(True, alpha=0.3)

plt.savefig(ARCHIVO_FIGURA, dpi=300, bbox_inches='tight')
plt.show()

print(f'\nArchivo por ciclos guardado en: {ARCHIVO_SALIDA_CICLOS}')
print(f'Resumen Cuadro 3 guardado en:   {ARCHIVO_SALIDA_CUADRO3}')
print(f'Figura guardada en:             {ARCHIVO_FIGURA}')
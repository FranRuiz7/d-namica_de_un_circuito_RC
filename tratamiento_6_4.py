import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# =============================================================================
# CONFIGURACIÓN DE RUTAS Y PARÁMETROS EXPERIMENTALES
# =============================================================================
CARPETA = r'C:\Users\franr\Documents\Facultad\Experimental 2\Informes\Lab 2 - Capacitores\tercera_vuelta'
ARCHIVO_ENTRADA_6_3 = os.path.join(CARPETA, 'compatibilidad_ciclos_6_3.csv')

ARCHIVO_SALIDA_CUADRO3 = os.path.join(
    CARPETA, 'Cuadro_3_completo_con_Bootstrap.csv'
)
ARCHIVO_SALIDA_CUADRO4 = os.path.join(CARPETA, 'Cuadro_4_Montecarlo_C_y_Rs.csv')
FIGURA_BOOTSTRAP = os.path.join(CARPETA, 'bootstrap_ciclos_6_4.png')
FIGURA_MONTECARLO = os.path.join(CARPETA, 'montecarlo_propagacion_6_4.png')

# Semilla fija para garantizar reproducibilidad exacta (requisito Sección 9)
SEED = 202610
rng = np.random.default_rng(SEED)

B_BOOT = 5000  # Número de remuestras bootstrap exigido por la guía
M_MC = 5000  # Número de realizaciones Montecarlo exigido por la guía

# Mediciones experimentales de resistencia (convertidas todas a Ohm)
# R_220 = (0.215 +- 0.004) kOhm = (215.0 +- 4.0) Ohm
RESISTENCIAS_EXP = {
    33: {'R_med': 33.1, 'u_R': 0.6, 'etiqueta': '33,1 ± 0,6 Ω (rango 200 Ω)'},
    100: {
        'R_med': 99.5,
        'u_R': 1.2,
        'etiqueta': '99,5 ± 1,2 Ω (rango 200 Ω)',
    },
    220: {
        'R_med': 215.0,
        'u_R': 4.0,
        'etiqueta': '215 ± 4 Ω [0,215 ± 0,004 kΩ]',
    },
}

# =============================================================================
# 1. LECTURA DE LOS 5 CICLOS POR RESISTENCIA DESDE LA SECCIÓN 6.3
# =============================================================================
df_ciclos = pd.read_csv(ARCHIVO_ENTRADA_6_3)

# =============================================================================
# 2. BOOTSTRAP POR CICLOS (N = 5, B = 5000 CON REEMPLAZO)
# =============================================================================
distribuciones_boot_ms = {}
ejemplo_una_remuestra = {}
filas_cuadro3 = []

for R_nom in [33, 100, 220]:
  sub = (
      df_ciclos[df_ciclos['R_nom_ohm'] == R_nom]
      .sort_values('Ciclo')
      .reset_index(drop=True)
  )

  # Vector de N = 5 constantes de tiempo por ciclo tau_{k,j} (en ms)
  taus_originales = sub['tau_kj_prom_ms'].values
  N = len(taus_originales)  # N = 5 ciclos

  # Generar matriz de 5000 remuestras con reemplazo de tamaño N = 5
  indices_boot = rng.integers(0, N, size=(B_BOOT, N))
  remuestras_boot = taus_originales[indices_boot]

  # Calcular la mediana en cada una de las 5000 remuestras
  medianas_boot = np.median(remuestras_boot, axis=1)
  distribuciones_boot_ms[R_nom] = medianas_boot
  ejemplo_una_remuestra[R_nom] = remuestras_boot[0]

  # Estimador puntual (mediana original) e Intervalo de Confianza central del 95%
  tau_hat_k = np.median(taus_originales)
  ic_inf, ic_sup = np.percentile(medianas_boot, [2.5, 97.5])
  u_boot_std = np.std(medianas_boot, ddof=1)

  # Medianas de carga y descarga para completar el Cuadro 3
  med_c = np.median(sub['tau_ms_c'].values)
  med_d = np.median(sub['tau_ms_d'].values)

  filas_cuadro3.append({
      'Config': f'R_{R_nom}',
      'R_medida_ohm': (
          f"{RESISTENCIAS_EXP[R_nom]['R_med']} ±"
          f" {RESISTENCIAS_EXP[R_nom]['u_R']}"
      ),
      'med_tau_c_ms': round(med_c, 3),
      'med_tau_d_ms': round(med_d, 3),
      'tau_hat_k_ms': round(tau_hat_k, 3),
      'IC_95%_bootstrap_ms': f'[{ic_inf:.3f}, {ic_sup:.3f}]',
      'u_boot_std_ms': round(u_boot_std, 4),
      'N_ciclos': N,
      'Diagnostico_residuos': (
          'Residuo estructurado en S (absorción dieléctrica + ESR); RMSE ~'
          ' 18-24 mV'
      ),
  })

df_cuadro3 = pd.DataFrame(filas_cuadro3)
df_cuadro3.to_csv(ARCHIVO_SALIDA_CUADRO3, index=False)

print(
    '\n=== CUADRO 3 COMPLETO (CON INTERVALOS DE CONFIANZA BOOTSTRAP 95%) ==='
)
print(df_cuadro3.to_string(index=False))

# =============================================================================
# 3. PROPAGACIÓN POR MONTECARLO HACIA C Y R_s (M = 5000 REALIZACIONES)
# =============================================================================
# 3.1 Generar realizaciones Montecarlo para cada resistencia R_k ~ N(R_med, u_R^2)
R_33_mc = rng.normal(
    RESISTENCIAS_EXP[33]['R_med'], RESISTENCIAS_EXP[33]['u_R'], size=M_MC
)
R_100_mc = rng.normal(
    RESISTENCIAS_EXP[100]['R_med'], RESISTENCIAS_EXP[100]['u_R'], size=M_MC
)
R_220_mc = rng.normal(
    RESISTENCIAS_EXP[220]['R_med'], RESISTENCIAS_EXP[220]['u_R'], size=M_MC
)

# 3.2 Seleccionar las realizaciones bootstrap de tau_hat_k (en ms)
tau_33_mc = distribuciones_boot_ms[33]
tau_100_mc = distribuciones_boot_ms[100]
tau_220_mc = distribuciones_boot_ms[220]

# Matrices (M_MC x 3) con los tres pares (R_k, tau_k) de cada realización
X_R = np.column_stack([R_33_mc, R_100_mc, R_220_mc])  # en Ohm
Y_tau_ms = np.column_stack([tau_33_mc, tau_100_mc, tau_220_mc])  # en ms

# 3.3 Ajuste lineal exacto por cuadrados mínimos en cada una de las 5000 realizaciones:
# Modelo: tau [ms] = m [ms/Ohm] * R [Ohm] + b [ms]
# Nota de unidades: 1 ms / Ohm = 1e-3 s / Ohm = 1e-3 F = 1000 uF.
# Por lo tanto: C [uF] = m [ms/Ohm] * 1000.0, y R_s [Ohm] = b [ms] / m [ms/Ohm].
R_media_fila = np.mean(X_R, axis=1, keepdims=True)
tau_media_fila = np.mean(Y_tau_ms, axis=1, keepdims=True)

dR = X_R - R_media_fila
dTau = Y_tau_ms - tau_media_fila

# Pendiente m y ordenada al origen b vectorizadas para las 5000 realizaciones
m_ms_por_ohm = np.sum(dR * dTau, axis=1) / np.sum(dR**2, axis=1)
b_ms = tau_media_fila.ravel() - m_ms_por_ohm * R_media_fila.ravel()

# Transformación a parámetros físicos: C (en uF) y R_s (en Ohm)
C_mc_uF = m_ms_por_ohm * 1000.0
Rs_mc_ohm = b_ms / m_ms_por_ohm

# Estadísticos de resumen: Mediana, IC 95% central, correlación y fracción Rs < 0
C_med = np.median(C_mc_uF)
C_ic_inf, C_ic_sup = np.percentile(C_mc_uF, [2.5, 97.5])
C_std = np.std(C_mc_uF, ddof=1)

Rs_med = np.median(Rs_mc_ohm)
Rs_ic_inf, Rs_ic_sup = np.percentile(Rs_mc_ohm, [2.5, 97.5])
Rs_std = np.std(Rs_mc_ohm, ddof=1)

# Coeficiente de correlación de Pearson entre C y R_s
correlacion_C_Rs = np.corrcoef(C_mc_uF, Rs_mc_ohm)[0, 1]

# Fracción de realizaciones con valores no físicos (R_s < 0 Ohm)
frac_Rs_negativo = 100.0 * np.mean(Rs_mc_ohm < 0.0)

# Ajuste puntual sobre los valores centrales (R_med, tau_hat_k) para calcular residuos de los 3 puntos
R_puntuales = np.array([
    RESISTENCIAS_EXP[33]['R_med'],
    RESISTENCIAS_EXP[100]['R_med'],
    RESISTENCIAS_EXP[220]['R_med'],
])
u_R_puntuales = np.array([
    RESISTENCIAS_EXP[33]['u_R'],
    RESISTENCIAS_EXP[100]['u_R'],
    RESISTENCIAS_EXP[220]['u_R'],
])
tau_puntuales = np.array([
    np.median(df_ciclos[df_ciclos['R_nom_ohm'] == 33]['tau_kj_prom_ms']),
    np.median(df_ciclos[df_ciclos['R_nom_ohm'] == 100]['tau_kj_prom_ms']),
    np.median(df_ciclos[df_ciclos['R_nom_ohm'] == 220]['tau_kj_prom_ms']),
])
m_central, b_central = np.polyfit(R_puntuales, tau_puntuales, 1)
residuos_recta_ms = tau_puntuales - (m_central * R_puntuales + b_central)

# =============================================================================
# 3.4 COMPARACIÓN: ¿QUÉ PASARÍA SI SOLO USÁRAMOS 2 PUNTOS (33 Y 100 OHM)? (Pregunta 9)
# =============================================================================
m_2ptos = (tau_100_mc - tau_33_mc) / (R_100_mc - R_33_mc)
b_2ptos = tau_33_mc - m_2ptos * R_33_mc
C_2ptos_uF = m_2ptos * 1000.0
Rs_2ptos_ohm = b_2ptos / m_2ptos

C_2ptos_med = np.median(C_2ptos_uF)
C_2ptos_ic = np.percentile(C_2ptos_uF, [2.5, 97.5])
Rs_2ptos_med = np.median(Rs_2ptos_ohm)
Rs_2ptos_ic = np.percentile(Rs_2ptos_ohm, [2.5, 97.5])

# Guardar Cuadro 4 en CSV
df_cuadro4 = pd.DataFrame([
    {
        'Magnitud': 'C del capacitor de 330 uF (con 3 resistencias)',
        'Estimacion_Mediana': round(C_med, 2),
        'IC_95%_Central': f'[{C_ic_inf:.2f}, {C_ic_sup:.2f}]',
        'Desv_Estandar_MC': round(C_std, 2),
        'Unidad': 'uF',
        'Metodo_Interpretacion': (
            f'Pendiente m de tau(R); corr(C, Rs) = {correlacion_C_Rs:.3f}'
        ),
    },
    {
        'Magnitud': 'R_s efectiva (con 3 resistencias)',
        'Estimacion_Mediana': round(Rs_med, 2),
        'IC_95%_Central': f'[{Rs_ic_inf:.2f}, {Rs_ic_sup:.2f}]',
        'Desv_Estandar_MC': round(Rs_std, 2),
        'Unidad': 'Ohm',
        'Metodo_Interpretacion': (
            f'Cociente b/m; {frac_Rs_negativo:.1f}% de realizaciones dan R_s'
            ' < 0 (no físico)'
        ),
    },
    {
        'Magnitud': '[Comparación Preg. 9] C usando solo 33 y 100 Ohm',
        'Estimacion_Mediana': round(C_2ptos_med, 2),
        'IC_95%_Central': f'[{C_2ptos_ic[0]:.2f}, {C_2ptos_ic[1]:.2f}]',
        'Desv_Estandar_MC': round(np.std(C_2ptos_uF, ddof=1), 2),
        'Unidad': 'uF',
        'Metodo_Interpretacion': (
            'Ajuste con solo 2 puntos (0 grados de libertad)'
        ),
    },
    {
        'Magnitud': '[Comparación Preg. 9] R_s usando solo 33 y 100 Ohm',
        'Estimacion_Mediana': round(Rs_2ptos_med, 2),
        'IC_95%_Central': f'[{Rs_2ptos_ic[0]:.2f}, {Rs_2ptos_ic[1]:.2f}]',
        'Desv_Estandar_MC': round(np.std(Rs_2ptos_ohm, ddof=1), 2),
        'Unidad': 'Ohm',
        'Metodo_Interpretacion': (
            'Extrapolación a R=0 sin el brazo de palanca de 215 Ohm'
        ),
    },
])
df_cuadro4.to_csv(ARCHIVO_SALIDA_CUADRO4, index=False)

print('\n=== CUADRO 4: PARÁMETROS INFERIDOS POR MONTECARLO (M = 5000) ===')
print(df_cuadro4.to_string(index=False))
print(f'\nCorrelación entre C y R_s: r = {correlacion_C_Rs:.4f}')
print(
    f'Porcentaje de realizaciones con R_s < 0 (no físico):'
    f' {frac_Rs_negativo:.2f}%'
)
print(
    'Residuos puntuales respecto de la recta (en ms):'
    f' R_33={residuos_recta_ms[0]:+.3f}, R_100={residuos_recta_ms[1]:+.3f},'
    f' R_220={residuos_recta_ms[2]:+.3f}'
)

# =============================================================================
# 4. FIGURA A: CONCEPTO Y RESULTADOS DE BOOTSTRAP POR CICLOS (ESTILO FIGURA 4)
# =============================================================================
fig_b, axes_b = plt.subplots(3, 3, figsize=(15, 10.5))
plt.subplots_adjust(hspace=0.40, wspace=0.28)
colores_ciclos = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

for row_idx, R_nom in enumerate([33, 100, 220]):
  sub = (
      df_ciclos[df_ciclos['R_nom_ohm'] == R_nom]
      .sort_values('Ciclo')
      .reset_index(drop=True)
  )
  taus_orig = sub['tau_kj_prom_ms'].values
  med_orig = np.median(taus_orig)
  remuestra_ej = ejemplo_una_remuestra[R_nom]
  med_remuestra = np.median(remuestra_ej)
  dist_boot = distribuciones_boot_ms[R_nom]
  ic_inf, ic_sup = np.percentile(dist_boot, [2.5, 97.5])

  # Columna 1: Muestra original (N = 5 ciclos)
  ax1 = axes_b[row_idx, 0]
  for j in range(5):
    ax1.plot(
        j + 1,
        taus_orig[j],
        'o',
        color=colores_ciclos[j],
        markersize=8,
        label=f'Ciclo {j+1}' if row_idx == 0 else None,
    )
  ax1.axhline(
      med_orig,
      color='black',
      ls='--',
      lw=1.5,
      label=f'Mediana original: {med_orig:.3f} ms',
  )
  ax1.set_title(
      f'1. Muestra Original ($N=5$, $R_{{nom}}={R_nom}\\,\\Omega$)',
      fontsize=9.5,
      fontweight='bold',
  )
  ax1.set_xlabel('Número de ciclo $j$')
  ax1.set_ylabel('$\\tau_{k,j}$ (ms)')
  ax1.set_xticks([1, 2, 3, 4, 5])
  ax1.legend(loc='best', fontsize=7.5)
  ax1.grid(True, alpha=0.3)

  # Columna 2: Ejemplo de una remuestra con reemplazo (N = 5)
  ax2 = axes_b[row_idx, 1]
  ax2.plot(
      range(1, 6),
      remuestra_ej,
      's--',
      color='#2ca02c',
      markersize=7,
      alpha=0.8,
  )
  ax2.axhline(
      med_remuestra,
      color='darkgreen',
      ls='--',
      lw=1.5,
      label=f'Mediana remuestra: {med_remuestra:.3f} ms',
  )
  ax2.set_title(
      f'2. Una Remuestra Bootstrap ($N=5$ con reemplazo)',
      fontsize=9.5,
      fontweight='bold',
  )
  ax2.set_xlabel('Posición en la remuestra')
  ax2.set_ylabel('$\\tau^*_{k,j}$ (ms)')
  ax2.set_xticks([1, 2, 3, 4, 5])
  ax2.set_ylim(ax1.get_ylim())
  ax2.legend(loc='best', fontsize=7.5)
  ax2.grid(True, alpha=0.3)

  # Columna 3: Distribución Bootstrap de las 5000 medianas
  ax3 = axes_b[row_idx, 2]
  valores_unicos, conteos = np.unique(
      np.round(dist_boot, 5), return_counts=True
  )
  frecuencias_rel = conteos / B_BOOT
  ancho_barra = max(
      (np.max(taus_orig) - np.min(taus_orig)) * 0.08, 0.002
  )
  ax3.bar(
      valores_unicos,
      frecuencias_rel,
      width=ancho_barra,
      color='#1f77b4',
      alpha=0.75,
      edgecolor='black',
  )
  ax3.axvline(
      med_orig,
      color='black',
      ls='--',
      lw=1.8,
      label=f'$\\hat{{\\tau}}_k = {med_orig:.3f}$ ms',
  )
  ax3.axvline(
      ic_inf,
      color='#d62728',
      ls=':',
      lw=1.8,
      label=f'IC 95%: [{ic_inf:.3f}, {ic_sup:.3f}]',
  )
  ax3.axvline(ic_sup, color='#d62728', ls=':', lw=1.8)
  ax3.set_title(
      f'3. Distribución Bootstrap ($B={B_BOOT}$)',
      fontsize=9.5,
      fontweight='bold',
  )
  ax3.set_xlabel('Mediana $\\hat{\\tau}^*_k$ (ms)')
  ax3.set_ylabel('Probabilidad')
  ax3.legend(loc='best', fontsize=7.5)
  ax3.grid(True, alpha=0.3)

plt.savefig(FIGURA_BOOTSTRAP, dpi=300, bbox_inches='tight')
plt.show()

# =============================================================================
# 5. FIGURA B: PROPAGACIÓN DE INCERTIDUMBRES POR MONTECARLO (ESTILO FIGURA 5)
# =============================================================================
fig_mc = plt.figure(figsize=(16, 8.5))
gs = fig_mc.add_gridspec(
    2, 3, width_ratios=[1.1, 1.25, 1.1], height_ratios=[1.3, 1.0],
    hspace=0.35, wspace=0.28
)

# Panel 1 (Izquierda arriba): Nubes de muestreo Montecarlo (R_k, tau_k)
ax_nube = fig_mc.add_subplot(gs[0, 0])
colores_r = {33: '#1f77b4', 100: '#ff7f0e', 220: '#9467bd'}
for idx_k, R_nom in enumerate([33, 100, 220]):
  ax_nube.scatter(
      X_R[:600, idx_k],
      Y_tau_ms[:600, idx_k],
      s=8,
      alpha=0.25,
      color=colores_r[R_nom],
  )
  ax_nube.errorbar(
      R_puntuales[idx_k],
      tau_puntuales[idx_k],
      xerr=u_R_puntuales[idx_k],
      fmt='o',
      color='black',
      markersize=6,
      capsize=4,
      label=f'$R_{{{R_nom}}}$: {RESISTENCIAS_EXP[R_nom]["etiqueta"]}',
  )
ax_nube.set_title(
    '1. Muestreo individual por punto\n'
    '$R_k \\sim \\mathcal{N}(R_{med}, u_R^2)$, $\\hat{\\tau}_k \\sim$ Bootstrap',
    fontsize=10,
    fontweight='bold',
)
ax_nube.set_xlabel('Resistencia externa $R$ ($\\Omega$)')
ax_nube.set_ylabel('Constante de tiempo $\\tau$ (ms)')
ax_nube.legend(loc='upper left', fontsize=7.8)
ax_nube.grid(True, alpha=0.3)

# Panel 2 (Izquierda abajo): Correlación conjunta entre C y R_s
ax_corr = fig_mc.add_subplot(gs[1, 0])
ax_corr.scatter(
    C_mc_uF[:1500], Rs_mc_ohm[:1500], s=6, alpha=0.2, color='teal'
)
ax_corr.axhline(0, color='#d62728', ls='--', lw=1.2, label='Límite físico $R_s = 0\\,\\Omega$')
ax_corr.plot(
    C_med,
    Rs_med,
    'o',
    color='black',
    markersize=7,
    label=f'Mediana ($r = {correlacion_C_Rs:.2f}$)',
)
ax_corr.set_title(
    f'Correlación entre parámetros ($r = {correlacion_C_Rs:.3f}$)',
    fontsize=10,
    fontweight='bold',
)
ax_corr.set_xlabel('Capacitancia $C$ ($\\mu$F)')
ax_corr.set_ylabel('Resistencia serie efectiva $R_s$ ($\\Omega$)')
ax_corr.legend(loc='upper right', fontsize=8)
ax_corr.grid(True, alpha=0.3)

# Panel 3 (Centro arriba): Haz de rectas Montecarlo tau = m*R + b
ax_rectas = fig_mc.add_subplot(gs[0, 1])
R_grid = np.linspace(0, 240, 200)
for i in range(250):
  ax_rectas.plot(
      R_grid,
      m_ms_por_ohm[i] * R_grid + b_ms[i],
      '-',
      color='#1f77b4',
      alpha=0.03,
  )

ax_rectas.plot(
    R_grid,
    m_central * R_grid + b_central,
    '-',
    color='navy',
    lw=2,
    label=(
        f'Recta central: $C={C_med:.1f}\\,\\mu$F,'
        f' $R_s={Rs_med:.2f}\\,\\Omega$'
    ),
)
ax_rectas.errorbar(
    R_puntuales,
    tau_puntuales,
    xerr=u_R_puntuales,
    fmt='o',
    color='#d62728',
    markersize=6,
    capsize=4,
    zorder=5,
    label='Datos $(R_k, \\hat{\\tau}_k)$',
)
ax_rectas.axvline(0, color='black', ls=':', lw=1)
ax_rectas.set_title(
    f'2. Ajuste de $M={M_MC}$ realizaciones\n$\\tau = m R + b$',
    fontsize=10,
    fontweight='bold',
)
ax_rectas.set_xlabel('Resistencia externa $R$ ($\\Omega$)')
ax_rectas.set_ylabel('Constante de tiempo $\\tau$ (ms)')
ax_rectas.legend(loc='upper left', fontsize=8)
ax_rectas.grid(True, alpha=0.3)

# Panel 4 (Centro abajo): Residuos del ajuste lineal (1 grado de libertad)
ax_res_lin = fig_mc.add_subplot(gs[1, 1], sharex=ax_rectas)
u_tau_proyectada_ms = (m_central * u_R_puntuales)
ax_res_lin.axhline(0, color='black', ls='--', lw=1.2)
ax_res_lin.errorbar(
    R_puntuales,
    residuos_recta_ms,
    yerr=u_tau_proyectada_ms,
    fmt='s',
    color='#2ca02c',
    markersize=7,
    capsize=5,
    lw=1.5,
    label='Residuo $\\hat{\\tau}_k - (m R_k + b) \\pm m\\,u(R_k)$',
)
ax_res_lin.set_title(
    'Residuos del modelo lineal (1 grado de libertad)',
    fontsize=10,
    fontweight='bold',
)
ax_res_lin.set_xlabel('Resistencia externa $R$ ($\\Omega$)')
ax_res_lin.set_ylabel('Residuo (ms)')
ax_res_lin.legend(loc='best', fontsize=8)
ax_res_lin.grid(True, alpha=0.3)

# Panel 5 (Derecha arriba): Histograma de Capacitancia C = m
ax_hist_c = fig_mc.add_subplot(gs[0, 2])
ax_hist_c.hist(
    C_mc_uF,
    bins=35,
    density=True,
    color='#4a90e2',
    edgecolor='white',
    alpha=0.85,
)
ax_hist_c.axvline(
    C_med,
    color='black',
    ls='--',
    lw=2,
    label=f'Mediana: {C_med:.1f} $\\mu$F',
)
ax_hist_c.axvline(
    C_ic_inf,
    color='#d62728',
    ls=':',
    lw=1.8,
    label=f'IC 95%: [{C_ic_inf:.1f}, {C_ic_sup:.1f}] $\\mu$F',
)
ax_hist_c.axvline(C_ic_sup, color='#d62728', ls=':', lw=1.8)
ax_hist_c.axvline(
    330.0,
    color='darkorange',
    ls='-.',
    lw=1.5,
    label='Nominal: 330 $\\mu$F',
)
ax_hist_c.set_title(
    '3. Parámetros inferidos: $C = m$', fontsize=10, fontweight='bold'
)
ax_hist_c.set_xlabel('Capacitancia $C$ ($\\mu$F)')
ax_hist_c.set_ylabel('Densidad de probabilidad')
ax_hist_c.legend(loc='upper right', fontsize=7.8)
ax_hist_c.grid(True, alpha=0.3)

# Panel 6 (Derecha abajo): Histograma de Resistencia serie efectiva R_s = b/m
ax_hist_rs = fig_mc.add_subplot(gs[1, 2])
ax_hist_rs.hist(
    Rs_mc_ohm,
    bins=35,
    density=True,
    color='#2ecc71',
    edgecolor='white',
    alpha=0.85,
)
ax_hist_rs.axvline(
    Rs_med,
    color='black',
    ls='--',
    lw=2,
    label=f'Mediana: {Rs_med:.2f} $\\Omega$',
)
ax_hist_rs.axvline(
    Rs_ic_inf,
    color='#d62728',
    ls=':',
    lw=1.8,
    label=f'IC 95%: [{Rs_ic_inf:.2f}, {Rs_ic_sup:.2f}] $\\Omega$',
)
ax_hist_rs.axvline(Rs_ic_sup, color='#d62728', ls=':', lw=1.8)
ax_hist_rs.axvline(
    0.0,
    color='purple',
    ls='-',
    lw=1.5,
    label=f'Límite $R_s=0$ ({frac_Rs_negativo:.1f}% $<0$)',
)
ax_hist_rs.set_title(
    'Resistencia serie efectiva: $R_s = b/m$', fontsize=10, fontweight='bold'
)
ax_hist_rs.set_xlabel('Resistencia serie $R_s$ ($\\Omega$)')
ax_hist_rs.set_ylabel('Densidad de probabilidad')
ax_hist_rs.legend(loc='upper right', fontsize=7.8)
ax_hist_rs.grid(True, alpha=0.3)

plt.savefig(FIGURA_MONTECARLO, dpi=300, bbox_inches='tight')
plt.show()

print(f'\nFigura Bootstrap guardada en:   {FIGURA_BOOTSTRAP}')
print(f'Figura Montecarlo guardada en:  {FIGURA_MONTECARLO}')
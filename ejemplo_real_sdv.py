import pandas as pd
import numpy as np
from datetime import datetime
import plotly.graph_objects as go

# 1. Definir un rango de fechas para la simulación
start_date = datetime(2025, 4, 1)
end_date = datetime(2025, 4, 30)
dates = pd.date_range(start=start_date, end=end_date, freq='h')

# 2. Simular una base de stock inicial
np.random.seed(43)
base_stock = np.random.randint(50, 150, size=len(dates))

# 3. Simular demanda base con patrón diario y ruido
horas = np.array([dt.hour for dt in dates])
demanda_diaria_base = 10 + 5 * np.sin(2 * np.pi * horas / 24)
ruido = np.random.normal(0, 2, size=len(dates))
base_demand = np.clip(demanda_diaria_base + ruido, 5, 20).astype(int)

# 4. Cargar eventos reales desde CSV (ahora incluye 'Popularidad')
eventos = pd.read_csv('eventos_futbol_abril2025.csv')
eventos['datetime_evento'] = pd.to_datetime(eventos['Fecha'] + ' ' + eventos['Hora'])
eventos['datetime_evento'] = eventos['datetime_evento'].dt.round('H')

# 5. Crear diccionarios de eventos
eventos_nombre = dict(zip(eventos['datetime_evento'], eventos['Evento']))
eventos_pop = dict(zip(eventos['datetime_evento'], eventos['Popularidad']))

# 6. Crear DataFrame base
data = pd.DataFrame({
    'timestamp': dates,
    'stock_inicial': base_stock,
    'base_demand': base_demand
})

# 7. Mapear eventos y popularidad
data['nombre_evento'] = data['timestamp'].map(eventos_nombre)
data['popularidad_evento'] = data['timestamp'].map(eventos_pop)
data['evento_real_futbol'] = data['nombre_evento'].notna()

# Usar la popularidad como multiplicador
# Normalizamos: popularidad mínima 1.0, máxima 5.0 → escala proporcional
data['multiplicador_futbol'] = data['popularidad_evento'].fillna(1)

# 8. Calcular demanda ajustada
ajuste_fuera_de_evento = np.where(data['evento_real_futbol'], 1.0, 0.85)
data['base_demand_ajustada'] = (data['base_demand'] * ajuste_fuera_de_evento).astype(int)
data['demanda_simulada'] = (data['base_demand_ajustada'] * data['multiplicador_futbol']).astype(int)
data['demanda_simulada'] = np.clip(data['demanda_simulada'], 0, data['stock_inicial'])

# 9. Calcular stock final
data['stock_final'] = data['stock_inicial'] - data['demanda_simulada']
data['stock_final'] = np.clip(data['stock_final'], 0, None)

# 10. SDV
from sdv.metadata import Metadata

metadata = Metadata.detect_from_dataframes({'mi_tabla': data})
print('Metadata detectada.')

from sdv.multi_table import HMASynthesizer
data_dict = {'mi_tabla': data}
synthesizer = HMASynthesizer(metadata)
synthesizer.fit(data_dict)
synthetic_data = synthesizer.sample(scale=1)

from sdv.evaluation.multi_table import run_diagnostic
diagnostic = run_diagnostic(
    real_data=data_dict,
    synthetic_data=synthetic_data,
    metadata=metadata
)

from sdv.evaluation.multi_table import evaluate_quality
quality_report = evaluate_quality(
    data_dict,
    synthetic_data,
    metadata
)
print("Evaluación de calidad:")
print(quality_report.get_details('Column Shapes'))

# 11. Visualización
data['timestamp_dt'] = pd.to_datetime(data['timestamp'])
full_range = pd.date_range(start=data['timestamp_dt'].min(), end=data['timestamp_dt'].max(), freq='h')
data = data.set_index('timestamp_dt').reindex(full_range).reset_index().rename(columns={'index': 'timestamp_dt'})

# Rellenar valores faltantes para visualización
data['demanda_simulada'] = data['demanda_simulada'].ffill()
data['evento_real_futbol'] = data['nombre_evento'].notna()

evento_futbol = data[data['evento_real_futbol']]
no_evento = data

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=no_evento['timestamp_dt'],
    y=no_evento['demanda_simulada'],
    mode='lines',
    name='Demanda simulada',
    line=dict(color='royalblue')
))

fig.add_trace(go.Scatter(
    x=evento_futbol['timestamp_dt'],
    y=evento_futbol['demanda_simulada'],
    mode='markers',
    name='Eventos de fútbol',
    text=evento_futbol['nombre_evento'] + " (Popularidad: " + evento_futbol['popularidad_evento'].astype(str) + ")",
    hoverinfo='text+x+y',
    marker=dict(color='red', size=6, symbol='circle')
))

fig.update_layout(
    title='Demanda de cerveza de un Local - Abril 2025',
    xaxis_title='Fecha',
    yaxis_title='Demanda de cerveza',
    template='simple_white',
    xaxis=dict(tickformat="%d/%m", dtick="D1")
)
# mostrar la gráfica
fig.show()

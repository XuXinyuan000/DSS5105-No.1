import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Load your data
data = pd.read_csv('../data/Metlife_score.csv')
data['Year'] = pd.to_datetime(data['Year'], format='%Y')
data.set_index('Year', inplace=True)

# Forecast future values for each score
forecast_results = {}
for column in ['Score E', 'Score S', 'Score G', 'Total ESG Score']:
    model = ExponentialSmoothing(
        data[column],
        trend='add',  # Adding a trend component
        seasonal=None  # No seasonal component
    ).fit()
    forecast = model.forecast(steps=3)
    forecast_results[column] = forecast

# Save results
forecast_df = pd.DataFrame(forecast_results)
forecast_df.index = [data.index.year[-1] + i + 1 for i in range(3)]
forecast_df.to_csv('forecast_results_corrected.csv')
forecast_df

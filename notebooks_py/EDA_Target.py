# %%
%pylab inline

import pandas as pd
import feather
import gc
import ipywidgets as widget
import plotly.io as pio
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from tqdm import tqdm

plt.style.use("ggplot")
pio.renderers.default = "notebook"
%config InlineBackend.figure_format='retina'

# %% [markdown]
# # Cargar todos los datos

# %%
df = feather.read_dataframe("train_downsampled.feather")
assets = df.investment_id.unique()

# %%
def plot_column(df, col = "target"):
    fig = make_subplots(rows = 1, cols = 1)
    fig.add_trace(
        go.Scatter(x = df.time_id, y = df[col]),
        row = 1, col = 1
    )
    fig.update_layout(showlegend=False, xaxis_rangeslider_visible=True)
    fig.show()

# %%
@widget.interact
def plot_asset_returns(column = widget.fixed("[investment_id]"), asset = assets):
    asset_df = df.query("investment_id == @asset")
    print("Cantidad de registros para", asset, ";", asset_df.shape[0])
    plot_column(asset_df)

# %% [markdown]
# # Cantidad de mediciones por time_id

# %%
df_temp = df[["time_id", "investment_id"]].copy()

fmin ,fmax = df_temp.time_id.agg([min, max])
df_temp["time_diff"] = df_temp.time_id - fmin

# Agrupar
g = df_temp.groupby(["investment_id"])
plt.figure(figsize = (12, 8))
ax = plt.scatter(g.time_diff.mean(), g.size(), edgecolor = "none", alpha = 0.5)
plt.title(f"Cantidad de datos por periodo entre {fmin} y {fmax}");

# %%
del df_temp, g, ax, fmin, fmax; gc.collect();

# %% [markdown]
# # Agrupar assets por volatilidad

# %%
returns = df[["time_id", "investment_id", "target"]].copy()
returns = pd.pivot_table(returns, 
                         values = "target", 
                         index = "time_id", 
                         columns = "investment_id")
returns

# %% [markdown]
# ## Distribución
# 
# Me tinca que deberían ser t-students

# %% [markdown]
# ### Test de hipótesis: Kolmogorov Smirnov, Lilliefors, Shapiro, t - test

# %%
# TODO:
from scipy.stats import kstest
from statsmodels.stats.diagnostic import lilliefors

# %% [markdown]
# ## Estacionariedad

# %%
from statsmodels.tsa.stattools import adfuller
from collections import defaultdict

def make_stationary(data: pd.Series, alpha: float = 0.05, max_diff_order: int = 10) -> dict:
    # Test to see if the time series is already stationary
    if adfuller(data)[1] < alpha:
        return {
            'differencing_order': 0,
            'time_series': np.array(data)
        }
    
    # A list to store P-Values
    p_values = []
    
    # Test for differencing orders from 1 to max_diff_order (included)
    for i in range(1, max_diff_order + 1):
        # Perform ADF test
        result = adfuller(data.diff(i).dropna())
        # Append P-value
        p_values.append((i, result[1]))
        
    # Keep only those where P-value is lower than significance level
    significant = [p for p in p_values if p[1] < alpha]
    # Sort by the differencing order
    significant = sorted(significant, key=lambda x: x[0])
    
    # Get the differencing order
    diff_order = significant[0][0]
    
    # Make the time series stationary
    stationary_series = data.diff(diff_order).dropna()
    
    return {
        'differencing_order': diff_order,
        'time_series': np.array(stationary_series)
    }

orders = defaultdict(lambda : "Asset no presente")
dificiles = defaultdict(lambda : "Todas son estacionarias")
for asset in tqdm(assets):
    try:
        orders[asset] = make_stationary(returns[asset].dropna())['differencing_order']
        if orders[asset] > 0: 
            orders[asset]
    except:
        print("No es posible calcular para asset id:", asset)

# %%
pd.Series(orders).value_counts() # La mayoría es estacionaria

# %%
dificiles

# %% [markdown]
# ## Clustering

# %% [markdown]
# ### UMAP

# %%
# TODO:
import umap

reducer = umap.UMAP(n_components = 3)
# El problema son los NaNs
embedding = reducer.fit_transform(returns)

# %% [markdown]
# ### Por volatilidad

# %% [markdown]
# 



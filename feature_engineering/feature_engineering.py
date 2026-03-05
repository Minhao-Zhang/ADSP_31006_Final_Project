# %%
import pandas as pd
import numpy as np
import re
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


pd.set_option('display.max_columns', 200)
pd.set_option('display.width', 180)

# %%
# Paths and config
ROOT = Path.cwd()
DATA = '../data/raw'
OUT_PARQUET = Path(DATA) / '../processed/merged_features_weekly.parquet'

WEEKLY_FREQ = 'W-FRI'
LAGS = [1, 2, 3, 4, 8, 12]
ROLL_WINDOWS = [4, 8]
TEST_START = pd.Timestamp('2025-01-03')  # first Friday in 2025

TARGET_FILE = Path(DATA) / 'cleaned_fruit_veg.csv'
API_FILE = Path(DATA) / 'API_20260129.csv'
FUEL_FILES = [
    Path(DATA) / 'weekly_road_fuel_prices_2003_to_2017.csv',
    Path(DATA) / 'weekly_road_fuel_prices_2018_to_now.csv',
]
SPPI_FILE = Path(DATA) / 'series-210226.csv'

assert TARGET_FILE.exists(), f'Missing {TARGET_FILE}'
assert API_FILE.exists(), f'Missing {API_FILE}'
assert SPPI_FILE.exists(), f'Missing {SPPI_FILE}'
for f in FUEL_FILES:
    assert f.exists(), f'Missing {f}'

print('Using data directory:', DATA)


# %% [markdown]
# ## 1) Load target series (weekly commodity prices)

# %%
target = pd.read_csv(TARGET_FILE)
target['date'] = pd.to_datetime(target['date'])
target = target.rename(
    columns={'display_name': 'commodity', 'price': 'target_price'})

target['date'] = target['date'] + \
    pd.to_timedelta((4 - target['date'].dt.dayofweek) % 7, unit='D')

target = target[['date', 'commodity', 'target_price']].copy()
target = target.sort_values(['commodity', 'date']).reset_index(drop=True)

print('Target shape:', target.shape)
print('Date range:', target['date'].min().date(),
      'to', target['date'].max().date())
print('Commodity count:', target['commodity'].nunique())
target.head()

# %%
SELECTED = sorted([
    'Bulb Onions (Yellow)',
    'Cabbage',
    'Carrots',
    'Lettuce',
])

target = target[target['commodity'].isin(SELECTED)]

# %%
# 1b) Reindex each commodity to a strict W-FRI grid and fill gaps
# ---------------------------------------------------------------
# Strategy differs by period to prevent leakage:
#   Training gaps  → linear interpolation (bidirectional, fine for lag continuity;
#                    we never evaluate on these interpolated rows)
#   Test gaps      → forward-fill only (causal: no future observation used)
#
# target_was_missing=1 marks every inserted week in both periods.
# Evaluation code downstream must exclude target_was_missing=1 rows in the test set.

_start = target['date'].min()
_end = target['date'].max()
_full_idx = pd.date_range(start=_start, end=_end, freq=WEEKLY_FREQ)

parts = []
for comm, grp in target.groupby('commodity'):
    grp = grp.set_index('date').reindex(_full_idx)
    grp['commodity'] = comm
    grp['target_was_missing'] = grp['target_price'].isna().astype('int8')

    original = grp['target_price'].copy()

    # Linear fill (for training portion)
    linear_filled = grp['target_price'].interpolate(
        method='linear').bfill().ffill()

    # Causal fill (for test portion — only carry last known value forward)
    causal_filled = grp['target_price'].ffill().bfill()

    # Combine: use linear in train, causal in test
    grp['target_price'] = linear_filled
    test_interpolated = (grp.index >= TEST_START) & original.isna()
    grp.loc[test_interpolated, 'target_price'] = causal_filled[test_interpolated]

    parts.append(grp)

target = (
    pd.concat(parts)
    .rename_axis('date')
    .reset_index()
    .sort_values(['commodity', 'date'])
    .reset_index(drop=True)
)

train_filled = target.loc[target['date'] <
                          TEST_START, 'target_was_missing'].sum()
test_filled = target.loc[target['date'] >=
                         TEST_START, 'target_was_missing'].sum()
print(f'Target rows after reindex      : {len(target)}')
print(f'Weeks filled — train (linear)  : {train_filled}')
print(
    f'Weeks filled — test  (ffill)   : {test_filled}  ← excluded from eval metrics')
target.head(3)

# %% [markdown]
# ## 2) Load and shape exogenous inputs

# %%
# API monthly indices
api = pd.read_csv(API_FILE)
api['date'] = pd.to_datetime(api['date'])

api_wide = api.pivot_table(
    index='date', columns='category', values='index', aggfunc='first').sort_index()

api_keep = [
    'energy_and_lubricants',
    'fertilisers_and_soil_improvers',
    'seeds',
    'plant_protection_products',
    'fresh_fruit',
    'fresh_vegetables',
]
api_keep = [c for c in api_keep if c in api_wide.columns]
api_wide = api_wide[api_keep].copy()
api_wide.columns = [f'api_{c}' for c in api_wide.columns]

print('API shape:', api_wide.shape)
api_wide.tail(3)

# %%
# Weekly road fuel prices (robust column mapping)
fuel_raw = pd.concat([pd.read_csv(f) for f in FUEL_FILES], ignore_index=True)


def _pick_col(cols, patterns):
    cols_l = {c.lower(): c for c in cols}
    for p in patterns:
        for lc, orig in cols_l.items():
            if re.search(p, lc):
                return orig
    return None


date_col = _pick_col(fuel_raw.columns, [r'^date$', r'date'])
petrol_col = _pick_col(fuel_raw.columns, [r'ulsp', r'petrol'])
diesel_col = _pick_col(fuel_raw.columns, [r'ulsd', r'diesel'])

fuel = fuel_raw[[date_col, petrol_col, diesel_col]].copy()
fuel.columns = ['date', 'fuel_petrol_price', 'fuel_diesel_price']
fuel['date'] = pd.to_datetime(fuel['date'], errors='coerce')
for c in ['fuel_petrol_price', 'fuel_diesel_price']:
    fuel[c] = pd.to_numeric(fuel[c], errors='coerce')

fuel = fuel.dropna(subset=['date']).sort_values('date').drop_duplicates('date')
fuel = fuel.set_index('date')

print('Fuel shape:', fuel.shape)
fuel.tail(3)

# %%
# ONS SPPI CSV parser (extracts quarter values, then upsamples)
sppi_raw = pd.read_csv(SPPI_FILE, header=None)
quarter_rows = []

for i in range(len(sppi_raw)):
    row_vals = [str(v).strip() for v in sppi_raw.iloc[i].dropna().tolist()]
    if len(row_vals) < 2:
        continue
    label = row_vals[0]
    val = row_vals[-1]
    m = re.match(r'^(\d{4})\s*Q([1-4])$', label)
    if not m:
        continue
    year = int(m.group(1))
    q = int(m.group(2))
    month = q * 3
    date = pd.Timestamp(year=year, month=month, day=1) + pd.offsets.MonthEnd(0)
    value = pd.to_numeric(val, errors='coerce')
    if pd.notna(value):
        quarter_rows.append((date, value))

sppi = pd.DataFrame(quarter_rows, columns=[
                    'date', 'sppi_road_freight']).drop_duplicates('date')
sppi = sppi.sort_values('date').set_index('date')

print('SPPI shape:', sppi.shape)
sppi.tail(3)

# %% [markdown]
# ## 3) Weekly alignment and merge

# %%
# Build shared weekly timeline anchored to the fruit/veg price dates (W-FRI)
start_date = target['date'].min()
end_date = target['date'].max()
weekly_idx = pd.date_range(start=start_date, end=end_date, freq=WEEKLY_FREQ)


def resample_to_weekly(df, weekly_idx):
    """
    Correctly forward-fill a lower/different-frequency series onto a W-FRI index.

    Plain reindex(weekly_idx) only matches exact dates, so monthly (1st of month),
    quarterly (quarter-end), or Monday-weekly series produce almost entirely NaN
    before ffill has anything to propagate. The fix: union the source dates with
    the target weekly dates first, ffill across the combined index, then select
    only the weekly dates.
    """
    combined = df.index.union(weekly_idx)
    return df.reindex(combined).ffill().reindex(weekly_idx)


# Track pre-fill missingness: 1 where no new source observation fell in that week
# NaN where no monthly date matched a Friday
api_w_raw = api_wide.reindex(weekly_idx)
# NaN where no Monday matched a Friday
fuel_w_raw = fuel.reindex(weekly_idx)
# NaN where no quarter-end matched a Friday
sppi_w_raw = sppi.reindex(weekly_idx)

exog_missing_flags = pd.concat([
    api_w_raw.isna().add_suffix('_was_missing'),
    fuel_w_raw.isna().add_suffix('_was_missing'),
    sppi_w_raw.isna().add_suffix('_was_missing'),
], axis=1).astype('int8')

# Resample each series to W-FRI using the union+ffill approach
api_w = resample_to_weekly(api_wide, weekly_idx)
fuel_w = resample_to_weekly(fuel,     weekly_idx)
sppi_w = resample_to_weekly(sppi,     weekly_idx)

exog_weekly = pd.concat([api_w, fuel_w, sppi_w, exog_missing_flags], axis=1)
exog_weekly.index.name = 'date'
exog_weekly = exog_weekly.reset_index()

merged = target.merge(exog_weekly, on='date', how='left')
merged = merged.sort_values(['commodity', 'date']).reset_index(drop=True)

print('Merged shape:', merged.shape)
print('API non-null after fix:',
      merged['api_energy_and_lubricants'].notna().sum(), '/', len(merged))
print('Fuel non-null after fix:',
      merged['fuel_petrol_price'].notna().sum(), '/', len(merged))
print('SPPI non-null after fix:',
      merged['sppi_road_freight'].notna().sum(), '/', len(merged))
merged.head(3)

# %% [markdown]
# ## 4) Time features and intervention flags

# %%
iso = merged['date'].dt.isocalendar()
merged['year'] = merged['date'].dt.year
merged['month'] = merged['date'].dt.month
merged['quarter'] = merged['date'].dt.quarter
merged['weekofyear'] = iso.week.astype(int)

# Cyclical encoding for week-of-year
merged['week_sin'] = np.sin(2 * np.pi * merged['weekofyear'] / 52.0)
merged['week_cos'] = np.cos(2 * np.pi * merged['weekofyear'] / 52.0)

# Structural break window for energy shock
shock_start = pd.Timestamp('2021-10-01')
shock_end = pd.Timestamp('2023-03-31')
merged['shock_2021q4_2023q1'] = ((merged['date'] >= shock_start) & (
    merged['date'] <= shock_end)).astype('int8')
merged['post_shock'] = (merged['date'] > shock_end).astype('int8')

merged[['date', 'commodity', 'weekofyear', 'week_sin',
        'week_cos', 'shock_2021q4_2023q1']].head(3)

# %% [markdown]
# ## 5) Lag, rolling, and YoY features

# %%
# Shared exogenous columns (numeric, non-target)
exclude_cols = {'date', 'commodity', 'target_price', 'target_was_missing'}
num_cols = [
    c for c in merged.columns if c not in exclude_cols and pd.api.types.is_numeric_dtype(merged[c])]

# Commodity-level target dynamics
g = merged.groupby('commodity', group_keys=False)
for lag in LAGS:
    merged[f'target_lag_{lag}'] = g['target_price'].shift(lag)

for w in ROLL_WINDOWS:
    merged[f'target_roll_mean_{w}'] = g['target_price'].shift(1).rolling(
        w, min_periods=1).mean().reset_index(level=0, drop=True)
    merged[f'target_roll_std_{w}'] = g['target_price'].shift(1).rolling(
        w, min_periods=2).std().reset_index(level=0, drop=True)

merged['target_yoy_diff_52'] = g['target_price'].diff(52)
merged['target_yoy_pct_52'] = g['target_price'].pct_change(52)

# Exogenous lags and rolling stats (same for all commodities, merged row-wise)
exog_numeric = [c for c in num_cols if not c.endswith('_was_missing')]
for c in exog_numeric:
    for lag in LAGS:
        merged[f'{c}_lag_{lag}'] = merged[c].shift(lag)
    for w in ROLL_WINDOWS:
        merged[f'{c}_roll_mean_{w}'] = merged[c].shift(
            1).rolling(w, min_periods=1).mean()
    merged[f'{c}_yoy_diff_52'] = merged[c].diff(52)

print('Feature engineering complete. Columns:', len(merged.columns))

# %% [markdown]
# ## 6) Final modeling table and split

# %%
# Keep rows with observed target only, then drop early rows lacking lags
model_df = merged.dropna(subset=['target_price']).copy()

# Minimum lag requirement (12-week lag + 52-week YoY implies ~52 initial rows dropped per commodity)
required_cols = [
    'target_lag_1', 'target_lag_4', 'target_lag_12',
    'target_yoy_diff_52',
]
model_df = model_df.dropna(subset=required_cols).reset_index(drop=True)

# Train/test flag for downstream evaluation
model_df['split'] = np.where(model_df['date'] < TEST_START, 'train', 'test')

# NOTE: target_was_missing=1 marks weeks whose price was not directly observed.
# When computing test-set metrics, filter to target_was_missing=0 to avoid
# evaluating against forward-filled (non-observed) values.
test_eval_rows = ((model_df['split'] == 'test') & (
    model_df['target_was_missing'] == 0)).sum()

print('Final table shape:', model_df.shape)
print('Train rows       :', (model_df['split'] == 'train').sum())
print('Test rows (total):', (model_df['split'] == 'test').sum())
print('Test rows (eval) :', test_eval_rows, ' ← use these for metrics')
print('Commodities      :', model_df['commodity'].nunique())

model_df.head(3)

# %%
# Save outputs

try:
    model_df.to_parquet(OUT_PARQUET, index=False)
    parquet_msg = f'Saved: {OUT_PARQUET}'
except Exception as e:
    parquet_msg = f'Parquet not saved ({type(e).__name__}: {e})'

print(parquet_msg)


# %% [markdown]
# ## 7) Quick quality checks

# %%
summary = {
    'rows': len(model_df),
    'columns': len(model_df.columns),
    'date_min': model_df['date'].min(),
    'date_max': model_df['date'].max(),
    'commodities': model_df['commodity'].nunique(),
}
print(summary)

missing_top = model_df.isna().mean().sort_values(ascending=False).head(15)
missing_top

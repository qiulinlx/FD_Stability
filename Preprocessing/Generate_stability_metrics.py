import statsmodels.api as sm
import pandas as pd

df = pd.read_csv('data/merged_EVI_PID.csv')

df["year"] = pd.to_datetime(df["year"].astype(int), format="%Y")
df["year"] = df["year"].dt.year

# Ensure correct types
df["year"] = df["year"].astype(float)
df["EVI"] = df["EVI_mean"].astype(float)

# Container for results
records = []

# Process each plot
for pid, sub in df.groupby("PID"):

    if len(sub) < 3:
        # Not enough years to fit regression
        continue

    # --- Fit EVI ~ Year ---
    X = sm.add_constant(sub["year"])    # adds intercept
    y = sub["EVI"]

    model = sm.OLS(y, X).fit()

    # Residuals
    residuals = model.resid
    residual_sd = residuals.std()

    # Store
    records.append({
        "PID": pid,
        "n_years": len(sub),
        "slope": model.params["year"],
        "intercept": model.params["const"],
        "residual_sd": residual_sd
    })

records= pd.DataFrame(records)
records.to_csv('data/stability_metrics.csv', index=False)
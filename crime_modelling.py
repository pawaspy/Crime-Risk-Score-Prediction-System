import os, math
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
import xgboost as xgb
import lightgbm as lgb
import joblib

DATA_PATH = "crime.csv"
assert os.path.exists(DATA_PATH), f"{DATA_PATH} not found."

df = pd.read_csv(DATA_PATH, low_memory=False)
print("=" * 70)
print("🚨 DELHI CRIME PREDICTION - ULTRA-DIVERSE CRIME MAPPING")
print("=" * 70)
print(f"\n📍 Loaded {len(df)} police stations from Delhi")

crime_columns = ['murder', 'rape', 'gangrape', 'robbery', 'theft', 'assualt murders', 'sexual harassement']
crime_columns = [col for col in crime_columns if col in df.columns]

print(f"\n📊 Original Crime Statistics:")
crime_totals = {}
for crime in crime_columns:
    total = df[crime].sum()
    crime_totals[crime] = total
    print(f"  {crime:25s}: {total:>6,} incidents")

max_count = max(crime_totals.values())
min_count = min([v for v in crime_totals.values() if v > 0])

weights = {}
for crime in crime_columns:
    if crime_totals[crime] > 0:
        ratio = max_count / crime_totals[crime]
        weights[crime] = min(20.0, math.sqrt(ratio) * 3.0)
    else:
        weights[crime] = 1.0

# Manual override for specific crimes to ensure visibility
weights['murder'] = 20.0 
weights['rape'] = 15.0
weights['gangrape'] = 18.0
weights['robbery'] = 8.0
weights['sexual harassement'] = 15.0
weights['assualt murders'] = 12.0
weights['theft'] = 1.0 

print(f"\n⚖️  Ultra-Aggressive Crime Weighting:")
for crime in sorted(weights.keys(), key=lambda x: weights[x], reverse=True):
    print(f"  {crime:25s}: {weights[crime]:>6.1f}x boost")

# Create weighted records
records = []
np.random.seed(42)

for idx, row in df.iterrows():
    lat = row['lat']
    lon = row['long']
    area = row['nm_pol']
    
    for crime_type in crime_columns:
        count = int(row[crime_type])
        if count > 0:
            # Apply aggressive weight
            adjusted_count = max(1, int(count * weights[crime_type]))
            
            for _ in range(adjusted_count):
                records.append({
                    'lat': lat,
                    'lon': lon,
                    'area': area,
                    'crime_type': crime_type,
                    'hour': np.random.randint(0, 24)
                })

df_expanded = pd.DataFrame(records)
print(f"\n✅ Created {len(df_expanded):,} weighted crime records")

# Show new distribution
print(f"\n🎯 Weighted Crime Distribution:")
dist = df_expanded['crime_type'].value_counts()
for crime in crime_columns:
    if crime in dist.index:
        count = dist[crime]
        pct = (count / len(df_expanded)) * 100
        print(f"  {crime:25s}: {count:>7,} records ({pct:>5.1f}%)")

# Create larger spatial grid for more mixing
GRID_SIZE = 0.02  # ~2km grid
df_expanded["lat_grid"] = (df_expanded["lat"] / GRID_SIZE).round(0) * GRID_SIZE
df_expanded["lon_grid"] = (df_expanded["lon"] / GRID_SIZE).round(0) * GRID_SIZE

print(f"\n🗺️  Creating {GRID_SIZE}° spatial grid cells...")

# For each grid, select DIVERSE top crime (not just mode)
def get_diverse_top_crime(crime_series):
    """Get the top crime, but prefer non-theft if it's close"""
    counts = crime_series.value_counts()
    if len(counts) == 0:
        return "theft"
    
    top_crime = counts.index[0]
    top_count = counts.iloc[0]
    
    # If theft is top but another crime has >40% of its count, use that instead
    if top_crime == "theft" and len(counts) > 1:
        second_crime = counts.index[1]
        second_count = counts.iloc[1]
        if second_count / top_count > 0.4:
            return second_crime
    
    return top_crime

# Group and aggregate
group = df_expanded.groupby(["lat_grid", "lon_grid"])

agg = group.agg(
    total_crimes=("hour", "count"),
    unique_crime_types=("crime_type", "nunique"),
    top_crime_type=("crime_type", get_diverse_top_crime),
).reset_index()

# Hourly stats
agg_hours = group["hour"].agg(
    mean_hour="mean",
    std_hour="std",
    night_prop=lambda x: ((x >= 0) & (x <= 6)).sum() / len(x) if len(x) > 0 else 0
).reset_index()

agg = agg.merge(agg_hours, on=["lat_grid", "lon_grid"], how="left")
agg.fillna({"mean_hour": 12, "std_hour": 0, "night_prop": 0}, inplace=True)

# Risk features
agg["risk_score"] = agg["total_crimes"] / agg["total_crimes"].max()
agg["risk_type"] = pd.qcut(
    agg["risk_score"].rank(method="first"), q=3, labels=["low", "medium", "high"]
)

print(f"\n✅ Created {len(agg)} grid cells")
print(f"\n🎯 Final Crime Type Distribution in Grid Cells:")
crime_dist = agg['top_crime_type'].value_counts()
for crime in crime_dist.index:
    count = crime_dist[crime]
    pct = (count / len(agg)) * 100
    bar = "█" * int(pct / 2)
    print(f"  {crime:25s}: {count:>3d} cells ({pct:>5.1f}%) {bar}")

# Check diversity
unique_crimes = agg['top_crime_type'].nunique()
print(f"\n🌈 Crime Type Diversity: {unique_crimes}/{len(crime_columns)} types present")

if unique_crimes < 5:
    print("⚠️  Warning: Low diversity detected. Consider running with different grid size.")

# Preprocessing
le_top = LabelEncoder()
agg["top_crime_code"] = le_top.fit_transform(agg["top_crime_type"].astype(str))

print(f"\n🔢 Label Encoding:")
for i, crime in enumerate(le_top.classes_):
    print(f"  {i}: {crime}")

# Prepare features
feature_cols = ["total_crimes", "unique_crime_types", "mean_hour", "std_hour", "night_prop", "top_crime_code"]
X = agg[feature_cols].fillna(0)
y_reg = agg["risk_score"].values
y_clf = LabelEncoder().fit_transform(agg["risk_type"].astype(str))

scaler = StandardScaler()
X_num = scaler.fit_transform(X.drop(columns=["top_crime_code"]))
X_prepared = np.hstack([X_num, X[["top_crime_code"]].values])

X_train, X_test, yreg_train, yreg_test, yclf_train, yclf_test = train_test_split(
    X_prepared, y_reg, y_clf, test_size=0.2, random_state=42
)

# Train models
print("\n" + "=" * 70)
print("🤖 TRAINING MACHINE LEARNING MODELS")
print("=" * 70)

# Regression
print("\n📈 Regression Models (Risk Score):")
dt = DecisionTreeRegressor(random_state=42, max_depth=6)
dt.fit(X_train, yreg_train)

xgb_model = xgb.XGBRegressor(objective="reg:squarederror", random_state=42, n_jobs=4, verbosity=0)
xgb_param = {"n_estimators":[50,100,200], "max_depth":[3,5,7], "learning_rate":[0.01,0.05,0.1], "subsample":[0.6,0.8,1.0]}
xgb_search = RandomizedSearchCV(xgb_model, xgb_param, n_iter=6, cv=3, scoring="neg_mean_squared_error", random_state=42, verbose=0)
xgb_search.fit(X_train, yreg_train)
xgb_best = xgb_search.best_estimator_

lgb_model = lgb.LGBMRegressor(random_state=42, verbose=-1)
lgb_param = {"n_estimators":[50,100,200], "num_leaves":[15,31,63], "learning_rate":[0.01,0.05,0.1]}
lgb_search = RandomizedSearchCV(lgb_model, lgb_param, n_iter=6, cv=3, scoring="neg_mean_squared_error", random_state=42, verbose=0)
lgb_search.fit(X_train, yreg_train)
lgb_best = lgb_search.best_estimator_

def eval_reg(model, name):
    yp = model.predict(X_test)
    rmse = math.sqrt(mean_squared_error(yreg_test, yp))
    r2 = r2_score(yreg_test, yp)
    print(f"  {name:20s}: RMSE={rmse:.4f}, R²={r2:.4f}")

eval_reg(dt, "DecisionTree")
eval_reg(xgb_best, "XGBoost")
eval_reg(lgb_best, "LightGBM")

# Classification
print("\n🎯 Classification Models (Risk Category):")
dtc = DecisionTreeClassifier(random_state=42, max_depth=6)
dtc.fit(X_train, yclf_train)
print(f"  {'DecisionTree':20s}: Acc={accuracy_score(yclf_test, dtc.predict(X_test)):.4f}")

xgbc = xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric="mlogloss", verbosity=0)
xgbc_param = {"n_estimators":[50,100], "max_depth":[3,5], "learning_rate":[0.05,0.1]}
xgbc_search = RandomizedSearchCV(xgbc, xgbc_param, n_iter=4, cv=3, random_state=42, verbose=0)
xgbc_search.fit(X_train, yclf_train)
xgbc_best = xgbc_search.best_estimator_
print(f"  {'XGBoost':20s}: Acc={accuracy_score(yclf_test, xgbc_best.predict(X_test)):.4f}")

lgbc = lgb.LGBMClassifier(random_state=42, verbose=-1)
lgbc_param = {"n_estimators":[50,100], "num_leaves":[15,31], "learning_rate":[0.05,0.1]}
lgbc_search = RandomizedSearchCV(lgbc, lgbc_param, n_iter=4, cv=3, random_state=42, verbose=0)
lgbc_search.fit(X_train, yclf_train)
lgbc_best = lgbc_search.best_estimator_
print(f"  {'LightGBM':20s}: Acc={accuracy_score(yclf_test, lgbc_best.predict(X_test)):.4f}")

# Save everything
print("\n" + "=" * 70)
print("💾 SAVING MODELS AND DATA")
print("=" * 70)

out_dir = Path("models")
out_dir.mkdir(exist_ok=True)

joblib.dump(dt, out_dir/"decision_tree_reg.joblib")
joblib.dump(xgb_best, out_dir/"xgboost_reg.joblib")
joblib.dump(lgb_best, out_dir/"lightgbm_reg.joblib")
joblib.dump(dtc, out_dir/"decision_tree_clf.joblib")
joblib.dump(xgbc_best, out_dir/"xgboost_clf.joblib")
joblib.dump(lgbc_best, out_dir/"lightgbm_clf.joblib")
joblib.dump(scaler, out_dir/"scaler.joblib")
joblib.dump(le_top, out_dir/"le_top.joblib")

agg.to_csv(out_dir / "grid_aggregated.csv", index=False)

print(f"\n✅ All files saved to: {out_dir}")
print(f"✅ Grid cells: {len(agg)}")
print(f"✅ Crime types available: {list(le_top.classes_)}")
print(f"✅ Diversity score: {unique_crimes}/{len(crime_columns)}")

print("\n" + "=" * 70)
print("🎉 PIPELINE COMPLETE - MAXIMUM CRIME DIVERSITY ACHIEVED!")
print("=" * 70)
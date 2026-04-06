import fastf1
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# -------------------------------
# CONFIG
# -------------------------------
YEAR = 2023
RACE = 'Monza'
SIMULATIONS = 500
CACHE_DIR = 'cache'

fastf1.Cache.enable_cache(CACHE_DIR)

# -------------------------------
# LOAD DATA (QUALI + RACE)
# -------------------------------
def load_event(year, race):
    quali = fastf1.get_session(year, race, 'Q')
    race_sess = fastf1.get_session(year, race, 'R')

    quali.load()
    race_sess.load()

    q_results = quali.results
    r_results = race_sess.results

    df = pd.DataFrame({
        'driver': q_results['Abbreviation'],
        'team': q_results['TeamName'],
        'quali_pos': q_results['Position'],
        'quali_time': q_results['Q3'].fillna(q_results['Q2']).fillna(q_results['Q1'])
    })

    df['race_pos'] = r_results['Position'].values

    return df

# -------------------------------
# FEATURE ENGINEERING
# -------------------------------
def engineer_features(df):
    # Convert timedelta to seconds
    df['quali_time_sec'] = df['quali_time'].dt.total_seconds()

    # Normalize pace (relative to fastest)
    fastest = df['quali_time_sec'].min()
    df['pace_delta'] = df['quali_time_sec'] - fastest

    return df

# -------------------------------
# ENCODING
# -------------------------------
def encode(df):
    le_driver = LabelEncoder()
    le_team = LabelEncoder()

    df['driver_enc'] = le_driver.fit_transform(df['driver'])
    df['team_enc'] = le_team.fit_transform(df['team'])

    return df, le_driver, le_team

# -------------------------------
# TRAIN MODEL
# -------------------------------
def train(df):
    features = ['quali_pos', 'pace_delta', 'driver_enc', 'team_enc']
    X = df[features]
    y = df['race_pos']

    model = RandomForestRegressor(n_estimators=200)
    model.fit(X, y)

    return model, features

# -------------------------------
# MONTE CARLO SIMULATION
# -------------------------------
def simulate_race(df, model, features, sims=1000):
    results = {driver: [] for driver in df['driver']}

    for _ in range(sims):
        sim_df = df.copy()

        # Add randomness (pace variation)
        sim_df['pace_delta_sim'] = sim_df['pace_delta'] + np.random.normal(0, 0.2, len(df))

        # Random DNF (5% chance)
        sim_df['dnf'] = np.random.rand(len(df)) < 0.05

        # Replace feature
        sim_df['pace_delta'] = sim_df['pace_delta_sim']

        X_sim = sim_df[features]

        sim_df['pred_pos'] = model.predict(X_sim)

        # Penalize DNFs
        sim_df.loc[sim_df['dnf'], 'pred_pos'] = 25

        sim_df = sim_df.sort_values(by='pred_pos')

        for i, row in enumerate(sim_df.itertuples()):
            results[row.driver].append(i + 1)

    return results

# -------------------------------
# ANALYZE RESULTS
# -------------------------------
def analyze(results):
    summary = []

    for driver, finishes in results.items():
        finishes = np.array(finishes)

        summary.append({
            'driver': driver,
            'win_prob': np.mean(finishes == 1),
            'podium_prob': np.mean(finishes <= 3),
            'avg_position': np.mean(finishes)
        })

    return pd.DataFrame(summary).sort_values(by='win_prob', ascending=False)

# -------------------------------
# MAIN
# -------------------------------
def main():
    df = load_event(YEAR, RACE)
    df = engineer_features(df)
    df, _, _ = encode(df)

    model, features = train(df)

    sim_results = simulate_race(df, model, features, SIMULATIONS)

    summary = analyze(sim_results)

    print("\n--- MONTE CARLO RESULTS ---")
    print(summary)


if __name__ == "__main__":
    main()
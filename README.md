# f1-race-predictor-ml-simulation
ML + physics-based F1 race predictor using Monte Carlo simulation for probabilistic race outcomes


# F1 Race Predictor: ML + Physics-Based Simulation

## Overview

This project predicts Formula 1 race outcomes using a combination of:

* Machine Learning (driver/team performance modeling)
* Physics-based lap time simulation (drag + tire degradation)
* Monte Carlo simulation (race uncertainty modeling)

Instead of predicting a single result, the model simulates hundreds of races to estimate:

* Win probability
* Podium probability
* Expected finishing position

---

## Key Features

### 1. Machine Learning Model

* Model: XGBoost / Random Forest
* Inputs:

  * Qualifying position
  * Lap time delta
  * Driver encoding
  * Team encoding
* Output:

  * Predicted race performance

---

### 2. Physics-Based Model

Lap time is modeled using:

* Aerodynamic drag effects
* Power unit performance
* Tire degradation over laps

This introduces real-world engineering realism into predictions.

---

### 3. Monte Carlo Simulation

* 500–2000 race simulations
* Randomized:

  * Pace variation
  * Mechanical failures (DNF probability)
* Output:

  * Probabilistic race outcomes

---

### 4. Interactive Dashboard

Built using Streamlit:

* Select race and season
* Run simulations
* View:

  * Win probabilities
  * Podium chances
  * Driver rankings

---

## Project Structure

f1_project/
│
├── app.py              # Streamlit dashboard
├── simulator.py        # Monte Carlo engine
├── model.py            # ML model
├── physics.py          # Lap time + tire model
├── data_loader.py      # FastF1 integration

---

## Example Output

| Driver | Win Prob | Podium Prob | Avg Position |
| ------ | -------- | ----------- | ------------ |
| VER    | 0.72     | 0.91        | 1.8          |
| PER    | 0.18     | 0.65        | 3.2          |
| HAM    | 0.05     | 0.40        | 5.1          |

---

## How to Run

### Install dependencies

pip install fastf1 pandas numpy scikit-learn xgboost streamlit

### Run dashboard

streamlit run app.py

---

## Engineering Insights

This project combines:

* Data-driven modeling (ML)
* First-principles physics (drag, degradation)
* Stochastic simulation (Monte Carlo)

This approach is analogous to:

* Digital twin systems
* Reliability engineering
* Race strategy simulation in motorsport

---

## Future Improvements

* Tire compound modeling
* Pit stop strategy optimization
* Safety car probability
* Real telemetry integration

---

## Author

Salil Bhat
Mechanical Engineering Student
Focus: Simulation, ML, Motorsport Engineering

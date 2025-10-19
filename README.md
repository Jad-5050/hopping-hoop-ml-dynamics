# 🌀 Hopping Hoop Dynamics & Machine Learning Classification

### A Dual Project for MECH 620 (Intermediate Dynamics) & MECH 534 (Machine Learning)
**Team:** Jad Dagher, Elias Sebaaly, Jawad Chamoun  
**Repository:** [github.com/Jad-5050/hopping-hoop-ml-dynamics](https://github.com/Jad-5050/hopping-hoop-ml-dynamics)

---

## 🎯 Project Overview

This project investigates the motion of an **eccentrically loaded rolling hoop**, a seemingly simple system that can display remarkably complex behaviors — **steady rolling, slipping, skidding, hopping**, and even chaotic motion — depending on its physical parameters.

The work is divided into two complementary phases:

1. **Mechanical Modeling:** Derive and simulate the equations of motion of the hoop using **Lagrangian dynamics** with static and kinetic friction and lift-off impacts.  
2. **Machine Learning:** Use simulation data to **classify motion regimes** (rolling, slipping, hopping, etc.) and visualize **nonlinear decision boundaries** in phase space.

This project bridges **classical dynamics** (MECH 620) and **machine learning** (MECH 534) into one integrated workflow.

---

## ⚙️ Project Structure

hopping-hoop-ml-dynamics/
│
├── 📄 README.md
│
├── 📁 dynamics_model/
│ ├── lagrange_derivation_notes/ # Hand derivations & symbolic checks
│ ├── solver.py # Numerical integration of EOMs
│ └── utils_physics.py # Friction, impact, constraint helpers
│
├── 📁 data/
│ ├── raw_simulations/ # Raw simulation outputs
│ ├── processed_dataset.csv # Labeled dataset for ML
│ └── parameters.json # Parameter sweeps (γ, μ, μs, ω₀)
│
├── 📁 ml_models/
│ ├── baseline_logistic_regression.ipynb # Baseline linear model
│ ├── mlp_model.ipynb # Nonlinear MLP classifier
│ └── temporal_model.ipynb # Markov-inspired model (1D CNN/RNN)
│
├── 📁 results/
│ ├── phase_space_plots/ # Visualization of regime boundaries
│ ├── model_comparison.csv # Accuracy/loss metrics
│ └── figures_for_report/ # Final report visuals
│
└── 📁 docs/
├── MECH534_Project_Proposal.pdf
└── References/

---

## 🧮 Dynamics Background

The hoop is modeled as a rigid body of radius **R** and mass **m**, with its center of mass offset by an eccentricity **γ = e/R**.  
The generalized coordinates are:
- **θ** — angular position of the hoop  
- **x** — horizontal displacement of the contact point  

The system’s motion obeys nonholonomic rolling constraints and experiences Coulomb friction and impacts during lift-off.  
The governing conditions are:

> **N ≥ 0**, **Fₜ = -μN sgn(v₍rel₎)**  

Lift-off occurs when **N = 0**, marking the transition to the hopping regime.

Both **static friction (μs)** and **kinetic friction (μ)** are modeled.  
If μs → ∞, the hoop rolls without slipping, preventing hopping entirely.  
In simulations, μs will be varied within a finite range (typically 0.2–0.8) to explore its influence on transitions between rolling, slipping, and hopping.

---

## 🤖 Machine Learning Component

Once the physics-based simulations are complete, labeled datasets will be used to train three models of increasing complexity:

1. **Multinomial Logistic Regression (Baseline)**  
   - Establishes linear separability in the state space [θ, ω, N, v₍rel₎].  
   - Provides an interpretable baseline aligned with theoretical intuition.  

2. **Feed-Forward Neural Network (MLP)**  
   - Learns curved, nonlinear regime boundaries between motion types.  
   - Captures the nonlinear relationships among eccentricity, angular velocity, and normal force.  

3. **Temporal / Markov-Inspired Model (1D CNN or RNN)**  
   - Incorporates short-term memory by considering the current and previous **k** timesteps.  
   - For **k = 0**, it reduces to the MLP; for **k > 0**, it captures temporal correlations and smoother transitions.  
   - By comparing different values of **k** (e.g., 3 vs 10), we can analyze how much past information improves predictive stability.

These models will be compared in terms of **accuracy**, **interpretability**, and **computational efficiency**.

---

## 🧠 Current Progress

✅ Analytical derivations of the Lagrange equations (in progress by hand)  
✅ Verification of limiting cases (pure rolling, no eccentricity)  
🔄 Symbolic setup in Python for validation  
🔜 Solver implementation and dataset generation (Weeks 2–3)  
🔜 Logistic regression and MLP model training (Weeks 3–4)

---

## 🗂️ How to Run

1. **Clone the repository**
   ```bash
   git clone https://github.com/Jad-5050/hopping-hoop-ml-dynamics.git
   cd hopping-hoop-ml-dynamics

2. **Install dependencies**
   pip install -r requirements.txt

3. **Run the dynamics simulation**
   python dynamics_model/solver.py

4. **Train the ML models**
   jupyter notebook ml_models/baseline_logistic_regression.ipynb

## 🧾 Dependencies

   Python ≥ 3.10

   NumPy

   SciPy

   Matplotlib

   Pandas

   scikit-learn

   PyTorch or TensorFlow (for advanced ML stages)

## 🗓️ Timeline Summary
| Week | Dates        | Key Deliverables                                            |
| ---- | ------------ | ----------------------------------------------------------- |
| 1    | Oct 20–26    | Hand derivations of Lagrange’s equations, constraint checks |
| 2    | Oct 27–Nov 2 | Solver setup, basic integration tests                       |
| 3    | Nov 3–9      | Simulations + Logistic Regression model                     |
| 4    | Nov 10–13    | MLP model + milestone presentation                          |
| 5    | Nov 14–20    | Temporal model, hyperparameter tuning                       |
| 6    | Nov 21–26    | Final analysis, report, and presentation                    |

## 📚 References

1. Tokieda, T. F. The Hopping Hoop, The American Mathematical Monthly, 1997.

2. Butler, J. P. Hopping Hoops Don’t Hop, The American Mathematical Monthly, 1999.

3. Pritchett, T. The Hopping Hoop Revisited, The American Mathematical Monthly, 1999.

4. Theron, W. F. D., & Maritz, M. F. The Amazing Variety of Motions of a Loaded Hoop, European Journal of Physics, 2008.

5. Geier, A., Bachmann, T., & Bartel, T. Machine Learning-Based State Maps for Complex Dynamical Systems, Mechanical Systems and Signal Processing, 2023.

6. Buzhardt, J. D. et al. A Pendulum-Driven Legless Rolling Jumping Robot, IEEE Robotics and Automation Letters, 2023.

7. The Cardillo Project. Nonlinear Dynamical Systems Simulation Framework. GitHub, 2025. https://github.com/cardilloproject/cardillo

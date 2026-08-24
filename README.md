# Regime-Aware Reinforcement Learning for Multi-Period Asset Allocation

An implementation of a multi-period financial decision engine built entirely from scratch in Python and PyTorch. This architecture transitions away from static, single-period optimization parameters toward an adaptive, model-free state-space framework capable of scaling across high-dimensional market regimes while incorporating path-dependent frictions.

The framework builds directly on the dual-regime and Mixture-of-Experts (MoE) methodologies pioneered by **Dr. John M. Mulvey** and **Yirui Luo** (Princeton University ORFE, 2026).

---

## 🔬 Core Methodology & Mathematical Architecture

Traditional multi-stage optimization models (e.g., Dynamic Programming or Multi-Stage Stochastic Programming) frequently succumb to the *curse of dimensionality* when expanding state vectors to accommodate real-world constraints. This deployment addresses those constraints by parameterizing policy and value spaces via an End-to-End deep Actor-Critic network trained on sample-based experience tuples $(s_t, a_t, r_t, s_{t+1})$.

### 1. Dual-Trigger Investable Universes (Action Masking)
The asset allocation universe is dynamically constrained based on joint global market conditions and asset-specific characteristics. Let $\hat{h}_{t+1}^{\text{Bear}} \in \{0, 1\}$ denote the forecasted global macro regime (proxied via a Statistical Jump Model paired with an XGBoost threshold layer set at $0.7$), and let $\hat{h}_{i,t+1}^{\text{Bull}} \in \{0, 1\}$ signify individual asset-level trends. 

Assets are programmatically assigned to the investable **Bear Market Defensive Assets (BMDA)** set via the following mapping logic:

$$\mathcal{A}_{t+1}^{\text{Bear}} = \{i : \hat{h}_{t+1}^{\text{Bear}} = 1 \text{ and } \hat{h}_{i,t+1}^{\text{Bull}} = 1\}$$

In the forward pass of the policy network, non-qualifying assets are dynamically pruned out using an **Action Mask Layer** before the final Softmax distribution computation:

$$\pi(a_t | s_t) = \text{Softmax}\left(\text{Logits}_{\text{raw}} + \mathbf{M}_t\right)$$

$$\text{where } \mathbf{M}_{t, i} = \begin{cases} 0 & \text{if } i \in \mathcal{A}_{t+1} \\ -\infty & \text{if } i \notin \mathcal{A}_{t+1} \end{cases}$$

This structure guarantees that procyclical equity weights are forced to absolute zero during systemwide turbulence, restricting the Bear Agent's optimization path exclusively to validated defensive hedges.

### 2. Path-Dependent Reward Space Formulation
To capture structural frictions that violate standard Markovian independence (such as multi-period tax layers or execution costs), the environment tracks sequential histories to penalize running drawdown profiles. The step-by-step reward landscape $R_t$ is formulated as:

$$R_t = \ln(1 + R_{p,t}) - \alpha \cdot \mathcal{D}_{21\text{d}} - \beta \cdot \mathcal{TC}_t$$

Where:
*   $\ln(1 + R_{p,t})$ models long-term log-utility of portfolio wealth.
*   $\mathcal{D}_{21\text{d}} = \frac{\max_{\tau \in [t-21, t]} W_\tau - W_t}{\max_{\tau \in [t-21, t]} W_\tau}$ tracks a rolling 21-day maximum path-dependent drawdown.
*   $\mathcal{TC}_t = \sum_{j} \tau_j |w_{j,t} - w_{j,t^-}|$ calculates endogenous turnover-adjusted transaction costs scaled at a baseline friction of 10 bps.

---

## 💻 Script Execution & Deep Learning Loop

The pipeline is coded from scratch to bypass black-box wrapper limitations, allowing for granular tracking of temporal difference (TD) errors and immediate gradient step adjustments.

### Dependencies
*   Python 3.10+
*   PyTorch 2.0+
*   NumPy

### Core Components Implemented:
*   `RegimeAwareFinancialEnv`: Custom tracking environment managing path-dependent drawdown calculations, transaction frictions, and mock historical matrix arrays.
*   `ActorCriticNetwork`: Deep architecture housing shared parameter layers with separate linear heads for action weight probabilities and continuous value tracking, processing an active `[returns_history + regime_signals + current_weights]` state vector.
*   `Optimization Step`: Exact calculation of Policy Gradient losses adjusted by sample-based advantage estimations:
    
    $$A_t = r_t + \gamma V(s_{t+1}) - V(s_t)$$

---

## 📚 References & Literature Baseline

1. **Luo, Y. & Mulvey, J. M. (2026).** *Bringing Reinforcement Learning to Multi-Period Financial Planning: A Bridge Between Learning-Enabled and Stochastic Optimization.* IMA Journal of Management Mathematics.
2. **Luo, Y. & Mulvey, J. M. (2026).** *Regime-Aware Asset Allocation with Dual-Regime Signals and Regime-Dependent Asset Selection.* The Journal of Financial Data Science, 8(3), 105-134.
3. **Schulman, J. et al. (2017).** *Proximal Policy Optimization Algorithms.* arXiv preprint arXiv:1707.06347.

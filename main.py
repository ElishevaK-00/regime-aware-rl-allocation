import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class RegimeAwareFinancialEnv:
    """
    A custom environment built from scratch to model Dr. Mulvey's 
    Regime-Aware Multi-Period Asset Allocation framework.
    """
    def __init__(self, num_assets=5, window_size=21, tc_fee=0.001):
        self.num_assets = num_assets
        self.window_size = window_size
        self.tc_fee = tc_fee # 10 bps transaction cost
        
        # Mock historical data for standalone execution simulation
        self.total_steps = 252
        self.returns_history = np.random.normal(0.0005, 0.01, (self.total_steps, num_assets))
        
        # 0 = Bull, 1 = Bear
        self.global_regimes = np.random.choice([0, 1], size=self.total_steps, p=[0.7, 0.3])
        # Asset-specific regimes (0 = Bull, 1 = Bear)
        self.asset_regimes = np.random.choice([0, 1], size=(self.total_steps, num_assets), p=[0.6, 0.4])
        
        self.reset()

    def reset(self):
        self.current_step = self.window_size
        self.portfolio_value_history = [1.0] * self.window_size
        self.peak_portfolio_value = 1.0
        self.current_weights = np.ones(self.num_assets) / self.num_assets
        return self._get_state()

    def _get_state(self):
        # State space consists of the past 21 days of asset returns + current regime signals
        past_returns = self.returns_history[self.current_step - self.window_size:self.current_step].flatten()
        global_sig = np.array([self.global_regimes[self.current_step]])
        asset_sigs = self.asset_regimes[self.current_step]
        
        state = np.concatenate([past_returns, global_sig, asset_sigs, self.current_weights])
        return torch.FloatTensor(state)

    def step(self, action_weights):
        # 1. Calculate Transaction Costs
        turnover = np.sum(np.abs(action_weights - self.current_weights))
        tc_penalty = turnover * self.tc_fee
        
        # 2. Process Market Returns
        step_returns = self.returns_history[self.current_step]
        portfolio_return = np.dot(action_weights, step_returns)
        
        # Update portfolio net wealth accounting for transaction fees
        new_value = self.portfolio_value_history[-1] * (1 + portfolio_return) - tc_penalty
        self.portfolio_value_history.append(new_value)
        
        # 3. Path-Dependent Risk Math (21-Day Max Drawdown Tracking)
        if new_value > self.peak_portfolio_value:
            self.peak_portfolio_value = new_value
        
        recent_window = self.portfolio_value_history[-self.window_size:]
        window_max = max(recent_window)
        current_drawdown = (window_max - new_value) / window_max if window_max > 0 else 0
        
        # 4. Formulate the Reward Space (Log-Utility minus Drawdown and TC Penalties)
        reward = np.log(1 + portfolio_return) - (2.0 * current_drawdown) - (1.5 * tc_penalty)
        
        # Advance environment state
        self.current_weights = action_weights
        self.current_step += 1
        done = self.current_step >= self.total_steps
        
        next_state = self._get_state() if not done else torch.zeros_like(self._get_state())
        return next_state, reward, done


class ActorCriticNetwork(nn.Module):
    """
    A foundational deep neural network handling both policy choices (Actor)
    and value generation estimations (Critic) simultaneously.
    """
    def __init__(self, input_dim, num_assets):
        super(ActorCriticNetwork, self).__init__()
        self.num_assets = num_assets
        
        # Shared feature base processing network
        self.shared_base = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        
        # Heads executing the independent output computations
        self.actor_head = nn.Linear(64, num_assets)
        self.critic_head = nn.Linear(64, 1)

    def forward(self, state, global_regime, asset_regimes):
        base_out = self.shared_base(state)
        
        # Calculate policy choices logits
        raw_logits = self.actor_head(base_out)
        
        # --- DR. MULVEY'S DUAL-TRIGGER ACTION MASK ENGINE ---
        # Matrix manipulation logic: check if market is in Global Bear (1)
        # If yes, block assets that are Asset-Specific Bear (1).
        # We achieve masking by setting unapproved logit parameters to negative infinity.
        mask = torch.zeros(self.num_assets)
        if global_regime == 1:
            for i in range(self.num_assets):
                if asset_regimes[i] == 1: # Asset is not resilient in a crisis
                    mask[i] = -1e9 # Block entirely during SoftMax scaling
                    
        masked_logits = raw_logits + mask
        action_probs = torch.softmax(masked_logits, dim=-1)
        
        # Calculate baseline state evaluation valuation
        state_value = self.critic_head(base_out)
        
        return action_probs, state_value


# --- ENGINE DISPATCH TRAINING LOOP FROM SCRATCH ---
if __name__ == "__main__":
    num_assets = 5
    window_size = 21
    
    # Calculate state inputs dimensionality: (21 days * 5 assets) + 1 global signal + 5 asset signals + 5 weight inputs
    state_input_dimension = (window_size * num_assets) + 1 + num_assets + num_assets
    
    env = RegimeAwareFinancialEnv(num_assets=num_assets, window_size=window_size)
    model = ActorCriticNetwork(input_dim=state_input_dimension, num_assets=num_assets)
    optimizer = optim.Adam(model.parameters(), lr=0.002)

    print("Beginning Training Routine for Dual-Regime Optimizer Agent...\n")
    
    for epoch in range(5): # Simulate 5 optimization training epochs
        state = env.reset()
        done = False
        epoch_rewards = 0
        
        while not done:
            # Extract historical indicators for current processing window
            g_regime = env.global_regimes[env.current_step]
            a_regimes = env.asset_regimes[env.current_step]
            
            # Forward computation step through networks
            probs, state_value = model(state, g_regime, a_regimes)
            
            # Categorical sample selection based on calculated probability weight distributions
            action_dist = torch.distributions.Categorical(probs)
            action_selected = action_dist.sample()
            
            # Convert Selected discrete scalar choices into real portfolio distribution vector
            # (In standard trading desks, action outputs map directly to structural allocation weights)
            target_weights = np.zeros(num_assets)
            target_weights[action_selected.item()] = 1.0 # Simulate full allocation for simplified tracking
            
            # Step environment forward
            next_state, reward, done = env.step(target_weights)
            epoch_rewards += reward
            
            # Calculate bootstrap values for the baseline loss function computation
            _, next_state_value = model(next_state, env.global_regimes[min(env.current_step, 251)], env.asset_regimes[min(env.current_step, 251)])
            target_value = reward + (0.99 * next_state_value.item() * (1 - int(done)))
            
            # Compute temporal difference advantage error math models
            advantage = target_value - state_value.item()
            
            # Explicit computation optimization of Loss structures
            critic_loss = torch.pow(target_value - state_value, 2)
            actor_loss = -action_dist.log_prob(action_selected) * advantage
            total_step_loss = actor_loss + 0.5 * critic_loss
            
            # Optimization backpropagation update updates execution steps
            optimizer.zero_grad()
            total_step_loss.backward()
            optimizer.step()
            
            state = next_state
            
        print(f"Epoch {epoch + 1} Optimization Sequence Complete. Total Cumulative Reward: {epoch_rewards:.4f}")

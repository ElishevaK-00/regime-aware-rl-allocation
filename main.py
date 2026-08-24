import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class MultiStageHybridEnv:
    """
    Forward-Thinking Hybrid MSP-RL Environment.
    Instead of standard model-free transitions, this system builds a forward-looking 
    Scenario Tree (Multi-Stage Stochastic Programming baseline) at each epoch. 
    The RL agent learns to approximate the terminal cost-to-go function, 
    truncating a massive multi-period tree to solve the 'curse of dimensionality'.
    """
    def __init__(self, num_assets=5, lookahead_stages=3, branching_factor=4):
        self.num_assets = num_assets
        self.lookahead_stages = lookahead_stages # Multi-period MSP horizon
        self.branching_factor = branching_factor # Scenario tree branching width
        self.total_steps = 252
        
        # Base Market Generation Parameters
        self.returns_mu = np.array([0.0006, 0.0004, 0.0008, 0.0002, 0.0005])
        self.returns_sigma = np.random.normal(0.01, 0.002, (num_assets, num_assets))
        self.covariance = np.dot(self.returns_sigma, self.returns_sigma.T)
        
        self.reset()

    def reset(self):
        self.current_step = 0
        self.portfolio_value = 1.0
        self.current_weights = np.ones(self.num_assets) / self.num_assets
        
        # --- Generalized Multi-Period Path Tracking ---
        # Treating taxes as a difficult special case by tracking individual asset cost bases
        self.asset_cost_basis = np.ones(self.num_assets)
        self.unrealized_gains = np.zeros(self.num_assets)
        
        return self._generate_msp_state()

    def _generate_msp_state(self):
        """
        Generates a forward-looking scenario matrix simulating a pruned multi-stage decision tree.
        This provides the structural forward-looking capacity of an MSP framework.
        """
        scenarios = np.random.multivariate_normal(
            self.returns_mu, self.covariance, 
            (self.lookahead_stages, self.branching_factor, self.num_assets)
        )
        
        # State vector: [Forward Scenarios (Flattened) + Current Allocations + Cost Basis Vectors]
        state_tensor = np.concatenate([
            scenarios.flatten(),
            self.current_weights,
            self.asset_cost_basis
        ])
        return torch.FloatTensor(state_tensor)

    def step(self, target_weights, tax_penalty_coefficient=0.15):
        """
        Executes a transition step. General parameters handle long-term log-utility, 
        while a specific tracking module isolates taxes as an endogenous special penalty.
        """
        # 1. Market Realization Vector
        realized_returns = np.random.multivariate_normal(self.returns_mu, self.covariance)
        
        # 2. Endogenous Special Case: Path-Dependent Tax Harvesting Math
        # Track structural changes in cost basis and calculate capital gains penalties
        turnover = target_weights - self.current_weights
        tax_drag = 0.0
        
        for i in range(self.num_assets):
            if turnover[i] < 0: # Asset Sale / Liquidation trigger
                realized_gain = abs(turnover[i]) * max(0, realized_returns[i] + 1 - self.asset_cost_basis[i])
                tax_drag += realized_gain * tax_penalty_coefficient
            elif turnover[i] > 0: # Asset Purchase / Basis recalculation
                # Update weighted average cost basis
                new_total_weight = self.current_weights[i] + turnover[i]
                self.asset_cost_basis[i] = ((self.current_weights[i] * self.asset_cost_basis[i]) + 
                                            (turnover[i] * (1.0 + realized_returns[i]))) / new_total_weight
        
        # 3. Dynamic Multi-Period Wealth Transition
        portfolio_return = np.dot(target_weights, realized_returns)
        net_step_wealth = (self.portfolio_value * (1.0 + portfolio_return)) - tax_drag
        
        # Calculate standard terminal reward objective (Generalized multi-period planning utility)
        reward = np.log(max(1e-6, net_step_wealth / self.portfolio_value))
        
        # Update system state
        self.portfolio_value = net_step_wealth
        self.current_weights = target_weights
        self.current_step += 1
        done = self.current_step >= self.total_steps
        
        next_state = self._generate_msp_state() if not done else torch.zeros_like(self._generate_msp_state())
        return next_state, reward, done


class MSPValueApproximator(nn.Module):
    """
    Dual-Headed Critic Policy Architecture.
    The critic functions explicitly as the Terminal Cost-to-Go evaluation engine 
    for the multi-period MSP scenario paths, removing the exponential branch problem.
    """
    def __init__(self, input_dim, num_assets):
        super(MSPValueApproximator, self).__init__()
        
        # Deep network layers map forward multi-stage scenarios
        self.network_core = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.Mish(), # Smooth continuous gradient propagation for complex scenario surfaces
            nn.Linear(256, 128),
            nn.Mish()
        )
        
        self.policy_head = nn.Linear(128, num_assets)
        self.value_head = nn.Linear(128, 1)

    def forward(self, state_vector):
        latent_features = self.network_core(state_vector)
        
        # Softmax returns optimal multi-period planning portfolio weights
        portfolio_distribution = torch.softmax(self.policy_head(latent_features), dim=-1)
        # Structural Value estimation replaces downstream tree computations
        terminal_cost_to_go = self.value_head(latent_features)
        
        return portfolio_distribution, terminal_cost_to_go


if __name__ == "__main__":
    num_assets = 5
    stages = 3
    branching = 4
    
    # Calculate dimensional footprint: (Stages * Branches * Assets) + Allocation Weights + Cost Basis Vector
    input_dimension = (stages * branching * num_assets) + num_assets + num_assets
    
    env = MultiStageHybridEnv(num_assets=num_assets, lookahead_stages=stages, branching_factor=branching)
    agent = MSPValueApproximator(input_dim=input_dimension, num_assets=num_assets)
    optimizer = optim.AdamW(agent.parameters(), lr=1e-4, weight_decay=1e-5)
    
    print("Initializing Multi-Stage Stochastic Programming + RL Hybrid Solver...")
    
    for epoch in range(3):
        state = env.reset()
        done = False
        cumulative_log_utility = 0.0
        
        while not done:
            # Predict best weights and approximate terminal value function
            weights_tensor, cost_to_go = agent(state)
            
            # Map structural activation parameters to environment vector
            action_weights = weights_tensor.detach().cpu().numpy()
            
            # Step forward through true transition probability path
            next_state, step_utility, done = env.step(action_weights)
            cumulative_log_utility += step_utility
            
            # Bootstrapping structural targets using value function truncation
            with torch.no_grad():
                _, next_stage_cost_to_go = agent(next_state)
                target_value = step_utility + (0.98 * next_stage_cost_to_go * (1.0 - int(done)))
            
            # Policy gradient and Value network updates via Actor-Critic TD-Loss
            advantage = target_value - cost_to_go
            critic_loss = torch.pow(target_value - cost_to_go, 2)
            
            # Reinforce policy selections using advantage directions
            policy_loss = -torch.log(weights_tensor).mean() * advantage.item()
            total_hybrid_loss = policy_loss + 0.5 * critic_loss
            
            optimizer.zero_grad()
            total_hybrid_loss.backward()
            optimizer.step()
            
            state = next_state
            
        print(f"Sequence Execution Complete -> Cumulative Log Utility Score: {cumulative_log_utility:.6f}")

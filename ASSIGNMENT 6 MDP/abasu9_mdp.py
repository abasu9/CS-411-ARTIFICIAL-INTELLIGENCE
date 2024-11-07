from pprint import pprint
import copy
from typing import List
import numpy as np

class MDP:
    def __init__(self, params) -> None:
        # Initialize the MDP with parameters
        self.num_rows = params[0]  # Number of rows in the grid
        self.num_cols = params[1]  # Number of columns in the grid
        self.wall_positions = params[2]  # List of wall positions
        self.terminal_states = params[3]  # List of terminal states with rewards
        self.default_reward = params[4]  # Default reward for non-terminal states
        self.transition_probs = params[5]  # Transition probabilities for actions
        self.discount_factor = params[6]  # Discount factor for future rewards
        self.epsilon_value = params[7]  # Convergence threshold for value iteration
        self.wall_char = 'x'  # Character representing walls
        
        self.initialize_mdp()  # Set up the MDP grid and terminal indicators
        
    def set_mdp(self, rows: int, cols: int, walls: List[List[int]], terminals: List[List[int]], reward: float, trans_probs: List[float], discount: float, epsilon: float) -> None:
        # Method to update MDP parameters and re-initialize
        self.num_rows = rows
        self.num_cols = cols
        self.wall_positions = walls
        self.terminal_states = terminals
        self.default_reward = reward
        self.transition_probs = trans_probs
        self.discount_factor = discount
        self.epsilon_value = epsilon
        
        self.initialize_mdp()  # Reinitialize MDP grid and indicators
    
    def initialize_mdp(self) -> None:
        # Create the grid and set terminal states and walls
        self.grid = [[self.default_reward for _ in range(self.num_cols)] for _ in range(self.num_rows)]
        self.terminal_indicators = [[False for _ in range(self.num_cols)] for _ in range(self.num_rows)]
        
        # Mark wall positions in the grid
        for wall in self.wall_positions:
            x, y = wall
            self.grid[x][y] = self.wall_char
        
        # Mark terminal states and their rewards
        for terminal in self.terminal_states:
            x, y, terminal_reward = terminal
            self.grid[x][y] = terminal_reward
            self.terminal_indicators[x][y] = True
            
        # Print the initialized grid
        pprint(self.grid)
  
    def get_mdp(self) -> List[List]: 
        # Return the grid size and grid data
        return self.num_rows, self.num_cols, self.grid
    
    def get_terminal_indicators(self) -> List[List[bool]]: 
        # Return the terminal state indicators
        return self.terminal_indicators
    
    def get_wall_char(self) -> str: 
        # Return the character representing walls
        return self.wall_char
    
    def get_discount_factor(self) -> float: 
        # Return the discount factor
        return self.discount_factor
    
    def get_epsilon_value(self) -> float: 
        # Return the epsilon value for convergence
        return self.epsilon_value
    
    def get_transition_probs(self) -> List[float]: 
        # Return the transition probabilities
        return self.transition_probs

class ValueIteration:
    actions = ['up', 'down', 'left', 'right']  # Possible actions
    
    def compute_positions(self, mdp, rows, cols, x, y, action):
        # Compute next positions based on the action taken
        positions = [x, y, x, y, x, y]  # Default to current position
        
        if action == 'left':
            positions = [x-1, y, x, y-1, x, y+1]  # Move left
        elif action == 'right':
            positions = [x+1, y, x, y+1, x, y-1]  # Move right
        elif action == 'down':
            positions = [x, y-1, x+1, y, x-1, y]  # Move down
        elif action == 'up':
            positions = [x, y+1, x-1, y, x+1, y]  # Move up
        else:
            print('Invalid action given:', action)
            exit(0)

        # Check for grid boundaries and wall collisions
        for idx in [0, 2, 4]:
            next_idx = idx + 1
            if positions[idx] < 0 or positions[idx] >= rows: 
                positions[idx] = x  # Stay in place if out of bounds
            if positions[next_idx] < 0 or positions[next_idx] >= cols: 
                positions[next_idx] = y  # Stay in place if out of bounds
            if mdp[positions[idx]][positions[next_idx]] == 'x':
                positions[idx] = x  # Stay in place if there's a wall
                positions[next_idx] = y  

        return positions

    def compute_q_value(self, mdp, rows, cols, x, y, action, utility_values):
        # Calculate the Q-value for a given action at a position
        positions = self.compute_positions(mdp, rows, cols, x, y, action)
        primary_x, primary_y, sec_x1, sec_y1, sec_x2, sec_y2 = positions

        # Calculate the Q-value based on transition probabilities
        q_value = (0.8 * (mdp[primary_x][primary_y] + utility_values[primary_x][primary_y]) +
                    0.1 * (mdp[sec_x1][sec_y1] + utility_values[sec_x1][sec_y1]) +
                    0.1 * (mdp[sec_x2][sec_y2] + utility_values[sec_x2][sec_y2]))
        return q_value

    def run_value_iteration(self, mdp: MDP):
        # Main method to run value iteration algorithm
        rows, cols, grid = mdp.get_mdp()
        terminal_states = mdp.get_terminal_indicators()
        wall_char = mdp.get_wall_char()
        discount = mdp.get_discount_factor()
        epsilon = mdp.get_epsilon_value()

        utility_values = [[0] * cols for _ in range(rows)]  # Initialize utility values
        utility_values_prev = [[0] * cols for _ in range(rows)]  # Previous utility values
        delta = float('inf')  # Convergence variable
        
        action_policy = [[0] * cols for _ in range(rows)]  # Initialize action policy

        iteration = 0
        while True:
            utility_values = copy.deepcopy(utility_values_prev)  # Copy previous values for comparison
            delta = 0  # Reset delta for this iteration

            for x in range(rows):
                for y in range(cols):
                    # Skip walls and terminal states
                    if grid[x][y] != wall_char and not terminal_states[x][y]:
                        max_q_value = 0
                        for action in self.actions:
                            # Compute Q-value for each action
                            q_value = self.compute_q_value(grid, rows, cols, x, y, action, utility_values)
                            if q_value > max_q_value:
                                max_q_value = q_value  # Update max Q-value
                                action_policy[x][y] = action  # Update action policy
                        utility_values_prev[x][y] = max_q_value  # Update utility value
                        delta = max(delta, abs(utility_values_prev[x][y] - utility_values[x][y]))  # Update delta

            # Print current utility values for debugging
            print(f"Iteration {iteration}:")
            pprint(np.flipud(np.transpose([[round(value, 4) for value in row] for row in utility_values_prev])))
            print("\n")

            iteration += 1
            # Check for convergence
            if delta <= (epsilon * (1 - discount) / discount):
                break
        
        # Print the optimal action policy
        pprint(np.flipud(np.transpose(action_policy)))
        print("-" * 70)

class PolicyIteration:
    actions = ['up', 'down', 'left', 'right']  # Possible actions
    
    def compute_transition(self, x, y, new_x, new_y, prob, utility_values, mdp: MDP):
        # Calculate the expected value from a transition
        rows, cols, grid = mdp.get_mdp()
        discount = mdp.get_discount_factor()
        wall_char = mdp.get_wall_char()
        result: float = None
        
        # Check for wall collisions and compute the result
        if grid[new_x][new_y] != wall_char:
            result = prob * (grid[new_x][new_y] + discount * utility_values[new_x][new_y])
        else: 
            result = prob * (grid[x][y] + discount * utility_values[x][y])
        return result
    
    def compute_positions(self, mdp: MDP, x: int, y: int, action: str):
        # Compute next positions based on the action taken
        rows, cols, grid = mdp.get_mdp()
        
        # Initialize positions for primary and secondary actions
        pri_x, pri_y, sec_x1, sec_y1, sec_x2, sec_y2, ter_x, ter_y = x, y, x, y, x, y, x, y
        if action == 'up':
            pri_x = max(0, x-1)
            sec_y1 = max(0, y-1)
            sec_y2 = min(cols-1, y+1)
            ter_x = min(rows-1, x+1)
        elif action == 'down':
            pri_x = min(rows-1, x+1)
            sec_y1 = min(cols-1, y+1)
            sec_y2 = max(0, y-1)
            ter_x = max(0, x-1)
        elif action == 'left':
            pri_y = max(0, y-1)
            sec_x1 = min(rows-1, x+1)
            sec_x2 = max(0, x-1)
            ter_y = min(cols-1, y+1)
        elif action == 'right':
            pri_y = min(cols-1, y+1)
            sec_x1 = max(0, x-1)
            sec_x2 = min(rows-1, x+1)
            ter_y = max(0, y-1)
        else:
            print("Incorrect Action")
        
        return pri_x, pri_y, sec_x1, sec_y1, sec_x2, sec_y2, ter_x, ter_y
    
    def policy_evaluation(self, policy: List[List[str]], utility_values: List[List[float]], mdp: MDP) -> List[List[float]]:
        # Evaluate the utility values for the current policy
        rows, cols, grid = mdp.get_mdp()
        discount = mdp.get_discount_factor()
        wall_char = mdp.get_wall_char()
        transition_probs = mdp.get_transition_probs()
        
        # Create coefficient matrix A and constant vector b for linear equations
        A = np.zeros((rows * cols, rows * cols))
        b = np.zeros(rows * cols)
        
        for x in range(rows):
            for y in range(cols):
                # Handle terminal states
                if policy[x][y] is None:
                    A[x * cols + y, x * cols + y] = 1
                    b[x * cols + y] = utility_values[x][y]  
                    continue
                
                A[x * cols + y, x * cols + y] = 1
                # Compute positions based on the action in the policy
                pri_x, pri_y, sec_x1, sec_y1, sec_x2, sec_y2, ter_x, ter_y = self.compute_positions(mdp, x, y, policy[x][y])
                positions = [(pri_x, pri_y), (sec_x1, sec_y1), (sec_x2, sec_y2), (ter_x, ter_y)]
                
                for idx, (new_x, new_y) in enumerate(positions):
                    # Update matrix A and vector b based on valid transitions
                    if 0 <= new_x < rows and 0 <= new_y < cols and grid[new_x][new_y] != wall_char:
                        A[x * cols + y, new_x * cols + new_y] -= discount * transition_probs[idx]
                        b[x * cols + y] += transition_probs[idx] * (grid[new_x][new_y] if isinstance(grid[new_x][new_y], (int, float)) else mdp.default_reward)
                    else:
                        A[x * cols + y, x * cols + y] -= discount * transition_probs[idx]
                        b[x * cols + y] += transition_probs[idx] * grid[x][y]
        
        # Solve the system of linear equations for utility values
        U_flat = np.linalg.solve(A, b)
        utility_values_new = U_flat.reshape((rows, cols)).tolist()
        
        return utility_values_new
  
    def compute_q_value(self, mdp: MDP, x, y, action, utility_values):
        # Calculate the Q-value for a given action at a position
        rows, cols, grid = mdp.get_mdp()
        transition_probs = mdp.get_transition_probs()
        
        # Compute positions based on the action
        pri_x, pri_y, sec_x1, sec_y1, sec_x2, sec_y2, ter_x, ter_y = self.compute_positions(mdp, x, y, action)

        # Sum expected values from all transitions for the Q-value
        q_value = (self.compute_transition(x, y, pri_x, pri_y, transition_probs[0], utility_values, mdp) +
                    self.compute_transition(x, y, sec_x1, sec_y1, transition_probs[1], utility_values, mdp) +
                    self.compute_transition(x, y, sec_x2, sec_y2, transition_probs[2], utility_values, mdp) +
                    self.compute_transition(x, y, ter_x, ter_y, transition_probs[3], utility_values, mdp))
        return q_value
        
    def run_policy_iteration(self, mdp):
        # Main method to run policy iteration algorithm
        rows, cols, grid = mdp.get_mdp()
        terminal_states = mdp.get_terminal_indicators()
        wall_char = mdp.get_wall_char()
        
        utility_values = [[0] * cols for _ in range(rows)]  # Initialize utility values
        policy = [['up'] * cols for _ in range(rows)]  # Initialize policy with default action
        
        # Set policy for terminal states to None
        for x in range(rows):
            for y in range(cols):
                if terminal_states[x][y] or grid[x][y] == wall_char:
                    policy[x][y] = None
        
        iteration = 0
        while True:
            utility_values = self.policy_evaluation(policy, utility_values, mdp)  # Evaluate policy
            unchanged = True
            
            # Print current utility values for debugging
            print(f"Iteration {iteration}:")
            print("Utility Values:")
            pprint(np.flipud(np.transpose([[round(value, 4) for value in row] for row in utility_values])))
            print("\n")
            
            for x in range(rows):
                for y in range(cols):
                    if policy[x][y] is None: 
                        continue  # Skip terminal states
                    max_q_value = 0
                    best_action = 'up'  # Default best action
                    for action in self.actions:
                        # Compute Q-value for each action
                        q_value = self.compute_q_value(mdp, x, y, action, utility_values)
                        if q_value > max_q_value:
                            max_q_value = q_value  # Update max Q-value
                            best_action = action  # Update best action
                    
                    # Update policy if there's a better action
                    if max_q_value > self.compute_q_value(mdp, x, y, policy[x][y], utility_values):
                        policy[x][y] = best_action
                        unchanged = False  # Policy has changed
            
            iteration += 1
            # Check for convergence
            if unchanged: 
                break
        
        # Print the final optimal policy
        pprint(np.flipud(np.transpose(policy)))
        print("-" * 70)

def parse_input_file(file_path):
    # Parse the input file to extract MDP parameters
    rows, cols = 0, 0
    walls, terminals = [], []
    reward, transition_probs = 0.0, []
    discount_factor, epsilon_value = 0.0, 0.0

    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if not line or line.startswith('#'):
                continue  # Skip empty lines and comments

            # Extract parameters from lines
            if line.startswith("size"):
                size_values = line.split(":")[1].strip().split()
                rows, cols = int(size_values[0]), int(size_values[1])
            elif line.startswith("walls"):
                wall_values = line.split(":")[1].strip().split()
                walls = [[int(wall_values[i])-1, int(wall_values[i+1])-1] for i in range(0, len(wall_values), 2)]
            elif line.startswith("terminal_states"):
                terminal_values = line.split(":")[1].strip().split(',')
                terminals = [[int(val.split()[0])-1, int(val.split()[1])-1, float(val.split()[2])]
                             for val in terminal_values]
            elif line.startswith("reward"):
                reward = float(line.split(":")[1].strip())
            elif line.startswith("transition_probabilities"):
                transition_probs = list(map(float, line.split(":")[1].strip().split()))
            elif line.startswith("discount_rate"):
                discount_factor = float(line.split(":")[1].strip())
            elif line.startswith("epsilon"):
                epsilon_value = float(line.split(":")[1].strip())

    return rows, cols, walls, terminals, reward, transition_probs, discount_factor, epsilon_value

if __name__ == '__main__':
    input_file_path = "/Users/abhishekbasu/Documents/CS 411   ARTIFICIAL INTELLIGENCE/Assignments/ASSIGNMENT 6 MDP/mdp_input.txt"  # Path to input file
    # Parse input file to get MDP parameters
    rows, cols, walls, terminals, reward, transition_probs, discount_factor, epsilon_value = parse_input_file(input_file_path)
    parameters = [rows, cols, walls, terminals, reward, transition_probs, discount_factor, epsilon_value]

    print("\nMDP with rewards:")
    mdp_instance = MDP(parameters)  # Create an MDP instance

    print("\nValue Iteration:")
    value_iter_instance = ValueIteration()  # Create a ValueIteration instance
    value_iter_instance.run_value_iteration(mdp_instance)  # Run value iteration

    print("\nPolicy Iteration:")
    policy_iter_instance = PolicyIteration()  # Create a PolicyIteration instance
    policy_iter_instance.run_policy_iteration(mdp_instance)  # Run policy iteration

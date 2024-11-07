This program simulates a Markov Decision Process (MDP) using both Value Iteration and Policy Iteration methods. It allows users to define an MDP through an input file and computes the utility values and optimal policies for the defined states.
Input File Format
The input file should follow a specific structure to define the parameters of the MDP. Each parameter must be clearly labelled:
1. Size of the Grid:
   * Format: size: <rows> <cols>
   * Example: size: 4 4
   * Description: Specifies the number of rows and columns in the grid.
2. Walls:
   * Format: walls: <row1> <col1> <row2> <col2> ...
   * Example: walls: 1 2 2 3
   * Description: Lists wall positions as pairs of integers (0-indexed).
3. Terminal States:
   * Format: terminal_states: <row1> <col1> <reward1>, <row2> <col2> <reward2>, ...
   * Example: terminal_states: 0 0 1.0, 3 3 -1.0
   * Description: Defines terminal states and their associated rewards.
4. Default Reward:
   * Format: reward: <default_reward>
   * Example: reward: -0.1
   * Description: Specifies the default reward for non-terminal states.
5. Transition Probabilities:
   * Format: transition_probabilities: <p1> <p2> <p3> <p4>
   * Example: transition_probabilities: 0.8 0.1 0.1 0.0
   * Description: Lists probabilities of transitioning to neighbouring states based on actions.
6. Discount Rate:
   * Format: discount_rate: <discount_factor>
   * Example: discount_rate: 0.9
   * Description: Specifies the discount factor (between 0 and 1).
7. Epsilon Value:
   * Format: epsilon: <epsilon_value>
   * Example: epsilon: 0.01
   * Description: Sets the convergence threshold for value iteration.
Example Input File


#size of the gridworld


size : 4 3


#list of location of walls


walls : 2 2


#list of terminal states (row,column,reward)


terminal_states : 4 2 -1 , 4 3 +1


#reward in non-terminal states


reward : -0.04


#transition probabilities


transition_probabilities : 0.8 0.1 0.1 0


discount_rate : 1


epsilon : 0.001






How to Run the Program
* Prepare the Input File:
   * Create a text file (e.g., mdp_input.txt) with the parameters defined above.
* Modify the File Path:
   * Open the script and update the following line in the __main__ block:
python
input_file_path = "/path/to/your/mdp_input.txt"
   * Replace "/path/to/your/mdp_input.txt" with the actual path to your input file.
   * Run the Program:
   * Ensure Python is installed on your machine.
   * Execute the script from the command line:
python abasu9_mdp.py
   * Replace abasu9_mdp.py with the name of your Python file.
      * View the Output:
      * The program will output the results of value iteration and policy iteration, displaying utility values and optimal policies for the MDP.
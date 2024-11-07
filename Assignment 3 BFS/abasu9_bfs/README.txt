ABHISHEK BASU || UIN : 658369163 || NET ID : abasu9
15 Puzzle 
This script uses a breadth-first search (BFS) approach to solve the 15 puzzles, a popular sliding puzzle consisting of a grid with numbered tiles and one empty space.
Requirements
* Python Version: 3. x (Recommended: Python 3.6 or higher)
* This program relies solely on the standard Python library; no external packages are required.
Instructions for Command Line Execution
1. Install Python:
   * Ensure that Python is installed on your machine. Download it from python.org if you haven't done so.
2. Save the Script:
   * Copy the Python code and save it as abasu9_bfs_15_puzzle.py.
3. Open a Command Prompt or Terminal:
   * Access your command prompt (Windows) or terminal (macOS/Linux).
4. Change Directory to Script Location:
   * Use the cd command to navigate to the directory where you saved abasu9_bfs_15_puzzle.py. For example:
cd path/to/your/directory
   5. Run the Program:
   * Execute the script using the command:
python abasu9_bfs_15_puzzle.py
      6. Input the Initial Board Setup:
      * When prompted, enter the puzzle configuration as 16 integers (use 0 for the empty tile), separated by spaces. Example input:
Copy code
1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 0
         7. View the Results:
         * The program will display the sequence of moves needed to solve the puzzle, the number of nodes expanded during the search, the time taken, and the memory used.
Sample Input/Output
Input:


Enter the initial board configuration (16 numbers including 0): 
1 0 2 4 5 7 3 8 9 6 11 12 13 10 14 15 



Output:
Enter the initial board configuration (16 numbers including 0): 
1 0 2 4 5 7 3 8 9 6 11 12 13 10 14 15 
Moves: RDLDDRR
Number of Nodes expanded: 358
Time Taken: 0.008458s
Memory Used: 16.78kb


** Process exited - Return Code: 0 **
Press Enter to exit terminal




Important Information
         * Ensure the initial configuration you enter is solvable; not every arrangement can lead to a solution.
         * If the configuration is unsolvable, the program will inform you with the message "No solution found."
Source code: 
from collections import deque
import sys
import time


N = 4  # Size of board 4x4 for 15 puzzle


class State:
    def __init__(self, board, a, b, moves):
        self.board = board
        self.a = a  # Empty tile coordinates
        self.b = b
        self.moves = moves


    def is_goal_state(self):
        num = 1
        for i in range(N):
            for j in range(N):
                if self.board[i][j] != num and not (i == N - 1 and j == N - 1):
                    return False
                num += 1
        return True


    def temp_children(self):
        children = []
        da = [0, 0, -1, 1]
        db = [-1, 1, 0, 0]
        move_chars = ['L', 'R', 'U', 'D']


        for d in range(4):
            nx = self.a + da[d]
            ny = self.b + db[d]


            if 0 <= nx < N and 0 <= ny < N:
                # Create new child state
                child_board = [row[:] for row in self.board]
                child_board[self.a][self.b], child_board[nx][ny] = child_board[nx][ny], child_board[self.a][self.b]
                child_state = State(child_board, nx, ny, self.moves + move_chars[d])
                children.append(child_state)


        return children


    def __hash__(self):
        return hash(tuple(map(tuple, self.board)))  # Unique hash for the board state


    def __eq__(self, other):
        return self.board == other.board  # Equality check based on board state


def solve_puzzle(initial_state):
    start_time = time.time()
    queue = deque()
    explored = set()


    queue.append(initial_state)


    while queue:
        current = queue.popleft()


        if current.is_goal_state():
            # Goal state reached
            print("Moves:", current.moves)
            print("Number of Nodes expanded:", len(explored))
            print("Time Taken: {:.6f}s".format(time.time() - start_time))
            print("Memory Used: {:.2f}kb".format(sys.getsizeof(current) * len(explored) / 1024))
            return


        explored.add(hash(current))  # Use the hash of the board state for immutability


        for child in current.temp_children():
            if hash(child) not in explored:  # Check if child state has been explored
                queue.append(child)


    print("No solution found.")


if __name__ == "__main__":
    initial_board = []
    input_board = input("Enter the initial board configuration (16 numbers including 0): ")
    numbers = list(map(int, input_board.split()))
    
    for i in range(N):
        initial_board.append(numbers[i * N:(i + 1) * N])
    
    empty_a, empty_b = next((i, j) for i in range(N) for j in range(N) if initial_board[i][j] == 0)


    initial_state = State(initial_board, empty_a, empty_b, "")
    solve_puzzle(initial_state)


OUTPUT : 
  







ABHISHEK BASU || UIN : 658369163 || NET ID : abasu9
# ABHISHEK BASU
# UIN : 658369163
# NET ID : abasu9 
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

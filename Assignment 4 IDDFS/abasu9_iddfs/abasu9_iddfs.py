from collections import deque
import sys
import time

N = 4  # Size of the board (4x4 for 15-puzzle)

class State:
    def __init__(self, board, a, b, moves, depth=0):
        self.board = board
        self.a = a  # Empty tile's x-coordinate
        self.b = b  # Empty tile's y-coordinate
        self.moves = moves
        self.depth = depth  # Current depth of the node

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
                # Swap the empty tile with the adjacent tile
                child_board[self.a][self.b], child_board[nx][ny] = child_board[nx][ny], child_board[self.a][self.b]
                # Create child state with incremented depth
                child_state = State(child_board, nx, ny, self.moves + move_chars[d], self.depth + 1)
                children.append(child_state)

        return children

    def __hash__(self):
        return hash(tuple(map(tuple, self.board)))  # Unique hash for the board state

    def __eq__(self, other):
        return self.board == other.board  # Equality check based on board state


def depth_limited_search(current, limit, explored):
    """
    Perform a depth-limited search up to a given depth limit.
    """
    if current.is_goal_state():
        return current

    if current.depth >= limit:
        return None

    explored.add(hash(current))  # Mark the node as explored

    for child in current.temp_children():
        if hash(child) not in explored:
            result = depth_limited_search(child, limit, explored)
            if result is not None:
                return result

    return None


def iterative_deepening_dfs(initial_state):
    """
    Perform iterative deepening depth-first search (IDDFS).
    """
    depth = 0
    nodes_expanded = 0
    start_time = time.time()

    while True:
        explored = set()
        result = depth_limited_search(initial_state, depth, explored)
        nodes_expanded += len(explored)

        if result is not None:
            # Goal state reached
            print("Moves:", result.moves)
            print("Number of Nodes expanded:", nodes_expanded)
            print("Time Taken: {:.6f}s".format(time.time() - start_time))
            print("Memory Used: {:.2f} KB".format(sys.getsizeof(result) * nodes_expanded / 1024))
            return
        depth += 1  # Increase depth limit


if __name__ == "__main__":
    # Input the initial board configuration
    initial_board = []
    input_board = input("Enter the initial board configuration (16 numbers including 0): ")
    numbers = list(map(int, input_board.split()))

    for i in range(N):
        initial_board.append(numbers[i * N:(i + 1) * N])

    # Find the position of the empty tile (0)
    empty_a, empty_b = next((i, j) for i in range(N) for j in range(N) if initial_board[i][j] == 0)

    # Create the initial state
    initial_state = State(initial_board, empty_a, empty_b, "")

    # Run Iterative Deepening Depth First Search
    iterative_deepening_dfs(initial_state)
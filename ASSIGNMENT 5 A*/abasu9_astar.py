import heapq
import sys
import time

N = 4  # Size of the board (4x4 for 15-puzzle)

class State:
    def __init__(self, board, a, b, moves, depth=0, heuristic_value=0):
        self.board = board
        self.a = a  # Empty tile's x-coordinate
        self.b = b  # Empty tile's y-coordinate
        self.moves = moves
        self.depth = depth  # Current depth of the node
        self.heuristic_value = heuristic_value  # Calculated heuristic value (f = g + h)

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

    def __lt__(self, other):
        return self.heuristic_value < other.heuristic_value  # For priority queue comparison

def num_misplaced_tiles(state):
    """
    Heuristic: Number of misplaced tiles.
    """
    misplaced = 0
    num = 1
    for i in range(N):
        for j in range(N):
            if state.board[i][j] != 0 and state.board[i][j] != num:
                misplaced += 1
            num += 1
    return misplaced

def manhattan_distance(state):
    """
    Heuristic: Manhattan distance.
    """
    distance = 0
    for i in range(N):
        for j in range(N):
            tile = state.board[i][j]
            if tile != 0:
                target_x = (tile - 1) // N
                target_y = (tile - 1) % N
                distance += abs(i - target_x) + abs(j - target_y)
    return distance

def a_star_search(initial_state, heuristic_func):
    """
    Perform A* search with the given heuristic function.
    """
    open_list = []
    explored = set()
    nodes_expanded = 0
    start_time = time.time()

    # Add the initial state to the priority queue
    initial_state.heuristic_value = initial_state.depth + heuristic_func(initial_state)
    heapq.heappush(open_list, initial_state)

    while open_list:
        current = heapq.heappop(open_list)

        # If goal is reached, return the result
        if current.is_goal_state():
            print("Moves:", current.moves)
            print("Number of Nodes expanded:", nodes_expanded)
            print("Time Taken: {:.6f}s".format(time.time() - start_time))
            print("Memory Used: {:.2f} KB".format(sys.getsizeof(current) * nodes_expanded / 1024))
            return

        # Add current state to explored set
        explored.add(hash(current))
        nodes_expanded += 1

        # Generate children and add to open list
        for child in current.temp_children():
            if hash(child) not in explored:
                child.heuristic_value = child.depth + heuristic_func(child)
                heapq.heappush(open_list, child)

    print("No solution found.")
    print("Number of Nodes expanded:", nodes_expanded)
    print("Time Taken: {:.6f}s".format(time.time() - start_time))
    print("Memory Used: {:.2f} KB".format(sys.getsizeof(current) * nodes_expanded / 1024))

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

   
    print("Running A* Search with Number of Misplaced Tiles Heuristic...")
    a_star_search(initial_state, num_misplaced_tiles)
    print("Running A* Search with Manhattan Distance Heuristic...")
    a_star_search(initial_state, manhattan_distance)


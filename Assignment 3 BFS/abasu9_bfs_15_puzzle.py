import time
import psutil
import sys
from collections import deque

# Goal state for 15-puzzle
GOAL_STATE = tuple([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0])

# Directions for moving the blank tile
DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # up, down, left, right

class Node:
    def __init__(self, state, parent=None, move=None, depth=0):
        self.state = state
        self.parent = parent
        self.move = move
        self.depth = depth

    def get_path(self):
        path = []
        node = self
        while node.parent is not None:
            path.append(node.move)
            node = node.parent
        return path[::-1]  # Reverse to get correct order of moves


def is_solvable(puzzle):
    # Function to check if the puzzle is solvable
    flat_puzzle = [x for x in puzzle if x != 0]
    inversions = 0
    for i in range(len(flat_puzzle)):
        for j in range(i + 1, len(flat_puzzle)):
            if flat_puzzle[i] > flat_puzzle[j]:
                inversions += 1
    return inversions % 2 == 0


def is_goal(state):
    return state == GOAL_STATE


def get_neighbors(node):
    neighbors = []
    state = node.state
    blank_index = state.index(0)  # Get the position of the blank tile (0)
    row, col = divmod(blank_index, 4)

    for direction in DIRECTIONS:
        new_row, new_col = row + direction[0], col + direction[1]
        if 0 <= new_row < 4 and 0 <= new_col < 4:
            new_blank_index = new_row * 4 + new_col
            new_state = list(state)
            new_state[blank_index], new_state[new_blank_index] = new_state[new_blank_index], new_state[blank_index]
            neighbors.append(Node(tuple(new_state), parent=node, move=direction, depth=node.depth + 1))

    return neighbors


def iterative_deepening_dfs(start_state):
    depth = 0
    nodes_expanded = 0

    while True:
        result, expanded = depth_limited_search(start_state, depth)
        nodes_expanded += expanded
        if result is not None:
            return result, nodes_expanded
        depth += 1


def depth_limited_search(start_state, limit):
    stack = deque([Node(start_state)])
    nodes_expanded = 0

    while stack:
        node = stack.pop()
        if is_goal(node.state):
            return node, nodes_expanded

        if node.depth < limit:
            nodes_expanded += 1
            neighbors = get_neighbors(node)
            stack.extend(neighbors)

    return None, nodes_expanded


def memory_usage_kb():
    # Get memory usage in KB
    process = psutil.Process()
    return process.memory_info().rss // 1024


def main():
    # Input the initial board configuration
    initial_board = input("Enter the initial board configuration (separate numbers with spaces): ").split()
    initial_board = tuple(map(int, initial_board))

    if not is_solvable(initial_board):
        print("The given puzzle is not solvable.")
        return

    start_time = time.time()

    result, nodes_expanded = iterative_deepening_dfs(initial_board)

    end_time = time.time()

    if result is None:
        print("No solution found.")
    else:
        # Print the solution details
        moves = result.get_path()
        print(f"Moves: {moves}")
        print(f"Number of Nodes expanded: {nodes_expanded}")
        print(f"Time Taken: {end_time - start_time} seconds")
        print(f"Memory Used: {memory_usage_kb()} KB")


if __name__ == "__main__":
    main()

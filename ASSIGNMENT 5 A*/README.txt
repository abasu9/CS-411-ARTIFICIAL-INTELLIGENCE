A* Search for 15-Puzzle
Instructions:
1. Make sure you have Python installed on your computer.
2. Save the code file as abasu9_astar.py.
3. Open the terminal (or command prompt) on your computer.
4. Navigate to the folder where the file is saved.
5. Run the program by typing python abasu9_astar.py and pressing Enter.
6. When the program asks, enter the initial puzzle configuration as 16 numbers (with 0 representing the empty tile). Example: 1 0 2 4 5 7 3 8 9 6 11 12 13 10 14 15.
Output:
After running the program, you will see:
* The sequence of moves (L for Left, R for Right, U for Up, D for Down).
* The total number of nodes (steps) the program expanded.
* The total time it took to solve the puzzle.
* The amount of memory used.

Enter the initial board configuration (16 numbers including 0): 
1 0 2 4 5 7 3 8 9 6 11 12 13 10 14 15 
Running A* Search with Number of Misplaced Tiles Heuristic...
Moves: RDLDDRR
Number of Nodes expanded: 7
Time Taken: 0.001060s
Memory Used: 0.33 KB
Running A* Search with Manhattan Distance Heuristic...
Moves: RDLDDRR
Number of Nodes expanded: 7
Time Taken: 0.001232s
Memory Used: 0.33 KB




** Process exited - Return Code: 0 **
Press Enter to exit terminal
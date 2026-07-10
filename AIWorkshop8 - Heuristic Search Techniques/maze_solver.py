from simpleai.search import SearchProblem, astar

# --------------------------
# Step 1: Define the Maze Structure
# --------------------------
MAZE = [
    "########################",
    "#O                     #",
    "# ##### ##########  ####",
    "#     #          #     #",
    "####### ######## # ####",
    "#                #     #",
    "# ###### ##########  ##",
    "#                    X#",
    "########################"
]

# Directions: Up, Down, Left, Right (only 4-directional movement, no diagonals)
DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


# --------------------------
# Step 2: Maze Solver Class (Inherits from SearchProblem)
# --------------------------
class MazeSolver(SearchProblem):
    def __init__(self, maze):
        self.maze = maze
        # Find start (O) and goal (X) positions
        self.start = self._find_position('O')
        self.goal = self._find_position('X')
        # Initialize SearchProblem with start position (tuple: (row, column))
        super().__init__(initial_state=self.start)

    def _find_position(self, target_char):
        """Helper method: Find the (row, column) of a target character (O/X) in the maze."""
        for row_idx, row in enumerate(self.maze):
            if target_char in row:
                col_idx = row.index(target_char)
                return (row_idx, col_idx)
        raise ValueError(f"Target character '{target_char}' not found in the maze.")

    def actions(self, state):
        """Override: Return all valid moves (directions) from the current state (row, col)."""
        valid_actions = []
        current_row, current_col = state

        for dr, dc in DIRECTIONS:
            new_row = current_row + dr
            new_col = current_col + dc
            # Check if new position is within maze bounds
            if 0 <= new_row < len(self.maze) and 0 <= new_col < len(self.maze[0]):
                # Check if new position is not an obstacle (#)
                if self.maze[new_row][new_col] != '#':
                    # Action is represented as the new (row, col) position
                    valid_actions.append((new_row, new_col))
        return valid_actions

    def result(self, state, action):
        """Override: Return the new state after taking the action (valid move)."""
        # Action is already the new (row, col) position (from actions() method)
        return action

    def is_goal(self, state):
        """Override: Check if current state is the goal position (X)."""
        return state == self.goal

    def cost(self, state, action, next_state):
        """Override: Cost of moving from state to next_state (1 step per move)."""
        return 1

    def heuristic(self, state):
        """Override: Heuristic function (Manhattan distance to goal)."""
        current_row, current_col = state
        goal_row, goal_col = self.goal
        # Manhattan distance = |current_row - goal_row| + |current_col - goal_col|
        return abs(current_row - goal_row) + abs(current_col - goal_col)


# --------------------------
# Step 3: Visualize the Solution Path
# --------------------------
def visualize_solution(maze, path):
    """Convert the maze and solution path into a human-readable string (mark path with '*')."""
    # Convert maze from list of strings to list of lists (to modify characters)
    maze_copy = [list(row) for row in maze]
    # Extract path positions (skip the first element: (None, start_state))
    path_positions = [step[1] for step in path if step[1] is not None]

    # Mark path with '*' (keep start 'O' and goal 'X')
    for (row, col) in path_positions:
        if maze_copy[row][col] not in ('O', 'X'):
            maze_copy[row][col] = '*'

    # Convert back to list of strings and join for output
    return '\n'.join([''.join(row) for row in maze_copy])


# --------------------------
# Step 4: Run the Maze Solver
# --------------------------
if __name__ == '__main__':
    # 1. Initialize the maze solver problem
    maze_problem = MazeSolver(MAZE)

    # 2. Solve using A* algorithm (from simpleai library)
    print("Solving maze...\n")
    solution = astar(maze_problem)

    # 3. Extract and print the solution details
    if solution:
        # Get the full path (each step is (action, state))
        solution_path = list(solution.path())
        # Calculate number of steps (subtract 1 for initial state)
        steps = len(solution_path) - 1

        # Print results
        print(f"✅ Maze solved!")
        print(f"Start position: {maze_problem.start}")
        print(f"Goal position: {maze_problem.goal}")
        print(f"Number of steps to goal: {steps}\n")
        print("Solution Path (marked with '*'):")
        print(visualize_solution(MAZE, solution_path))
    else:
        print("❌ No valid path found from 'O' to 'X'.")
import copy
import math


class SudokuEnv:
    """
    Minimal multi-turn Sudoku environment for Agentic ES experiments.

    Board convention:
        0 = empty cell
        1..N = Sudoku values

    Action:
        (row, col, value)

    Reward:
        1.0 -> puzzle solved
        0.0 -> otherwise

    Invalid actions consume one environment step.
    """

    def __init__(self, puzzle, solution, max_steps=None):
        self.initial_puzzle = copy.deepcopy(puzzle)
        self.solution = copy.deepcopy(solution)

        self.size = len(puzzle)
        self.subgrid_size = int(math.sqrt(self.size))

        if self.subgrid_size ** 2 != self.size:
            raise ValueError(
                f"Board size {self.size} must have an integer square root."
            )

        if len(solution) != self.size:
            raise ValueError("Puzzle and solution must have the same size.")

        for row in puzzle:
            if len(row) != self.size:
                raise ValueError("Puzzle must be square.")

        for row in solution:
            if len(row) != self.size:
                raise ValueError("Solution must be square.")

        self.max_steps = (
            max_steps
            if max_steps is not None
            else self.size * self.size
        )

        self.board = None
        self.steps_taken = 0

    # --------------------------------------------------
    # Environment API
    # --------------------------------------------------

    def reset(self):
        self.board = copy.deepcopy(self.initial_puzzle)
        self.steps_taken = 0

        return self.get_observation()

    def get_observation(self):
        if self.board is None:
            return None

        return copy.deepcopy(self.board)

    def step(self, action):
        """
        Execute one Sudoku action.

        action:
            (row, col, value)

        returns:
            observation,
            reward,
            done,
            info
        """

        if self.board is None:
            self.reset()

        self.steps_taken += 1

        # ----------------------------------------------
        # Parse action
        # ----------------------------------------------

        if (
            not isinstance(action, (tuple, list))
            or len(action) != 3
        ):
            return self._invalid_transition("invalid_action_format")

        row, col, value = action

        # ----------------------------------------------
        # Type checking
        # ----------------------------------------------

        if not all(isinstance(x, int) for x in (row, col, value)):
            return self._invalid_transition("action_must_contain_integers")

        # ----------------------------------------------
        # Range checking
        # ----------------------------------------------

        if not (0 <= row < self.size):
            return self._invalid_transition("row_out_of_range")

        if not (0 <= col < self.size):
            return self._invalid_transition("col_out_of_range")

        if not (1 <= value <= self.size):
            return self._invalid_transition("value_out_of_range")

        # ----------------------------------------------
        # Cannot overwrite an existing cell
        # ----------------------------------------------

        if self.board[row][col] != 0:
            return self._invalid_transition("cell_not_empty")

        # ----------------------------------------------
        # Sudoku constraint checking
        # ----------------------------------------------

        if not self.is_valid(row, col, value):
            return self._invalid_transition("invalid_move")

        # ----------------------------------------------
        # Apply valid action
        # ----------------------------------------------

        self.board[row][col] = value

        solved = self.is_solved()
        timeout = self.steps_taken >= self.max_steps

        reward = 1.0 if solved else 0.0
        done = solved or timeout

        info = {
            "success": solved,
            "invalid": False,
            "steps": self.steps_taken,
            "timeout": timeout and not solved,
        }

        return (
            self.get_observation(),
            reward,
            done,
            info,
        )

    # --------------------------------------------------
    # Sudoku logic
    # --------------------------------------------------

    def is_valid(self, row, col, value):
        """
        Check whether placing value at board[row][col]
        satisfies Sudoku constraints.
        """

        if not (1 <= value <= self.size):
            return False

        # ----------------------------------------------
        # Row constraint
        # ----------------------------------------------

        if value in self.board[row]:
            return False

        # ----------------------------------------------
        # Column constraint
        # ----------------------------------------------

        for r in range(self.size):
            if self.board[r][col] == value:
                return False

        # ----------------------------------------------
        # Subgrid constraint
        # ----------------------------------------------

        box_row = (
            row // self.subgrid_size
        ) * self.subgrid_size

        box_col = (
            col // self.subgrid_size
        ) * self.subgrid_size

        for r in range(
            box_row,
            box_row + self.subgrid_size
        ):
            for c in range(
                box_col,
                box_col + self.subgrid_size
            ):
                if self.board[r][c] == value:
                    return False

        return True

    def is_solved(self):
        """
        For reproduction purposes we compare against
        the known ground-truth solution.
        """

        return self.board == self.solution

    # --------------------------------------------------
    # Helpers
    # --------------------------------------------------

    def _invalid_transition(self, reason):
        """
        Invalid actions still consume one turn.

        We keep reward sparse:
            reward = 0

        Episode ends if max_steps is exhausted.
        """

        timeout = self.steps_taken >= self.max_steps

        info = {
            "success": False,
            "invalid": True,
            "error": reason,
            "steps": self.steps_taken,
            "timeout": timeout,
        }

        return (
            self.get_observation(),
            0.0,
            timeout,
            info,
        )

    def render(self):
        if self.board is None:
            print("Environment has not been reset.")
            return

        separator = "-" * (self.size * 2 + self.subgrid_size + 1)

        for r in range(self.size):
            if r > 0 and r % self.subgrid_size == 0:
                print(separator)

            row_parts = []

            for c in range(self.size):
                if c > 0 and c % self.subgrid_size == 0:
                    row_parts.append("|")

                value = self.board[r][c]

                if value == 0:
                    row_parts.append(".")
                else:
                    row_parts.append(str(value))

            print(" ".join(row_parts))

    # --------------------------------------------------
    # Useful diagnostics
    # --------------------------------------------------

    def num_empty_cells(self):
        return sum(
            cell == 0
            for row in self.board
            for cell in row
        )

    def get_minimum_successful_horizon(self):
        """
        If one valid action can fill at most one cell,
        the theoretical minimum successful horizon
        equals the number of masked cells.
        """

        return sum(
            cell == 0
            for row in self.initial_puzzle
            for cell in row
        )
        
        
        
if __name__ == "__main__": 
    solution = [
        [1, 2, 3, 4],
        [3, 4, 1, 2],
        [2, 1, 4, 3],
        [4, 3, 2, 1],
    ]

    puzzle = [
        [1, 0, 3, 4],
        [3, 4, 1, 0],
        [2, 1, 4, 3],
        [4, 3, 2, 1],
    ]

    env = SudokuEnv(
        puzzle=puzzle,
        solution=solution,
        max_steps=6,
    )

    obs = env.reset()

    env.render()

    print(
        "H* =",
        env.get_minimum_successful_horizon()
    )
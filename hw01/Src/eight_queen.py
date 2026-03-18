def solve_n_queens(n):
    if n == 0:
        return []
    
    def is_safe(board, row, col):
        for i in range(row):
            if board[i] == col or abs(board[i] - col) == abs(i - row):
                return False
        return True
    
    def backtrack(row, current_solution):
        if row == n:
            solutions.append(current_solution[:])
            return
        
        for col in range(n):
            if is_safe(current_solution, row, col):
                current_solution.append(col)
                backtrack(row + 1, current_solution)
                current_solution.pop()
    
    solutions = []
    backtrack(0, [])
    return solutions

def print_board(solution):
    n = len(solution)
    for row in range(n):
        line = []
        for col in range(n):
            if solution[row] == col:
                line.append('Q')
            else:
                line.append('.')
        print(' '.join(line))
    print()
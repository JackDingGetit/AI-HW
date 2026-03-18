import unittest
import sys
import os

# 添加src目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from eight_queen import solve_n_queens

class TestNQueens(unittest.TestCase):
    def test_n_4(self):
        solutions = solve_n_queens(4)
        self.assertEqual(len(solutions), 2)
        expected_solutions = [[1, 3, 0, 2], [2, 0, 3, 1]]
        for solution in solutions:
            self.assertIn(solution, expected_solutions)
    
    def test_n_8(self):
        solutions = solve_n_queens(8)
        self.assertEqual(len(solutions), 92)
    
    def test_n_1(self):
        solutions = solve_n_queens(1)
        self.assertEqual(len(solutions), 1)
        self.assertEqual(solutions, [[0]])
    
    def test_n_0(self):
        solutions = solve_n_queens(0)
        self.assertEqual(len(solutions), 0)
        self.assertEqual(solutions, [])

if __name__ == '__main__':
    unittest.main()
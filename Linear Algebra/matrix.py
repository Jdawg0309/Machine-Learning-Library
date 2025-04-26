import math
import random
from copy import deepcopy

random.seed(42)

class Matrix:
    def __init__(self, m, n):
        self.m = m
        self.n = n
        self.result = []
        self._validate_dimensions()
        
    def _validate_dimensions(self):
        if self.m <= 0 or self.n <= 0:
            raise ValueError("Matrix dimensions must be positive integers")
        if not isinstance(self.m, int) or not isinstance(self.n, int):
            raise TypeError("Matrix dimensions must be integers")

    def populate(self, data=None, random_num=False, default_val=2):
        """Populate matrix with:
        - Custom 2D array (data parameter)
        - Random numbers (random_num=True)
        - Default value (default_val)
        """
        if data is not None:
            # Validate custom data structure
            if len(data) != self.m:
                raise ValueError(f"Data must contain exactly {self.m} rows")
            for row in data:
                if len(row) != self.n:
                    raise ValueError(f"Each row must contain exactly {self.n} elements")
            self.result = deepcopy(data)
        else:
            # Generate new data
            self.result = []
            for _ in range(self.m):
                if random_num:
                    row = [random.randint(1, 10) for _ in range(self.n)]
                else:
                    row = [default_val] * self.n
                self.result.append(row)
        return self.result

    def _validate_row_length(self, row):
        if len(row) != self.n:
            raise ValueError(f"Row must contain exactly {self.n} elements")

    def __str__(self):
        return '\n'.join(['\t'.join(map(str, row)) for row in self.result])

    def display(self):
        print(self)
        print()

    def scalarMultiplication(self, scalar):
        """Multiply matrix by a scalar value"""
        for i in range(self.m):
            for j in range(self.n):
                self.result[i][j] *= scalar

    def matrixAddition(self, other_matrix):
        """Add another matrix to this matrix"""
        self._validate_operation_dimensions(other_matrix)
        return Matrix.from_2d_list([
            [self.result[i][j] + other_matrix.result[i][j]
             for j in range(self.n)]
            for i in range(self.m)
        ])

    def _validate_operation_dimensions(self, other):
        if self.m != other.m or self.n != other.n:
            raise ValueError("Matrices must have same dimensions for this operation")

    def transpose(self):
        """Return new transposed matrix"""
        return Matrix.from_2d_list([
            [self.result[j][i] for j in range(self.m)]
            for i in range(self.n)
        ])

    def matrixMultiplication(self, other_matrix):
        """Multiply with another matrix"""
        if self.n != other_matrix.m:
            raise ValueError(
                "Columns of first matrix must match rows of second matrix")
                
        result = [
            [
                sum(a * b for a, b in zip(row, col))
                for col in other_matrix.transpose().result
            ]
            for row in self.result
        ]
        return Matrix.from_2d_list(result)

    @classmethod
    def from_2d_list(cls, data):
        """Create matrix from 2D list"""
        m = len(data)
        n = len(data[0]) if m > 0 else 0
        new_matrix = cls(m, n)
        new_matrix.result = deepcopy(data)
        return new_matrix

    # Gaussian Elimination methods with enhanced safety
    def _validate_row_index(self, index):
        if not (0 <= index < self.m):
            raise IndexError(f"Row index must be between 0 and {self.m-1}")

    def gaussRowAddition(self, target_row, source_row, factor=1):
        """Add scaled row to another row"""
        self._validate_row_index(target_row)
        self._validate_row_index(source_row)
        self.result[target_row] = [
            self.result[target_row][j] + factor * self.result[source_row][j]
            for j in range(self.n)
        ]

    def gaussRowScalar(self, row_index, scalar):
        """Multiply row by scalar"""
        self._validate_row_index(row_index)
        self.result[row_index] = [x * scalar for x in self.result[row_index]]

    def gaussRowSwap(self, row1, row2):
        """Swap two rows"""
        self._validate_row_index(row1)
        self._validate_row_index(row2)
        self.result[row1], self.result[row2] = self.result[row2], self.result[row1]

    def gaussElimination(self):
        """Convert matrix to row-echelon form"""
        for pivot in range(min(self.m, self.n)):
            max_row = self._find_pivot_row(pivot)
            if max_row != pivot:
                self.gaussRowSwap(pivot, max_row)

            if self.result[pivot][pivot] == 0:
                continue

            self._normalize_pivot_row(pivot)
            self._eliminate_below_pivot(pivot)

    def _find_pivot_row(self, col):
        max_row = col
        for row in range(col, self.m):
            if abs(self.result[row][col]) > abs(self.result[max_row][col]):
                max_row = row
        return max_row

    def _normalize_pivot_row(self, pivot):
        divisor = self.result[pivot][pivot]
        self.gaussRowScalar(pivot, 1/divisor)

    def _eliminate_below_pivot(self, pivot):
        for row in range(pivot + 1, self.m):
            factor = self.result[row][pivot]
            self.gaussRowAddition(row, pivot, -factor)

    def getDeterminant(self):
        if self.m != self.n:
            raise ValueError("Determinant requires square matrix")
        
        # Create a working copy that preserves original data types
        temp = Matrix(self.m, self.n)
        temp.result = deepcopy(self.result)
        sign = 1
        det = 1

        for col in range(self.n):
            # Find pivot row
            max_row = col
            for row in range(col, self.m):
                if abs(temp.result[row][col]) > abs(temp.result[max_row][col]):
                    max_row = row

            if max_row != col:
                temp.result[col], temp.result[max_row] = temp.result[max_row], temp.result[col]
                sign *= -1

            if temp.result[col][col] == 0:
                return 0

            # Calculate pivot before elimination
            pivot = temp.result[col][col]
            det *= pivot

            # Eliminate below without normalizing
            for row in range(col + 1, self.m):
                factor = temp.result[row][col] / pivot
                for c in range(col, self.n):
                    temp.result[row][c] -= factor * temp.result[col][c]

        return det * sign
        
    def Cramer(self, b):
        solution_set = []
        determinant = self.getDeterminant()
        for i in range(self.n):
            a = self.get_column(i)
            self.edit_column(i, b)
            numerator_determinant = self.getDeterminant()
            solution_set.append(numerator_determinant/determinant)
            self.edit_column(i, a)
        return solution_set
    
    def getInverse(self):
        """Returns the inverse of the matrix using Gauss-Jordan elimination"""
        if self.m != self.n:
            raise ValueError("Matrix must be square to compute inverse")
        
        n = self.m
        det = self.getDeterminant()
        if det == 0:
            raise ValueError("Matrix is singular and cannot be inverted")

        # Create augmented matrix [self | I]
        augmented = Matrix(n, 2*n)
        augmented.result = [
            row.copy() + [1.0 if i == j else 0.0 for j in range(n)]
            for i, row in enumerate(self.result)
        ]

        # Perform Gauss-Jordan elimination
        for col in range(n):
            # Find pivot row with maximum absolute value
            pivot_row = max(range(col, n), key=lambda r: abs(augmented.result[r][col]))
            
            if augmented.result[pivot_row][col] == 0:
                raise ValueError("Matrix is singular")
                
            if pivot_row != col:
                augmented.gaussRowSwap(col, pivot_row)

            # Normalize pivot row
            pivot_val = augmented.result[col][col]
            augmented.gaussRowScalar(col, 1.0/pivot_val)

            # Eliminate other rows
            for row in range(n):
                if row != col:
                    factor = augmented.result[row][col]
                    augmented.gaussRowAddition(row, col, -factor)

        # Extract inverse from augmented matrix
        inverse = Matrix(n, n)
        inverse.result = [row[n:] for row in augmented.result]
        return inverse

    def get_null_space(self):
        b = [0 for i in range(self.m)]
        return self.Cramer(b)
    
    # Safe access methods
    def get_row(self, index):
        self._validate_row_index(index)
        return deepcopy(self.result[index])

    def get_column(self, index):
        if not (0 <= index < self.n):
            raise IndexError(f"Column index must be between 0 and {self.n-1}")
        return [row[index] for row in self.result]

    def edit_row(self, index, new_row):
        self._validate_row_index(index)
        if len(new_row) != self.n:
            raise ValueError(f"New row must contain {self.n} elements")
        self.result[index] = deepcopy(new_row)

    def edit_column(self, index, new_col):
        if not (0 <= index < self.n):
            raise IndexError(f"Column index must be between 0 and {self.n-1}")
        if len(new_col) != self.m:
            raise ValueError(f"New column must contain {self.m} elements")
        for i in range(self.m):
            self.result[i][index] = new_col[i]


# Example usage
if __name__ == "__main__":
    try:
        # Create and display matrix
        m = Matrix(3, 3)
        m.populate([[1, 3, 6], [2, 2, 2], [7, 8, 9]])
        print("Original Matrix:")
        m.display()

        m.getInverse().display()


    except Exception as e:
        print(f"Matrix Error: {e}")

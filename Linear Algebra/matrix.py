import math
import random

random.seed(42)

class Matrix:
    def __init__(self, m, n):
        self.m = m 
        self.n = n
        self.result = []

    def populate(self, randomNum=False):
        row = []
        for i in range(self.m):
            column = []
            for j in range(self.n):
                if randomNum:
                    x = random.randint(1, 10)
                else:
                    x = 2
                column.append(x)
            row.append(column)
        self.result = row  
        return self.result

    def display(self):
        for i in self.result:
            print(i)
        print()

    def scalarMultiplication(self, scalar):
        for i in range(self.m):
            for j in range(self.n):
                self.result[i][j] *= scalar

    def matrixAddition(self, other_matrix):
        if self.m != other_matrix.m or self.n != other_matrix.n:
            raise ValueError("Matrix dimensions must match for addition")
        
        result = Matrix(self.m, self.n)
        result.result = [
            [self.result[i][j] + other_matrix.result[i][j] for j in range(self.n)]
            for i in range(self.m)
        ]
        return result

    def matrixTranspose(self, matrix):
        return [[matrix.result[j][i] for j in range(len(matrix.result))] for i in range(len(matrix.result[0]))]

    def matrixMultiplication(self, matrix):
        if self.n != matrix.m:
            raise ValueError("Matrix A columns must equal Matrix B rows for multiplication")

        transposed = self.matrixTranspose(matrix)
        product = []
        for i in range(len(self.result)):
            product_row = []
            for j in range(len(transposed)):
                dot = sum(a * b for a, b in zip(self.result[i], transposed[j]))
                product_row.append(dot)
            product.append(product_row)
        return product

    def gaussRowAddition(self, row1, row2):
        if row1 >= self.m or row2 >= self.m:
            raise ValueError("Row index out of bounds")
        for i in range(self.n):
            self.result[row1][i] += self.result[row2][i]
        
    def gaussRowScalar(self, row1, scalar):
        for i in range(self.n):
            self.result[row1][i] = self.result[row1][i] * scalar

    
    def gaussRowSwap(self, row1, row2):
        if row1 >= self.m or row2 >= self.m:
            raise ValueError("Row index out of bounds")
        self.result[row1], self.result[row2] = self.result[row2], self.result[row1]


    def guassElimination(self):
        """ Performs Gaussian elimination to convert matrix to row-echelon form
        Handles row swaps, scaling, and elimination directly """
        # Iterate through pivot positions along the diagonal
        for pivot in range(min(self.m, self.n)):
            # Find pivot with maximum absolute value in current column
            max_row = pivot
            for row in range(pivot, self.m):
                if abs(self.result[row][pivot]) > abs(self.result[max_row][pivot]):
                    max_row = row

            # Swap rows if necessary
            if max_row != pivot:
                self.result[pivot], self.result[max_row] = self.result[max_row], self.result[pivot]

            # Skip if column is all zeros
            if self.result[pivot][pivot] == 0:
                continue

            # Normalize pivot row
            pivot_val = self.result[pivot][pivot]
            for col in range(pivot, self.n):
                self.result[pivot][col] /= pivot_val

            # Eliminate entries below pivot
            for row in range(pivot + 1, self.m):
                factor = self.result[row][pivot]
                for col in range(pivot, self.n):
                    self.result[row][col] -= factor * self.result[pivot][col]

# Example usage
test = Matrix(2, 3)
test2 = Matrix(3, 2)

test.populate(True)
test2.populate(True)

test.display()
test2.display()

# result = test.matrixMultiplication(test2)
# print("Matrix Multiplication Result:")
# for row in result:
#     print(row)

# test.gaussRowAddition(0, 1)
test.gaussRowScalar(0, 2)
test.gaussRowSwap(0, 1)
test.display()

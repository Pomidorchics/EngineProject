from math import *
from exceptions import *


class Matrix:
    def __init__(self, matrix):
        self.rows = len(matrix)
        self.columns = len(matrix[0])
        self.matrix = matrix.copy()

        for i in range(1, self.rows - 1):
            if len(matrix[i]) != self.columns:
                raise MatrixException(MatrixException.WrongMatrixSize)

    @staticmethod
    def identity(size):
        m = []

        for i in range(0, size):
            m.append([])
            for j in range(0, size):
                if i == j:
                    m[i].append(1)
                else:
                    m[i].append(0)

        return Matrix(m)

    @staticmethod
    def zero_matrix(rows, columns):
        m = []

        for i in range(0, rows):
            m.append([])
            for j in range(0, columns):
                m[i].append(0)

        return Matrix(m)

    def transpose(self):
        if isinstance(self, Matrix):
            m = Matrix.zero_matrix(self.columns, self.rows)

            for i in range(0, self.rows):
                for j in range(0, self.columns):
                    m[j][i] = self[i][j]

            return m

    def minor(self, i, j):
        m = Matrix.zero_matrix(self.rows - 1, self.columns - 1)
        ki = 0

        for s in range(0, m.rows):
            if s == i:
                ki = 1
            kj = 0
            for c in range(0, m.rows):
                if c == j:
                    kj = 1
                m[s][c] = self[s + ki][c + kj]

        return m

    def determinant(self):
        if self.rows != self.columns:
            raise MatrixException(MatrixException.Determinant)

        j, det = 0, 0
        if self.rows == 1:
            det = self[0][0]

            return det

        elif isinstance(self[0][0], (int, float)):
            for i in range(0, self.rows):
                m = self.minor(0, i)
                det = det + (-1) ** i * self[0][i] * m.determinant()

            return det

        size = self.columns
        res = Vector([[0, 0, 0]])
        for i in range(size):
            m = self.minor(0, i)
            det = m.determinant()
            res = res + (-1) ** i * det * self.matrix[0][i]

        return res

    def inverse(self):
        if self.rows != self.columns:
            raise MatrixException(MatrixException.InverseMatrixSquare)

        if self.determinant() == 0:
            raise MatrixException(MatrixException.InverseMatrixDet)

        m = Matrix.zero_matrix(self.rows, self.columns)
        k = 1
        for i in range(0, m.rows):
            for j in range(0, m.columns):
                m[i][j] = k * self.minor(i, j).determinant()
                k = -k
        m = m.transpose()
        m = m / self.determinant()

        return m

    @staticmethod
    def gram(n, vectors):
        for i in range(0, n):
            if not isinstance(vectors[i], Vector):
                raise MatrixException(MatrixException.GramListError)

            if vectors[i].is_vertical is True:
                if vectors[i].rows != n:
                    raise MatrixException(MatrixException.GramDifferentSize)

            if vectors[i].columns != n:
                raise MatrixException(MatrixException.GramDifferentSize)

        m = Matrix.zero_matrix(n, n)
        for i in range(0, n):
            for j in range(0, n):
                m[i][j] = vectors[i] % vectors[j]

        return m

    def rotate(self, axes, angle):
        angle = angle * pi / 180
        rotation_mat = Matrix.identity(self.columns)
        n = (-1) ** (axes[0] + axes[1])

        rotation_mat[axes[0]][axes[0]] = cos(angle)
        rotation_mat[axes[1]][axes[1]] = cos(angle)
        rotation_mat[axes[1]][axes[0]] = n * sin(angle)
        rotation_mat[axes[0]][axes[1]] = (-n) * sin(angle)

        return rotation_mat

    def __getitem__(self, index):
        return self.matrix[index]

    def __setitem__(self, index, new_val):
        self.matrix[index] = new_val

    def __add__(self, matrix):
        if not (isinstance(self, Vector) and isinstance(matrix, Point)) and \
                not(isinstance(matrix, Matrix)):
            raise EngineException(EngineException.WrongUsage)

        if isinstance(matrix, Matrix):
            if not (self.rows == matrix.rows and self.columns == matrix.columns):
                raise MatrixException(MatrixException.MatrixAddition)

            m = Matrix.zero_matrix(self.rows, self.columns)
            for i in range(0, self.rows):
                for j in range(0, self.columns):
                    m[i][j] = self[i][j] + matrix[i][j]

            return m

        if isinstance(self, Vector) and isinstance(matrix, Point):
            return matrix + self

    def __sub__(self, matrix):
        m = Matrix((matrix * (-1)).matrix)

        return self + m

    def __eq__(self, other):
        if self.rows == other.rows and self.columns == other.columns:
            eps = 10 ** (-4)

            return all(abs(self[i][j] - other[i][j]) < eps for i in range(0, self.rows) for j in range(0, self.columns))

        return False

    def __mul__(self, other):
        if not isinstance(other, Matrix) and not isinstance(other, (int, float)):
            raise EngineException(EngineException.WrongUsage)

        if isinstance(other, Matrix):
            if self.columns != other.rows:
                raise MatrixException(MatrixException.MatrixMultiplication)

            m = Matrix.zero_matrix(self.rows, other.columns)
            for i in range(0, self.rows):
                for j in range(0, other.columns):
                    s = 0
                    for k in range(0, self.columns):
                        s += self[i][k] * other[k][j]
                    m[i][j] = s

            return m

        elif isinstance(other, (int, float)):
            m = Matrix.zero_matrix(self.rows, self.columns)
            for i in range(0, self.rows):
                for j in range(0, self.columns):
                    m[i][j] = other * self[i][j]

            return m

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, num):
        if not isinstance(num, (int, float)):
            raise EngineException(EngineException.WrongUsage)

        if num == 0:
            raise EngineException(ZeroDivisionError)

        return self * (1 / num)


class Vector(Matrix):
    def __init__(self, array):
        self.matrix = array.copy()
        self.rows = len(array)
        self.columns = len(array[0])
        self.is_vertical = True
        if self.rows == 1:
            self.is_vertical = False

    def minor(self, i, j):
        pass

    def determinant(self):
        pass

    def inverse(self):
        pass

    def scalar_product(self, vec):
        if not (vec.rows == self.rows and vec.columns == self.columns):
            raise VectorException(VectorException.ScalarProduct)

        n = 0
        if self.is_vertical is True:
            for i in range(0, self.rows):
                n += self[i][0] * vec[i][0]
        else:
            for i in range(0, self.columns):
                n += self[0][i] * vec[0][i]

        return n

    def vector_product(self, vec):
        if not isinstance(vec, Vector):
            raise VectorException(VectorException.WrongUsage)

        if not (self.rows == vec.rows == 3 or
                self.columns == vec.columns == 3):
            raise VectorException(VectorException.VectorProduct)

        basis = [Vector([[1, 0, 0]]), Vector([[0, 1, 0]]), Vector([[0, 0, 1]])]
        m = Matrix([[basis[0], basis[1], basis[2]], self.matrix[0], vec.matrix[0]])

        return m.determinant()

    def length(self):
        l = 0
        if self.is_vertical is True:
            for i in range(0, self.rows):
                l += self[i][0] ** 2
        else:
            for i in range(0, self.columns):
                l += self[0][i] ** 2

        return l ** 0.5

    def __mod__(self, vec):
        return self.scalar_product(vec)

    def __pow__(self, vec):
        return self.vector_product(vec)


class VectorSpace:
    def __init__(self, basis):
        self.basis = basis.copy()

    def scalar_product(self, vec1, vec2):
        return ((vec1 * Matrix.gram(len(self.basis), self.basis)) * vec2.transpose())[0][0]


class Point:
    def __init__(self, coordinates):
        self.coordinates = coordinates.copy()

    def __eq__(self, other):
        if len(self.coordinates) == len(other.coordinates):
            for i in range(0, len(self.coordinates)):
                if self[i] != other[i]:
                    return False
            return True
        return False

    def __getitem__(self, index):
        return self.coordinates[index]

    def __setitem__(self, index, new_val):
        self.coordinates[index] = new_val

    def __add__(self, vec):
        if not isinstance(vec, Vector):
            raise EngineException(EngineException.WrongUsage)

        if len(self.coordinates) != vec.rows and vec.is_vertical is True or \
                len(self.coordinates) != vec.columns and vec.is_vertical is False:
            raise PointException(PointException.PointAndVectorAdd)

        if vec.is_vertical is True:
            return Point([self[i] + vec[i][0] for i in range(0, len(self.coordinates))])

        return Point([self[i] + vec[0][i] for i in range(0, len(self.coordinates))])

    def __mul__(self, num):
        pass


class CoordinateSystem:
    def __init__(self, point, basis):
        self.point = point
        self.basis = basis.copy()


def bilinear_form(matrix, v1, v2):
    if not v1.is_vertical:
        v1 = v1.transpose()
    if not v2.is_vertical:
        v2 = v2.transpose()

    if matrix.rows != v1.rows or matrix.columns != v2.rows:
        raise EngineException(EngineException.BilinearForm)

    s = 0
    for i in range(0, matrix.rows):
        for j in range(0, matrix.columns):
            s += matrix[i][j] * v1[i][0] * v2[j][0]

    return s

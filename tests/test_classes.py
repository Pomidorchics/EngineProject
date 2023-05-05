import os
import sys
import pytest

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from classes import *


class TestMatrices:
    def test_init_matrix(self):
        m = Matrix([[0, 3],
                    [5, 2]])
        act = isinstance(m, Matrix)

        assert act

    def test_matrix_size(self):
        m1 = Matrix([[9, 8, 7],
                     [6, 4, 0],
                     [2, 0, 1],
                     [[1, 2, 3]]])
        m2 = Matrix([[1],
                     [2]])

        act = m1.rows == 4 and \
            m1.columns == 3 and \
            m2.rows == 2 and \
            m2.columns == 1

        assert act

    def test_matrices_equality(self):
        m1 = Matrix([[1, 2],
                     [3, 4]])
        m2 = Matrix([[1, 2],
                     [3, 4]])
        m3 = Matrix([[4, 3],
                     [2, 1]])

        act = m1 == m2 != m3

        assert act

    def test_wrong_matrix_size_exc(self):
        with pytest.raises(EngineException):
            Matrix([[1, 2],
                    [3],
                    [4, 5]])

    def test_getitem(self):
        m = Matrix([[1, 2],
                    [3, 4]])
        act = m[0] == [1, 2] and m[0][0] == 1

        assert act

    def test_setitem(self):
        m = Matrix([[1, 2],
                    [3, 4]])
        m[0][1] = 5
        act = m[0][1] == 5

        assert act

    def test_matrices_addition(self):
        m1 = Matrix([[1, 2],
                     [3, 4]])

        m2 = Matrix([[0, 5],
                     [1, 6]])

        m3 = Matrix([[0, 0],
                     [0, 0]])

        act = m1 + m2 == m2 + m1 == Matrix([[1, 7],
                                            [4, 10]]) and \
            m1 + m3 == m3 + m1 == Matrix([[1, 2],
                                          [3, 4]])

        assert act

    def test_matrix_and_number_addition_exc(self):
        m = Matrix([[1, 2],
                    [3, 4]])

        with pytest.raises(EngineException):
            m + 5

    def test_matrices_addition_exc(self):
        m1 = Matrix([[1, 2],
                     [3, 4]])

        m2 = Matrix([[4, 5, 6],
                     [1, 2, 3]])

        with pytest.raises(EngineException):
            m1 + m2

    def test_matrices_subtraction(self):
        m1 = Matrix([[9, 8],
                     [7, 6]])

        m2 = Matrix([[1, 2],
                     [3, 4]])

        act = m1 - m2 == Matrix([[8, 6],
                                 [4, 2]])

        assert act

    def test_mul_matrix_and_number(self):
        m = Matrix([[1, 2],
                    [3, 4]])

        res_m = Matrix([[4, 8],
                        [12, 16]])

        act = m * 4 == 4 * m == res_m

        assert act

    def test_mul_matrices(self):
        m1 = Matrix([[1, 2],
                     [3, 4]])

        m2 = Matrix([[0, 3],
                     [5, 2]])

        m3 = Matrix([[10, 7],
                     [20, 17]])

        m4 = Matrix([[1, 2, 3],
                     [4, 5, 6]])

        m5 = Matrix([[1, 4, 6, 7],
                     [0, 1, 5, 3],
                     [4, 0, 1, 2]])

        m6 = Matrix([[13, 6, 19, 19],
                     [28, 21, 55, 55]])

        act = m1 * m2 == m3 and m4 * m5 == m6

        assert act

    def test_mul_matrices_exc(self):
        m1 = Matrix([[1, 2, 3, 4],
                     [2, 3, 4, 0]])
        m2 = Matrix([[5, 4, 3],
                     [2, 1, 0],
                     [9, 8, 7]])

        with pytest.raises(EngineException):
            m1 * m2

    def test_div_matrix_and_number(self):
        m = Matrix([[5, 10],
                    [15, 20]])
        n = 5
        act = m / n == Matrix([[1, 2],
                               [3, 4]])

        assert act

    def test_div_matrices_exc(self):
        m1 = Matrix([[1, 2],
                     [3, 4]])
        m2 = Matrix([[3, 0],
                     [7, 2]])

        with pytest.raises(EngineException):
            m1 / m2

    def test_identity_matrix(self):
        m = Matrix.identity(5)
        act = m == Matrix([[1, 0, 0, 0, 0],
                           [0, 1, 0, 0, 0],
                           [0, 0, 1, 0, 0],
                           [0, 0, 0, 1, 0],
                           [0, 0, 0, 0, 1]])

        assert act

    def test_zero_matrix(self):
        m = Matrix.zero_matrix(4, 4)
        act = m == Matrix([[0, 0, 0, 0],
                           [0, 0, 0, 0],
                           [0, 0, 0, 0],
                           [0, 0, 0, 0]])

        assert act

    def test_transpose_matrix(self):
        m = Matrix([[1, 2, 3, 4],
                    [5, 6, 7, 8],
                    [3, 4, 5, 6]])

        act = m.transpose() == Matrix([[1, 5, 3],
                                       [2, 6, 4],
                                       [3, 7, 5],
                                       [4, 8, 6]])

        assert act

    def test_determinant(self):
        m1 = Matrix.identity(3)
        m2 = Matrix.zero_matrix(4, 4)
        m3 = Matrix([[1, 4],
                     [8, 1]])
        act = m1.determinant() == 1 and \
              m2.determinant() == 0 and \
              m3.determinant() == -31

        assert act

    def test_determinant_exc(self):
        m = Matrix([[1, 2, 3],
                    [4, 5, 6]])

        with pytest.raises(EngineException):
            m.determinant()

    def test_inverse_matrix(self):
        m = Matrix([[1, 2, 3],
                    [4, 5, 6],
                    [7, 10, 9]])
        act = m * m.inverse() == m.inverse() * m == Matrix.identity(3)

        assert act

    def test_inverse_exc1(self):
        m = Matrix([[7, 11, 12],
                    [3, 8, 0]])

        with pytest.raises(EngineException):
            m.inverse()

    def test_inverse_exc2(self):
        m = Matrix([[1, 2, 3],
                    [4, 5, 6],
                    [7, 8, 9]])

        with pytest.raises(EngineException):
            m.inverse()

    def test_gram(self):
        vec1 = Vector([[1, 2, 3]])
        vec2 = Vector([[6, 0, 2]])
        vec3 = Vector([[4, 8, 0]])
        m = Matrix.gram(3, [vec1, vec2, vec3])
        act = m == Matrix([[14, 12, 20],
                           [12, 40, 24],
                           [20, 24, 80]])

        assert act

    def test_gram_exc(self):
        with pytest.raises(EngineException):
            Matrix.gram(4, [Vector([[1, 2, 3, 4]]),
                            Vector([[3, 7, 1]]),
                            Vector([[5, 0, 1, 6]]),
                            Vector([[3, 4]])])

    def test_rotate(self):
        m = Matrix([[1, 2, 3],
                    [4, 0, 1],
                    [5, 3, 8]])

        act = m.rotate([0, 1], 45) == Matrix([[0.707107, 0.707107, 0],
                                             [-0.707107, 0.707107, 0],
                                             [0, 0, 1]])

        assert act


class TestVectors:
    def test_init_vector(self):
        vec = Vector([[1, 2, 3, 4]])
        act = isinstance(vec, Vector)

        assert act

    def test_vector_size(self):
        vec1 = Vector([[1, 2, 3, 4]])
        vec2 = Vector([[2, 5, 0]])
        act = vec1.rows == 1 and vec1.columns == 4 and \
            vec2.rows == 1 and vec2.columns == 3

        assert act

    def test_vec_equality(self):
        vec1 = Vector([[1, 2, 3]])
        vec2 = Vector([[1, 2, 3]])
        vec3 = Vector([[4, 5, 6]])
        act = vec1 == vec2 != vec3

        assert act

    def test_vec_transpose(self):
        vec = Vector([[1, 2, 3]])
        act = vec.transpose().matrix == [[1], [2], [3]]

        assert act

    def test_scalar_product(self):
        vec1 = Vector([[1, 2, 3, 4]])
        vec2 = Vector([[4, 6, 0, 1]])
        vec3 = Vector([[2, 7, 8, 0]])
        act = vec1 % vec2 == 20 and \
            vec1 % vec3 == 40 and \
            vec2 % vec3 == 50

        assert act

    def test_scalar_product_exc(self):
        vec1 = Vector([[1, 2, 3, 4]])
        vec2 = Vector([[1, 4, 9]])

        with pytest.raises(EngineException):
            vec1 % vec2

    def test_vector_product(self):
        vec1 = Vector([[1, 2, 3]])
        vec2 = Vector([[3, 7, 8]])
        act = vec1 ** vec2 == Vector([[-5, 1, 1]])

        assert act

    def test_vector_product_exc1(self):
        vec1 = Vector([[1, 2, 3, 4]])
        vec2 = Vector([[1, 2]])

        with pytest.raises(EngineException):
            vec1 ** vec2

    def test_vector_product_exc2(self):
        vec1 = Vector([[1, 2, 3]])
        mat = Matrix([[5, 6, 7]])

        with pytest.raises(EngineException):
            vec1 ** mat

    def test_mul_vec_and_num(self):
        vec = Vector([[1, 2, 3]])
        act = vec * 5 == 5 * vec == Vector([[5, 10, 15]])

        assert act

    def test_vectors_addition(self):
        vec1 = Vector([[1, 2, 3]])
        vec2 = Vector([[3, 0, 1]])
        act = vec1 + vec2 == vec2 + vec1 == Vector([[4, 2, 4]])

        assert act

    def test_vec_and_number_addition_exc(self):
        vec = Vector([[1, 2, 3]])

        with pytest.raises(EngineException):
            vec + 5

    def test_vec_addition_exc(self):
        vec1 = Vector([[1, 2, 3]])
        vec2 = Vector([[1, 2, 3, 4]])

        with pytest.raises(EngineException):
            vec1 + vec2

    def test_vec_subtraction(self):
        vec1 = Vector([[5, 5, 5]])
        vec2 = Vector([[1, 2, 3]])
        act = vec1 - vec2 == Vector([[4, 3, 2]]) and \
            vec2 - vec1 == Vector([[-4, -3, -2]])

        assert act

    def test_length_vec(self):
        vec1 = Vector([[1, 0, 0]])
        vec2 = Vector([[3, 4, 0]])
        act = vec1.length() == 1 and \
            vec2.length() == 5

        assert act


class TestPoints:
    def test_init_point(self):
        p = Point([1, 2, 3])
        act = isinstance(p, Point)

        assert act

    def test_eq_points(self):
        p1 = Point([1, 2, 3])
        p2 = Point([1, 2, 3])
        p3 = Point([3, 4, 5])
        act = p1 == p2 != p3

        assert act

    def test_getitem(self):
        p = Point([1, 2, 3, 4])
        act = p[0] == 1 and p[3] == 4

        assert act

    def test_setitem(self):
        p = Point([1, 2, 3, 4])
        p[0] = 5
        act = p[0] == 5

        assert act

    def test_point_and_vec_addition(self):
        p = Point([1, 2, 3])
        vec = Vector([[5, 6, 7]])
        act = p + vec == vec + p == Point([6, 8, 10])

        assert act


class TestCoordinateSystem:
    def test_init_coord_system(self):
        p = Point([0, 0, 0])
        bas1 = Vector([[1, 0, 0]])
        bas2 = Vector([[0, 1, 0]])
        bas3 = Vector([[0, 0, 1]])
        cs = CoordinateSystem(p, [bas1, bas2, bas3])
        act = isinstance(cs, CoordinateSystem)

        assert act


class TestBilinearForm:
    def test_bilinear_form(self):
        vec1 = Vector([[3, 7, 1]])
        vec2 = Vector([[0, 2, 5]])
        m = Matrix([[3, 6, 1],
                    [9, 0, 3],
                    [0, 5, 1]])
        act = bilinear_form(m, vec1, vec2) == 171

        assert act

    def test_bilinear_form_exc(self):
        vec1 = Vector([[3, 7, 1]])
        vec2 = Vector([[0, 2, 5]])

        m = Matrix([[3, 6, 1],
                    [9, 0, 3]])

        with pytest.raises(EngineException):
            bilinear_form(m, vec1, vec2)


class TestVectorSpace:
    def test_init_vector_space(self):
        bas1 = Vector([[1, 0, 0]])
        bas2 = Vector([[0, 1, 0]])
        bas3 = Vector([[0, 0, 1]])
        vs = VectorSpace([bas1, bas2, bas3])
        act = isinstance(vs, VectorSpace)

        assert act

    def test_scalar_product(self):
        bas1 = Vector([[1, 0, 0]])
        bas2 = Vector([[0, 1, 0]])
        bas3 = Vector([[0, 0, 1]])
        vs = VectorSpace([bas1, bas2, bas3])
        vec1 = Vector([[1, 7, 4]])
        vec2 = Vector([[8, 1, 0]])
        act = vs.scalar_product(vec1, vec2) == 15

        assert act









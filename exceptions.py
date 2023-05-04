class EngineException(Exception):
    WrongUsage = "this operation is not allowed with this data"
    BilinearForm = "wrong size of matrix and vectors"


class MatrixException(EngineException):
    WrongMatrixSize = "wrong matrix size"
    MatrixMultiplication = "it is impossible to multiply matrices"
    Determinant = "the matrix is non-square so it is impossible to calculate the determinant"
    InverseMatrixDet = "the determinant is zero so it is impossible to calculate the inverse matrix"
    InverseMatrixSquare = "the matrix is non-square so it is impossible to calculate the inverse matrix"
    MatrixAddition = "different size of matrices so it is impossible to calculate the scalar product"
    GramDifferentSize = "different size of vectors"
    GramListError = "the list must only contain vectors"


class VectorException(EngineException):
    ScalarProduct = "different size of vectors so it is impossible to calculate the scalar product"
    VectorProduct = "vectors must have 3 coordinates"


class PointException(EngineException):
    PointAndVectorAdd = "different size of point and vector so it is impossible to add them"

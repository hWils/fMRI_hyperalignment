"""
Solve the orthogonal Procrustes problem.

"""
import numpy as np

__all__ = ['orthogonal_procrustes']

def orthogonal_procrustes_edited(A, B, check_finite=True):
    """
    Compute the matrix solution of the orthogonal Procrustes problem.

    Given matrices A and B of equal shape, find an orthogonal matrix R
    that most closely maps A to B using the algorithm given in [1]_.

    Parameters
    ----------
    A : (M, N) array_like
        Matrix to be mapped.
    B : (M, N) array_like
        Target matrix.
    check_finite : bool, optional
        Whether to check that the input matrices contain only finite numbers.
        Disabling may give a performance gain, but may result in problems
        (crashes, non-termination) if the inputs do contain infinities or NaNs.

    Returns
    -------
    R : (N, N) ndarray
        The matrix solution of the orthogonal Procrustes problem.
        Minimizes the Frobenius norm of ``(A @ R) - B``, subject to
        ``R.T @ R = I``.
    scale : float
        Sum of the singular values of ``A.T @ B``.

    Raises
    ------
    ValueError
        If the input array shapes don't match or if check_finite is True and
        the arrays contain Inf or NaN.

    Notes
    -----
    Note that unlike higher level Procrustes analyses of spatial data, this
    function only uses orthogonal transformations like rotations and
    reflections, and it does not use scaling or translation.

    .. versionadded:: 0.15.0

    References
    ----------
    .. [1] Peter H. Schonemann, "A generalized solution of the orthogonal
           Procrustes problem", Psychometrica -- Vol. 31, No. 1, March, 1996.

    Examples
    --------
    >>> from scipy.linalg import orthogonal_procrustes
    >>> A = np.array([[ 2,  0,  1], [-2,  0,  0]])

    Flip the order of columns and check for the anti-diagonal mapping
    
    >>> R, sca = orthogonal_procrustes(A, np.fliplr(A))
    >>> R
    array([[-5.34384992e-17,  0.00000000e+00,  1.00000000e+00],
           [ 0.00000000e+00,  1.00000000e+00,  0.00000000e+00],
           [ 1.00000000e+00,  0.00000000e+00, -7.85941422e-17]])
    >>> sca
    9.0

    """
    print("edited version")
    print("original orthogonal procrustes method here")
    if check_finite:
        A = np.asarray_chkfinite(A)
        B = np.asarray_chkfinite(B)
    else:
        A = np.asanyarray(A)
        B = np.asanyarray(B)
    if A.ndim != 2:
        raise ValueError('expected ndim to be 2, but observed %s' % A.ndim)
    if A.shape != B.shape:
        raise ValueError('the shapes of A and B differ (%s vs %s)' % (
            A.shape, B.shape))
    A = A.astype(np.float32) #HW
    B = B.astype(np.float32)  #HW
    # Be clever with transposes, with the intention to save memory.

    transposed_B = B.T

    dotty = np.einsum('ij,jk', transposed_B, A)
    transposeddotty = dotty.T
    print("Transposed dotty is byte size ", transposeddotty.itemsize)
    u, w, vt = np.linalg.svd(transposeddotty, 'A','A')
    
    R = np.dot(u, vt)#  lapack_driver='gesvd'HW THISS WORKS RL

    scale = w.sum()  # RL
    print("sum  is complete")
    return R, scale

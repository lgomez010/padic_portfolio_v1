import numpy as np

def test_strong_traingle_inequality():
    """
    Tests if a given matrix satisfies the ultrametric inequality
    d(x,y) <= max(d(x,z),d(z,y))
    """
    dummy_ultrametric = np.array([
        [0.0, 1.0, 5.0],
        [1.0, 0.0, 5.0],
        [5.0, 5.0, 0.0]
    ])

    matrix_size = dummy_ultrametric.shape[0]

    for x in range(matrix_size):
        for y in range(matrix_size):
            for z in range(matrix_size):
                if x!=y and y!=z and x!=z:
                    d_xy = dummy_ultrametric[x,y]
                    d_xz = dummy_ultrametric[x,z]
                    d_zy = dummy_ultrametric[z,y]

                    allowed_max = max(d_xz, d_zy)

                    assert d_xy <= allowed_max + 1e-9, \
                        f"Math failed! Stocks {x},{y},{z} broke the rule: {d_xy} is not <= {allowed_max}"
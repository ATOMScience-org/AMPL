import numpy as np
import atomsci.ddm.pipeline.featurization as feat

correct_v = np.array([[np.nan, 4.5622, np.nan, np.nan],
            [np.nan, np.nan, 5.0905, np.nan],
            [np.nan, np.nan, 4.3972, np.nan],
            [np.nan, 5.0177, np.nan, np.nan],
            [5.8538, np.nan, np.nan, np.nan],
            [6.2218, np.nan, np.nan, np.nan]], dtype=float)

correct_v_0 = np.array([[0, 4.5622, 0, 0],
            [0, 0, 5.0905, 0],
            [0, 0, 4.3972, 0],
            [0, 5.0177, 0, 0],
            [5.8538, 0, 0, 0],
            [6.2218, 0, 0, 0]], dtype=float)

correct_w = np.array([[0.,1.,0.,0.],
            [0., 0., 1., 0.],
            [0., 0., 1., 0.],
            [0., 1., 0., 0.],
            [1., 0., 0., 0.],
            [1., 0., 0., 0.]], dtype=float)

def test_nan_make_weights():
    temp_vals = np.array([[np.nan, 4.5622, np.nan, np.nan],
       [np.nan, np.nan, 5.0905, np.nan],
       [np.nan, np.nan, 4.3972, np.nan],
       [np.nan, 5.0177, np.nan, np.nan],
       [5.8538, np.nan, np.nan, np.nan],
       [6.2218, np.nan, np.nan, np.nan]], dtype=float)
 
    v, w = feat.make_weights(temp_vals)

    assert np.array_equal(v, correct_v, equal_nan=True)
    assert np.max(np.abs(w-correct_w)) < 1e-5

def test_str_make_weights():
    temp_vals = np.array([['', 4.5622, '', ''],
                ['', '', 5.0905, ''],
                ['', '', 4.3972, ''],
                ['', 5.0177, '', ''],
                [5.8538, '', '', ''],
                [6.2218, '', '', '']])

    v, w = feat.make_weights(temp_vals)

    assert np.array_equal(v, correct_v, equal_nan=True)
    assert np.max(np.abs(w-correct_w)) < 1e-5

def test_str_make_weights_class():
    temp_vals = np.array([['', 4.5622, '', ''],
                ['', '', 5.0905, ''],
                ['', '', 4.3972, ''],
                ['', 5.0177, '', ''],
                [5.8538, '', '', ''],
                [6.2218, '', '', '']])

    v, w = feat.make_weights(temp_vals, is_class=True)

    assert np.array_equal(v, correct_v_0, equal_nan=True)
    assert np.max(np.abs(w-correct_w)) < 1e-5

if __name__ == '__main__':
    test_str_make_weights()
    test_str_make_weights_class()
    test_nan_make_weights()

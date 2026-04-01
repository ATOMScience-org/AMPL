import atomsci.ddm.pipeline.model_pipeline as mp
import numpy as np

def test_AD_kmean_dist():
    a = np.array([
        [1, 1, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 1, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 1, 0, 0, 0, 0, 0],

        [0, 0, 0, 0, 0, 0, 1, 1, 1],
        [0, 0, 0, 0, 0, 1, 0, 1, 1],
        [0, 0, 0, 0, 1, 0, 0, 1, 1],
    ])

    b = np.array([
        [0, 0, 1, 1, 0, 0, 0, 0, 0],
        [0, 1, 1, 0, 0, 0, 0, 0, 0],
    ])

    ad_index, nn_index = mp.calc_AD_kmean_dist(a, b, 2, dist_metric='jaccard')
    # the actual value of both ad_indexes is .999 repeating
    assert np.allclose(ad_index, [1, 1])
    assert (set(nn_index[0])=={1,2}) and (set(nn_index[1])=={0,1})

def test_calc_AD_kmean_local_density():
    a = np.array([
        [1, 1, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 1, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 1, 0, 0, 0, 0, 0],

        [0, 0, 0, 0, 0, 0, 1, 1, 1],
        [0, 0, 0, 0, 0, 1, 0, 1, 1],
        [0, 0, 0, 0, 1, 0, 0, 1, 1],
    ])

    b = np.array([
        [0, 0, 1, 1, 0, 0, 0, 0, 0],
        [0, 1, 1, 0, 0, 0, 0, 0, 0],
    ])

    ad_index = mp.calc_AD_kmean_local_density(a, b, 2, dist_metric='jaccard')
    assert list(ad_index) == [1., 1.]
 

if __name__ == '__main__':
    test_calc_AD_kmean_local_density()
    test_AD_kmean_dist()
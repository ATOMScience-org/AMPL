import atomsci.ddm.pipeline.model_wrapper as mw
import logging
import time
from argparse import Namespace
from atomsci.ddm.utils import llnl_utils
import pytest

@pytest.mark.slurm_required
def test_LCTimerIterator_too_long():
    if not llnl_utils.is_lc_system():
        assert True
        return
    
    # make fake parameters
    params = Namespace(max_epochs=100, slurm_time_limit=3)

    # make a logger
    log = logging.getLogger('ATOM')

    # make a fake pipeline
    pipeline = Namespace(start_time = time.time())

    lcit = mw.LCTimerIterator(params, pipeline, log)
    for ei in lcit:
        # each iteration takes 10 seconds. Since the time limit is
        # set to 3 minute, this won't reach max_epochs and should quit
        # in at most 18 iterations
        time.sleep(10)


    assert params.max_epochs <= 18

def test_LCTimerIterator_finishes_all_epochs():
    # make fake parameters
    params = Namespace(max_epochs=10, slurm_time_limit=60)

    # make a logger
    log = logging.getLogger('ATOM')

    # make a fake pipeline
    pipeline = Namespace(start_time = time.time())

    lcit = mw.LCTimerIterator(params, pipeline, log)
    for ei in lcit:
        # each iteration takes 1 second. Since the time limit is
        # set to 1 minute, this will finish all epochs
        time.sleep(1)

    assert params.max_epochs == 10

@pytest.mark.slurm_required
def test_LCTimerKFoldIterator_too_long():
    if not llnl_utils.is_lc_system():
        assert True
        return

    # make fake parameters
    params = Namespace(max_epochs=100, slurm_time_limit=3)

    # make a logger
    log = logging.getLogger('ATOM')

    # make a fake pipeline
    pipeline = Namespace(start_time = time.time())

    lcit = mw.LCTimerKFoldIterator(params, pipeline, log)

    for ei in lcit:
        # each iteration takes 10 seconds. Since the time limit is
        # set to 3 minute, this won't reach max_epochs and should quit
        # in less than 18 iterations
        time.sleep(10)

    assert params.max_epochs <= 18

def test_LCTimerKFoldIterator_finishes_all_epochs():
    # make fake parameters
    params = Namespace(max_epochs=10, slurm_time_limit=60)

    # make a logger
    log = logging.getLogger('ATOM')

    # make a fake pipeline
    pipeline = Namespace(start_time = time.time())

    lcit = mw.LCTimerKFoldIterator(params, pipeline, log)
    for ei in lcit:
        # each iteration takes 1 second. Since the time limit is
        # set to 1 minute, this will finish all epochs
        time.sleep(1)

    assert params.max_epochs == 10

if __name__ == '__main__':
    test_LCTimerIterator_too_long()
    test_LCTimerIterator_finishes_all_epochs()
    test_LCTimerKFoldIterator_too_long()
    test_LCTimerKFoldIterator_finishes_all_epochs()

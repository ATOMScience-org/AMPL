"""Utility functions for running and managing jobs on LC machines."""

import os
import subprocess
import time


def is_lc_system():
    """Use heuristic to determine if we're running on an LC system.

    Args:
        none

    Returns:
        result: True if it appears we're running on an LC machine.
    """
    # Currently the vast filesystem is the only one accessible from all unclassified 
    # LC machines (including lassen). This of course could change.

    return os.path.exists('/p/vast1')


def get_command_output(cmd):
    """Runs the given shell command in a subprocess and returns its output as a string.
    Used by throttle_jobs function.

    Args:
        cmd: Command to run

    Returns:
        output: Output of command
    """
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
    (output, err) = p.communicate()
    p.wait()
    return output.decode('utf-8').rstrip('\n')

def get_my_username():
    """Returns the username of the effective userid of the current process.
    Used by throttle_jobs function.

    Args:
        none

    Returns:
        username (str): The current effective username
    """
    return get_command_output('whoami')


def throttle_jobs(max_jobs, retry_time=10, my_username=None, verbose=True):
    """Checks the number of SLURM jobs currently queued or running under the current userid.
    Returns immediately if this number is less than max_jobs; otherwise, loops indefinitely,
    checking the job count every retry_time seconds. Returns when the job count drops below
    max_jobs. Call this function before queueing a batch job in order to implement a self-
    throttling mechanism.

    Args:
        max_jobs (int): Number of jobs we are allowing ourselves to have queued or running.

        retry_time (int): Number of seconds to wait between job count checks.

        my_username (str): Effective username of the caller; defaults to output of 'whoami'
        command.

    Returns:
        none
    """
    if my_username is None:
        my_username = get_my_username()

    command = 'squeue -u %s | wc -l' % my_username
    njobs = int(get_command_output(command)) - 1
    while njobs >= max_jobs:
        if verbose:
            print("%d jobs in queue, sleeping" % njobs)
        time.sleep(retry_time)
        njobs = int(get_command_output(command)) - 1

import os
import time

from config import PROJECT_ROOT


def save_log(info):
    log_dir = os.path.join(PROJECT_ROOT, 'log')
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    curr_time_str = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    log_path = os.path.join(log_dir, '%s.log' % curr_time_str)
    with open(log_path, 'w') as f:
        f.write(info)

import pytest
from stonesoup.serialise import YAML
import sys
import numpy as np

from ..runmanagercore import RunManagerCore

rmc = RunManagerCore()

test_config = "tests\\test_configs\\test_config_all.yaml"
test_config_nomm = "tests\\test_configs\\test_config_nomm.yaml"
test_config_nogt = "tests\\test_configs\\test_config_nogt.yaml"
test_config_trackeronly = "tests\\test_configs\\test_config_trackeronly.yaml"
test_json = "tests\\test_configs\\dummy.json"

def test_read_json():
    test_json_data = rmc.read_json(test_json)
    assert type(test_json_data) is dict

def test_run():
    pass

def test_run_simulation():
    pass

def test_set_trackers():
    test_combo = [{'SingleTargetTracker.initiator.initiator.prior_state.num_particles': 500},
                  {'SingleTargetTracker.initiator.initiator.prior_state.num_particles': 540},
                  {'SingleTargetTracker.initiator.initiator.prior_state.num_particles': 580},
                  {'SingleTargetTracker.initiator.initiator.prior_state.num_particles': 620},
                  {'SingleTargetTracker.initiator.initiator.prior_state.num_particles': 660},
                  {'SingleTargetTracker.initiator.initiator.prior_state.num_particles': 700}]

    with open(test_config, 'r') as file:
        tracker, gt, mm, _ = rmc.read_config_file(file)
    file.close()

    trackers, ground_truths, metric_managers = rmc.set_trackers(test_combo,
                                                                tracker, gt, mm)

    assert type(trackers) is list
    assert type(ground_truths) is list
    assert type(metric_managers) is list

    assert len(trackers) > 0
    assert "tracker" in str(type(trackers[0]))
    assert ground_truths[0] == trackers[0].detector.groundtruth
    assert "metricgenerator" in str(type(metric_managers[0]))

def test_set_param():

    with open(test_config, 'r') as file:
        tracker, gt, mm, _ = rmc.read_config_file(file)
    file.close()

    test_split_path = ['initiator', 'initiator', 'prior_state', 'num_particles']
    test_value = 250

    assert test_split_path[-1] not in dir(tracker.initiator.initiator.prior_state)

    rmc.set_param(test_split_path, tracker, test_value)

    assert test_split_path[-1] in dir(tracker.initiator.initiator.prior_state)
    assert tracker.initiator.initiator.prior_state.num_particles is test_value

def test_read_config_file():
    # Config with all tracker, grountruth, metric manager
    with open(test_config, 'r') as file:
        tracker, gt, mm, _ = rmc.read_config_file(file)
    assert "tracker" in str(type(tracker))
    assert gt == tracker.detector.groundtruth
    assert "metricgenerator" in str(type(mm))
    file.close()

def test_read_config_file_nomm():
    # Config with tracker and groundtruth but no metric manager
    with open(test_config_nomm, 'r') as file:
        tracker, gt, mm, _ = rmc.read_config_file(file)
    assert "tracker" in str(type(tracker))
    assert gt == tracker.detector.groundtruth
    assert mm is None
    file.close()

def test_read_config_file_nogt():
    # Config with tracker and metric manager but no groundtruth
    with open(test_config_nogt, 'r') as file:
        tracker, gt, mm, _ = rmc.read_config_file(file)
    assert "tracker" in str(type(tracker))
    assert gt is None
    assert "metricgenerator" in str(type(mm))
    file.close()

def test_read_config_file_tracker_only():
    # Config with tracker only
    with open(test_config_trackeronly, 'r') as file:
        tracker, gt, mm, _ = rmc.read_config_file(file)
    assert "tracker" in str(type(tracker))
    assert gt is None
    assert mm is None
    file.close()

def test_read_config_file_tracker_only():
    # Config with tracker only
    with open(test_config_trackeronly, 'r') as file:
        tracker, gt, mm, _ = rmc.read_config_file(file)
    assert "tracker" in str(type(tracker))
    assert gt is None
    assert mm is None
    file.close()

def test_read_config_file_csv():
    # Config with tracker only
    with open(test_config, 'r') as file:
        _, _, _, csv_data = rmc.read_config_file(file)

    assert type(csv_data) is np.ndarray
    file.close()

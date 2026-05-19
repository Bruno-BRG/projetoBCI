from brainbridge_v2.infrastructure.ml.physionet_eegmmidb_protocol import (
    LEFT_RIGHT_RUNS,
    UNSUPPORTED_LEFT_RIGHT_RUNS,
    describe_run,
    get_run_protocol,
    infer_physionet_run,
    is_left_right_training_file,
)


def test_physionet_protocol_marks_only_unilateral_fist_runs_as_left_right():
    assert LEFT_RIGHT_RUNS == frozenset({3, 4, 7, 8, 11, 12})
    assert UNSUPPORTED_LEFT_RIGHT_RUNS == frozenset({1, 2, 5, 6, 9, 10, 13, 14})


def test_physionet_protocol_maps_event_labels_by_run_type():
    unilateral = get_run_protocol(3)
    bilateral = get_run_protocol(5)
    baseline = get_run_protocol(1)

    assert unilateral.t1_label == "left"
    assert unilateral.t2_label == "right"
    assert unilateral.usable_for_left_right is True

    assert bilateral.t1_label == "both_fists"
    assert bilateral.t2_label == "both_feet"
    assert bilateral.usable_for_left_right is False

    assert baseline.t1_label is None
    assert baseline.t2_label is None
    assert baseline.usable_for_left_right is False


def test_physionet_protocol_infers_run_from_converted_csv_name():
    path = "data/eegmmidb/1.0.0/S001/S001R04_csv_openbci.csv"

    assert infer_physionet_run(path) == 4
    assert is_left_right_training_file(path) is True
    assert "T1=left" in describe_run(path)


def test_physionet_protocol_excludes_hand_foot_runs_and_allows_generic_patient_csv():
    assert is_left_right_training_file("data/S001/S001R06_csv_openbci.csv") is False
    assert is_left_right_training_file("recordings/patient_5_session.csv") is True

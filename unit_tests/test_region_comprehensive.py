#!/usr/bin/env python3
"""
Comprehensive test suite for RegionBasedMatching implementation
"""

import numpy as np
from panoptica.panoptica_evaluator import panoptic_evaluate
from panoptica.instance_matcher import RegionBasedMatching
from panoptica.instance_approximator import ConnectedComponentsInstanceApproximator
from panoptica.metrics import Metric
from panoptica.utils.constants import CCABackend
from panoptica.utils.processing_pair import SemanticPair

def test_scenario_1_basic():
    """Test basic case with non-overlapping regions"""
    print("Test 1: Basic non-overlapping regions")

    gt = np.zeros((30, 30, 10), dtype=np.int32)
    pred = np.zeros((30, 30, 10), dtype=np.int32)

    # GT regions
    gt[5:15, 5:15, 2:8] = 1
    gt[20:25, 20:25, 2:8] = 2

    # Pred regions - slightly offset
    pred[6:16, 6:16, 3:9] = 1
    pred[19:24, 19:24, 3:9] = 2

    return gt, pred

def test_scenario_2_overlapping():
    """Test case with overlapping predictions"""
    print("Test 2: Overlapping predictions")

    gt = np.zeros((30, 30, 10), dtype=np.int32)
    pred = np.zeros((30, 30, 10), dtype=np.int32)

    # GT regions
    gt[5:15, 5:15, 2:8] = 1
    gt[20:25, 20:25, 2:8] = 2

    # Overlapping predictions
    pred[8:18, 8:18, 3:9] = 1  # Overlaps with both GT regions
    pred[21:26, 21:26, 3:9] = 2

    return gt, pred

def test_scenario_3_empty_prediction():
    """Test case with no predictions"""
    print("Test 3: Empty predictions")

    gt = np.zeros((30, 30, 10), dtype=np.int32)
    pred = np.zeros((30, 30, 10), dtype=np.int32)

    # Only GT regions, no predictions
    gt[5:15, 5:15, 2:8] = 1
    gt[20:25, 20:25, 2:8] = 2

    return gt, pred

def test_scenario_4_extra_predictions():
    """Test case with more predictions than GT regions"""
    print("Test 4: Extra predictions")

    gt = np.zeros((40, 40, 10), dtype=np.int32)
    pred = np.zeros((40, 40, 10), dtype=np.int32)

    # GT regions
    gt[5:15, 5:15, 2:8] = 1

    # Multiple predictions
    pred[6:16, 6:16, 3:9] = 1  # Close to GT
    pred[20:25, 20:25, 3:9] = 2  # Far from GT
    pred[30:35, 30:35, 3:9] = 3  # Even farther

    return gt, pred

def run_test_scenario(gt, pred, scenario_name):
    """Run a test scenario and return results"""
    print(f"\n{scenario_name}")
    print(f"GT unique values: {np.unique(gt)}")
    print(f"Pred unique values: {np.unique(pred)}")

    try:
        # Create components
        matcher = RegionBasedMatching(cca_backend=CCABackend.scipy)
        approximator = ConnectedComponentsInstanceApproximator()
        semantic_pair = SemanticPair(prediction_arr=pred, reference_arr=gt)

        # Run evaluation
        result = panoptic_evaluate(
            input_pair=semantic_pair,
            instance_approximator=approximator,
            instance_matcher=matcher,
            instance_metrics=[Metric.DSC, Metric.IOU],
            global_metrics=[Metric.DSC],
            verbose=False
        )

        print(f"✅ {scenario_name} successful!")
        print(f"  Pred instances: {result.num_pred_instances}, Ref instances: {result.num_ref_instances}")
        print(f"  TP: {result.tp}, FP: {result.fp}, FN: {result.fn}")

        # Check individual metrics if available
        if hasattr(result, 'list_metrics') and result.list_metrics:
            for metric, values in result.list_metrics.items():
                if values:  # Only print if there are values
                    avg_val = np.mean(values) if values else 0
                    print(f"  {metric}: avg={avg_val:.3f}, values={len(values)}")

        return True

    except Exception as e:
        print(f"❌ {scenario_name} failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all test scenarios"""
    print("🧪 Running comprehensive RegionBasedMatching tests...")

    scenarios = [
        (test_scenario_1_basic, "Test 1: Basic non-overlapping"),
        (test_scenario_2_overlapping, "Test 2: Overlapping predictions"),
        (test_scenario_3_empty_prediction, "Test 3: Empty predictions"),
        (test_scenario_4_extra_predictions, "Test 4: Extra predictions"),
    ]

    results = []
    for scenario_func, name in scenarios:
        gt, pred = scenario_func()
        success = run_test_scenario(gt, pred, name)
        results.append(success)

    # Summary
    passed = sum(results)
    total = len(results)
    print(f"\n📊 Test Summary: {passed}/{total} scenarios passed")

    if passed == total:
        print("🎉 All comprehensive tests passed!")
        return True
    else:
        print("💥 Some tests failed!")
        return False

if __name__ == "__main__":
    main()
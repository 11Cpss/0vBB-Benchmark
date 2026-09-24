"""Event identity and score-orientation checks for the native-input adapter."""
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

SCRIPT = Path(__file__).resolve().parents[1] / 'standardize.py'


class StandardizationTests(unittest.TestCase):
    def run_cli(self, root, data, metadata=None, extra=()):
        np.savez(root / 'native.npz', **data)
        command = [sys.executable, '-B', str(SCRIPT), '--input', str(root / 'native.npz'),
                   '--output', str(root / 'output.npz'), '--energy-unit', 'MeV', *extra]
        if metadata is not None:
            np.savez(root / 'physical.npz', **metadata)
            command += ['--energy-metadata', str(root / 'physical.npz')]
        return subprocess.run(command, text=True, capture_output=True)

    def test_event_join_units_exo_direction_and_no_range_filter(self):
        data = dict(event_id=np.array(['b', 'a', 'd', 'c']), label=np.array([1, 0, 1, 0]),
                    score=np.array([2., -1., 3., 0.]))
        metadata = dict(event_id=np.array(['c', 'd', 'a', 'b']), label=np.array([0, 1, 0, 1]),
                        energy=np.array([3., 3.001, .005, -.001]))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = self.run_cli(root, data, metadata,
                                  ['--energy-key', 'energy', '--positive-label', '0', '--score-label', '1'])
            self.assertEqual(result.returncode, 0, result.stderr)
            with np.load(root / 'output.npz') as output:
                np.testing.assert_array_equal(output['event_id'], data['event_id'])
                np.testing.assert_array_equal(output['label'], 1 - data['label'])
                np.testing.assert_array_equal(output['score'], -data['score'])
                np.testing.assert_allclose(output['energy_keV'], [-1., 5., 3001., 3000.])
                self.assertEqual(output['energy_keV'].dtype, np.dtype('float64'))

    def test_equal_length_different_event_ids_is_rejected(self):
        data = dict(event_id=np.array(['a', 'b']), label=np.array([0, 1]), score=np.array([0., 1.]))
        metadata = dict(event_id=np.array(['a', 'c']), energy_keV=np.array([.1, .2]))
        with tempfile.TemporaryDirectory() as directory:
            result = self.run_cli(Path(directory), data, metadata)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('does not cover', result.stderr)

    def test_disagreeing_labels_is_rejected(self):
        data = dict(event_id=np.array(['a', 'b']), label=np.array([0, 1]), score=np.array([0., 1.]))
        metadata = dict(event_id=np.array(['a', 'b']), label=np.array([1, 0]), energy_keV=np.array([.1, .2]))
        with tempfile.TemporaryDirectory() as directory:
            result = self.run_cli(Path(directory), data, metadata)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('labels disagree', result.stderr)

    def test_duplicate_ids_and_multi_output_scores_are_rejected(self):
        for ids, score in [(np.array(['a', 'a']), np.array([0., 1.])),
                           (np.array(['a', 'b']), np.ones((2, 4)))]:
            data = dict(event_id=ids, label=np.array([0, 1]), score=score, energy_keV=np.array([.1, .2]))
            with tempfile.TemporaryDirectory() as directory:
                self.assertNotEqual(self.run_cli(Path(directory), data).returncode, 0)


if __name__ == '__main__':
    unittest.main(verbosity=2)

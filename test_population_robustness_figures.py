"""Checks for the population statistics used by the replacement figures."""
import unittest
import numpy as np
import population_robustness_figures as f


class PopulationMetrics(unittest.TestCase):
    def test_prefix_partition(self):
        self.assertEqual(len(f.PREFIXES),63)
        self.assertEqual(int(f.OLD.sum()),51)
        self.assertEqual(int(f.NEW.sum()),12)
        self.assertEqual(len(set(f.PREFIXES)),63)

    def test_clean_reference_is_thirty_not_self_competition(self):
        canonical=np.arange(63*30).reshape(63,30)
        noisy=np.repeat(canonical[None],f.TRIALS,axis=0)
        own,rival=f.compare(noisy,canonical)
        np.testing.assert_array_equal(own,np.full((f.TRIALS,63),30))
        np.testing.assert_array_equal(rival,np.zeros((f.TRIALS,63)))

    def test_true_collision_is_a_full_strength_competitor(self):
        canonical=np.arange(63*30).reshape(63,30)
        canonical[1]=canonical[0]
        own,rival=f.compare(np.repeat(canonical[None],f.TRIALS,axis=0),canonical)
        np.testing.assert_array_equal(rival[:,:2],np.full((f.TRIALS,2),30))
        np.testing.assert_array_equal(rival[:,2:],np.zeros((f.TRIALS,61)))

    def test_partial_membership_is_counted_exactly(self):
        canonical=np.arange(63*30).reshape(63,30)
        noisy=np.repeat(canonical[None],f.TRIALS,axis=0)
        noisy[0,0,:6]=np.arange(1900,1906)
        own,rival=f.compare(noisy,canonical)
        self.assertEqual(own[0,0],24)
        self.assertEqual(rival[0,0],0)


if __name__=='__main__':unittest.main()

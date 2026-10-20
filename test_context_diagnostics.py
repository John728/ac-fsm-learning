"""Independent simulator checks for the cached-readout diagnostic."""
import unittest
import numpy as np
import context_diagnostics as diag
from pyac import Area, Connection, NetworkSpec, Network, Hebbian, Homeostasis


class ReadoutEquivalence(unittest.TestCase):
    def test_dense_replay_matches_sparse_pyac(self):
        original_n, original_k = diag.old.N, diag.old.K
        diag.old.N, diag.old.K = 40, 4
        try:
            spec = NetworkSpec(
                areas=tuple(Area(a,40,4,kind='input') for a in diag.AREAS)+(Area('word',40,4),),
                connections=tuple(Connection(a,'word',p=.6) for a in diag.AREAS),
                plasticity=Hebbian(.4),homeostasis=Homeostasis(interval=1))
            net=Network(spec,seed=12)
            initial=np.stack([net.weights[a,'word'].toarray() for a in diag.AREAS])
            caps=np.array([np.arange(i*4,(i+1)*4) for i in range(4)])
            r=diag.Readout(initial,caps,(0,1,2))
            rng=np.random.default_rng(19)
            samples=np.array([[np.sort(rng.choice(40,4,replace=False)) for a in diag.AREAS] for _ in range(20)])
            labels=rng.integers(0,4,20)
            for x,y in zip(samples,labels):
                for a,active in zip(diag.AREAS,x):net.set_activity(a,active)
                net.step(targets=('word',),forced_activations={'word':caps[y]})
                r.fit(x[None],np.array([y]))
            expected=np.concatenate([net.weights[a,'word'].toarray() for a in diag.AREAS])
            np.testing.assert_allclose(r.w,expected,rtol=1e-12,atol=1e-14)
            batched=diag.Readout(initial,caps,(0,1,2))
            batched.fit(samples,labels)
            np.testing.assert_allclose(batched.w,expected,rtol=1e-12,atol=1e-14)
            for x in samples:
                for a,active in zip(diag.AREAS,x):net.set_activity(a,active)
                net.step(targets=('word',),plasticity=False,normalize=False)
                actual=np.array([np.isin(cap,net.assembly('word').indices).sum() for cap in caps])
                np.testing.assert_array_equal(r.predict(x[None],4)[0][0],actual)
        finally:diag.old.N,diag.old.K=original_n,original_k

    def test_ties_are_reported_and_not_counted_as_strict_success(self):
        m=diag.metrics(np.array([[1,1],[0,2]]),np.array([0,1]))
        self.assertEqual(m['accuracy'],1)
        self.assertEqual(m['strict_accuracy'],.5)
        self.assertEqual(m['ties'],1)


if __name__=='__main__':unittest.main()

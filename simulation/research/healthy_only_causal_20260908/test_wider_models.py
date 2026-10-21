import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).parent))
import wider_models as m

def test_all_variants_finite_deterministic():
    c={"a":np.ones((4,22)),"b":np.full((3,22),2.)}
    for v in m.VARIANTS:
        s=m.fit(c,v); t=m.fit(c,v)
        assert all(np.array_equal(a,b) for a,b in zip(s["global"],t["global"]))
        assert np.all(np.isfinite(m.score(s,"a",c["a"])))

def test_center_gradient_finite_difference():
    x=np.ones((3,22)); a=m._init(); c=m._center(x,a); _,g=m._loss_grads(x,a,c)
    e=1e-6; p=[z.copy() for z in a]; q=[z.copy() for z in a]; p[0].flat[2]+=e; q[0].flat[2]-=e
    fd=(m._loss_grads(x,tuple(p),c)[0]-m._loss_grads(x,tuple(q),c)[0])/(2*e)
    assert np.isclose(g[0].flat[2],fd,rtol=2e-4,atol=2e-5)

def test_all_parameter_array_gradients_with_center():
    x=np.linspace(-.4,.6,44).reshape(2,22); a=m._init(); c=m._center(x,a); _,g=m._loss_grads(x,a,c)
    for layer, index in ((0,3),(1,1),(2,7),(3,5)):
        e=1e-6; p=[z.copy() for z in a]; q=[z.copy() for z in a]; p[layer].flat[index]+=e; q[layer].flat[index]-=e
        fd=(m._loss_grads(x,tuple(p),c)[0]-m._loss_grads(x,tuple(q),c)[0])/(2*e)
        assert np.isclose(g[layer].flat[index],fd,rtol=3e-4,atol=3e-5)

def test_empty_rows_rejected():
    try: m.fit({"empty":np.empty((0,22))},"global_h12")
    except ValueError: return
    assert False

def test_personalized_mapping_and_payload():
    s=m.fit({"a":np.ones((4,22)),"b":np.full((4,22),2.)},"personalized_h12")
    assert not np.array_equal(s["private"]["a"][2],s["private"]["b"][2])
    assert m.communication(s)["float32_parameters"]==276
    assert m.communication(s)["fragments_per_update"]==6

def test_wider_parameter_count_and_payload():
    s=m.fit({"a":np.ones((2,22))},"global_h12")
    assert sum(z.size for z in s["global"])==562
    assert m.communication(s)["fragments_per_update"]==12


def test_hidden4_kernel_exact_legacy_and_prefix():
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
    import thesis_participating_node_scaling as old
    import json
    from types import SimpleNamespace
    protocol=json.loads((Path(__file__).resolve().parents[2]/'thesis_70_device_federated_protocol.json').read_text())
    rng=np.random.default_rng(6)
    clients={str(i):rng.normal(size=(5+i,22)) for i in range(3)}
    expected=old.fit_federated({k:SimpleNamespace(train=x) for k,x in clients.items()},tuple(clients),'fedadam',protocol)
    a=m._init(hidden=4); mom=tuple(np.zeros_like(v) for v in a); var=tuple(np.zeros_like(v) for v in a)
    for t in range(1,13):
        loc=[]
        for x in clients.values():
            b=a
            for _ in range(2): b=m._step(b,x)
            loc.append(b)
        a,mom,var=m._adam(a,loc,mom,var,t)
    for x,y in zip(a,expected.arrays()): np.testing.assert_array_equal(x,y)
    s=m.fit(clients,'personalized_h12')
    for mid in clients:
        np.testing.assert_array_equal(s['private'][mid][0],s['global'][0])
    x=rng.normal(size=(18,22))
    np.testing.assert_allclose(m.score(s,'0',x)[:5],m.score(s,'0',x[:5]))

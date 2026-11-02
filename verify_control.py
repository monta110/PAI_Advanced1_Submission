"""Validate causal control, exact baseline equivalence, trace metrics and replay."""
import argparse,csv,hashlib,json
import numpy as np
from upgrade import *
from robustness import episode as old_episode

def unit_tests():
    weights=json.loads((BASE/'results/model.json').read_text())
    ref=Controller('original',weights);slow=Controller('low_gain',weights)
    assert slow.act(.41,0)[0]==ref.act(.41,0)[0]
    action,raw,_,scale,phase=slow.act(.399,0)
    assert scale==.2 and phase==1 and abs(action-raw*.2)<1e-12
    # Once near contact, backing up does not accidentally re-enable fast motion.
    assert slow.act(.41,8)[3]==.2
    filt=Controller('filtered_low_gain',weights)
    assert filt.act(.399,4)[2]==1
    assert filt.act(.399,4)[2]==1.75
    # Sensor information is the only force input; no world parameters enter act().
    a=Controller('no_force',weights);b=Controller('no_force',weights)
    for z in [.43,.4,.38,.37]:assert a.act(z,0)[0]==b.act(z,999)[0]
    c=Condition(8000,.05,6)
    _,old,_=old_episode(.25,'with_force',weights,27000,c)
    _,new,_=episode(.25,'original',weights,27000,c)
    assert np.array_equal(old,new[:240,:9]),'Original first 8 seconds changed'
    # A brief terminal in-band segment does not count as settled.
    arr=np.zeros((720,len(COLUMNS)));arr[:,0]=(np.arange(720)+1)/30;arr[:,2]=arr[:,3]=8
    assert score(arr)['success'] and score(arr)['settling_s']==1/30
    arr[-10:,2]=10
    assert score(arr)['settling_s'] is None
    print('PASS: approach phase, gain scaling, phase latch, causal EMA, force masking, original 8s equivalence, unsettled censoring')

def audit(out=OUT):
    if any(not (out/f'traces_k{k}.npz').exists() for k in STIFFNESSES):
        raise SystemExit('Raw traces are absent: extract the experiment-data ZIP beside the submission folder, or run upgrade.py first.')
    manifest=json.loads((out/'protocol.json').read_text())
    assert hashlib.sha256((BASE/'results/model.json').read_bytes()).hexdigest()==manifest['model_sha256']
    for p,h in manifest['source_sha256'].items():assert hashlib.sha256((BASE/p).read_bytes()).hexdigest()==h
    rows=list(csv.DictReader((out/'trials.csv').open()))
    summaries=json.loads((out/'summary.json').read_text())
    n=manifest['n_per_cell'];assert len(rows)==27*4*n
    heights=np.random.default_rng(HEIGHT_SEED).uniform(.235,.265,n)
    for stiffness in STIFFNESSES:
        traces=np.load(out/f'traces_k{stiffness}.npz')
        for r in [r for r in rows if int(r['stiffness'])==stiffness]:
            assert float(r['height_m'])==heights[int(r['trial'])]
            a=traces[f'{r["condition"]}__{r["mode"]}__{r["trial"]}']
            assert a.shape==(720,len(COLUMNS)) and np.isfinite(a).all()
            m=score(a)
            for k,v in m.items():
                if isinstance(v,bool):assert (r[k]=='True')==v
                elif v is None:assert r[k]==''
                else:assert np.isclose(float(r[k]),v,atol=1e-10)
            d=int(r['delay_steps']);expected=np.r_[np.zeros(d),a[:720-d,7]] if d else a[:,7]
            assert np.array_equal(a[:,6],expected)
            assert np.all(abs(a[:,5])<=ACTION_LIMIT)
        for i in range(n):
            assert np.array_equal(traces[f'k{stiffness}_n0.05_d0__no_force__{i}'][:,1:6],traces[f'k{stiffness}_n1.5_d6__no_force__{i}'][:,1:6])
    for s in summaries:
        subset=[r for r in rows if r['condition']==s['condition'] and r['mode']==s['mode']]
        assert len(subset)==n
        for field,trial in [('successes','success'),('successes_8s','success_8s'),('settled_count','settled')]:assert s[field]==sum(r[trial]=='True' for r in subset)
        assert np.isclose(s['mean_mae_n'],np.mean([float(r['mae_n']) for r in subset]))
    w=json.loads((BASE/'results/model.json').read_text())
    _,a,_=episode(float(heights[0]),'low_gain',w,NOISE_SEED,Condition(8000,1.5,6))
    assert np.allclose(a,np.load(out/'traces_k8000.npz')['k8000_n1.5_d6__low_gain__0'],atol=1e-8)
    print(f'PASS: frozen sources/model, {len(rows)} traces, 8s/24s metrics, exact delay, full mask invariance, {len(summaries)} summaries, stress replay')

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--unit-only',action='store_true');ap.add_argument('--out',type=Path,default=OUT);a=ap.parse_args()
    unit_tests()
    if not a.unit_only:audit(a.out)

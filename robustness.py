"""Frozen-policy robustness benchmark: stiffness x force noise x sensor delay.

All 27 conditions share paired heights and Gaussian draws. Scores use true
physics forces, not delayed/noisy sensor readings. No feedback retuning.
"""
from collections import deque
from dataclasses import dataclass, asdict
from pathlib import Path
import argparse, csv, hashlib, itertools, json, time, platform
import numpy as np
from experiment import World, features, BASE, TARGET, GAIN, ACTION_LIMIT, STEPS

OUT=BASE/'results_robustness'
MODES=('no_force','with_force','teacher')
STIFFNESSES=(500,2000,8000)    # assigned to BOTH interacting bodies, not effective contact k
NOISE_LEVELS=(.05,.5,1.5)    # N, Gaussian std before nonnegative clipping
DELAYS=(0,3,6)              # extra 30 Hz control samples = 0, 100, 200 ms
HEIGHT_SEED=90301
NOISE_SEED=15000
COLUMNS=('time_s','ee_z_before_m','true_force_n','substep_peak_n','raw_action_m',
         'applied_action_m','observed_force_n','fresh_sensor_n','command_z_after_m')

@dataclass(frozen=True)
class Condition:
    stiffness: int=2000
    noise_std: float=.05
    delay_steps: int=0
    @property
    def key(self):
        return f'k{self.stiffness}_n{self.noise_std:g}_d{self.delay_steps}'

class ForceSensor:
    """Delay noisy samples by exactly d extra ticks; unavailable history is zero."""
    def __init__(self, noise_std, delay_steps, seed):
        if noise_std<0 or delay_steps<0:raise ValueError('Noise and delay must be nonnegative')
        self.noise_std=noise_std
        self.rng=np.random.default_rng(seed)
        self.queue=deque([0.]*delay_steps)
    def read(self,true_force):
        fresh=max(0.,true_force+self.rng.normal(0,self.noise_std))
        self.queue.append(fresh)
        return self.queue.popleft(),fresh

def metrics(trace):
    """Settling: in-band forever after t AND at least 1 second observed thereafter.
    Non-settled episodes are censored (None), never counted as fast successes.
    """
    force=trace[:,2];tail=force[-60:]
    in_band=abs(force-TARGET)<=1
    suffix=np.logical_and.accumulate(in_band[::-1])[::-1]
    eligible=np.flatnonzero(suffix & (np.arange(len(force))<=len(force)-30))
    settling=float(trace[eligible[0],0]) if len(eligible) else None
    contacts=np.flatnonzero(force>=.5)
    contact=float(trace[contacts[0],0]) if len(contacts) else None
    peak=float(trace[:,3].max())
    return {'mae_n':float(np.mean(abs(tail-TARGET))),
            'rmse_n':float(np.sqrt(np.mean((tail-TARGET)**2))),
            'peak_n':peak,'overshoot_n':max(0.,peak-TARGET),
            'in_band_fraction':float(in_band[-60:].mean()),
            'success':bool(in_band[-60:].mean()>=.9 and peak<=16),
            'settled':settling is not None,'settling_s':settling,
            'first_contact_s':contact,
            'settling_after_contact_s':settling-contact if settling is not None and contact is not None else None}

def episode(height,mode,weights,seed,condition,capture=False):
    if mode not in MODES:raise ValueError(mode)
    w=World(height,stiffness=condition.stiffness)
    sensor=ForceSensor(condition.noise_std,condition.delay_steps,seed)
    force=0.;trace=[];frames=[]
    try:
        for i in range(STEPS):
            observed,fresh=sensor.read(force)
            z=w.observe_z()
            if mode=='teacher':action=GAIN*(observed-TARGET)
            else:action=float(features(z,observed,mode=='with_force')@weights[mode])/1000
            applied=float(np.clip(action,-ACTION_LIMIT,ACTION_LIMIT))
            force,peak=w.move(applied)
            trace.append([(i+1)/30,z,force,peak,action,applied,observed,fresh,w.z])
            if capture and i%2==0:frames.append(w.frame())
    finally:w.close()
    a=np.asarray(trace)
    return metrics(a),a,frames

def summarize(records):
    groups=[]
    for key in dict.fromkeys(r['condition'] for r in records):
        for mode in MODES:
            rows=[r for r in records if r['condition']==key and r['mode']==mode]
            settled=[r['settling_s'] for r in rows if r['settled']]
            groups.append({'condition':key,'stiffness':rows[0]['stiffness'],
                'noise_std':rows[0]['noise_std'],'delay_steps':rows[0]['delay_steps'],
                'mode':mode,'n':len(rows),'successes':sum(r['success'] for r in rows),
                'mean_mae_n':float(np.mean([r['mae_n'] for r in rows])),
                'mean_overshoot_n':float(np.mean([r['overshoot_n'] for r in rows])),
                'max_peak_n':max(r['peak_n'] for r in rows),
                'settled_count':len(settled),
                'mean_settling_s_settled_only':float(np.mean(settled)) if settled else None})
    return groups

def write_csv(path,rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)

def evaluate(n=30,out=OUT):
    out.mkdir(parents=True,exist_ok=True)
    model_bytes=(BASE/'results/model.json').read_bytes();weights=json.loads(model_bytes)
    conditions=[Condition(*x) for x in itertools.product(STIFFNESSES,NOISE_LEVELS,DELAYS)]
    heights=np.random.default_rng(HEIGHT_SEED).uniform(.235,.265,n)
    manifest={'protocol':'frozen-policy robustness v1','trial_count_per_cell':n,
        'conditions':[{'key':c.key,**asdict(c)} for c in conditions],
        'modes':MODES,'height_seed':HEIGHT_SEED,'noise_seed_base':NOISE_SEED,
        'heights_m':heights.tolist(),'control_hz':30,'physics_hz':240,'steps':STEPS,
        'target_n':TARGET,'damping_per_body':30,'columns':COLUMNS,
        'model_sha256':hashlib.sha256(model_bytes).hexdigest(),
        'code_sha256':{p:hashlib.sha256((BASE/p).read_bytes()).hexdigest() for p in ['robustness.py','experiment.py']},
        'python':platform.python_version(),'platform':platform.system(),
        'noise_note':'Gaussian then clipped at zero, then delayed; same Gaussian draws across conditions/modes.',
        'delay_note':'Extra ticks beyond ordinary one-cycle feedback; initialized to zero.',
        'stiffness_note':'Both pad and tool use the listed contactStiffness; not a material modulus or calibrated effective stiffness.',
        'settling_note':'Earliest sample staying within 8+-1 N to end with at least 30 samples (1 s) remaining; absent is null, excluded from conditional mean.',
        'success_note':'Last 60 samples at least 90% within 8+-1 N, and all-episode substep peak <=16 N.'}
    # Record protocol before running any benchmark episodes.
    (out/'protocol.json').write_text(json.dumps(manifest,indent=2))
    rows=[];traces={};started=time.monotonic()
    for c in conditions:
        for i,height in enumerate(heights):
            for mode in MODES:
                stats,trace,_=episode(float(height),mode,weights,NOISE_SEED+i,c)
                rows.append({'condition':c.key,**asdict(c),'trial':i,'height_m':float(height),'mode':mode,**stats})
                traces[f'{c.key}__{mode}__{i}']=trace
        print(f'{c.key}: completed {len(rows)} / {len(conditions)*n*3} episodes',flush=True)
    write_csv(out/'trials.csv',rows)
    groups=summarize(rows);write_csv(out/'summary.csv',groups)
    (out/'summary.json').write_text(json.dumps(groups,indent=2,allow_nan=False))
    np.savez_compressed(out/'traces.npz',**traces)
    # Per-condition paired differences (no multiple-comparison inference).
    improvements=[]
    for c in conditions:
        a=[r['mae_n'] for r in rows if r['condition']==c.key and r['mode']=='no_force']
        b=[r['mae_n'] for r in rows if r['condition']==c.key and r['mode']=='with_force']
        diff=np.array(a)-b
        ci=np.quantile(np.random.default_rng(78).choice(diff,(5000,n),replace=True).mean(axis=1),[.025,.975])
        improvements.append({'condition':c.key,'mean_mae_improvement_n':float(diff.mean()),'bootstrap_95_ci':ci.tolist()})
    (out/'paired_differences.json').write_text(json.dumps(improvements,indent=2))
    print(f'Finished in {time.monotonic()-started:.1f} seconds',flush=True)
    return groups

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--trials',type=int,default=30);ap.add_argument('--video',action='store_true');ap.add_argument('--out',type=Path,default=OUT);a=ap.parse_args()
    if a.trials<2:ap.error('--trials must be >= 2')
    evaluate(a.trials,a.out)
    if a.video:
        from make_robustness_video import build_video
        build_video(a.out)

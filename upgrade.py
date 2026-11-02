"""Compare frozen learned force control with conservative post-processing.

Controllers receive ONLY end-effector z and the causal sensor stream.
True surface height, contact stiffness and true contact force are not policy inputs.
"""
from pathlib import Path
import argparse,hashlib,itertools,json,time,platform
import numpy as np
from experiment import World,features,BASE,ACTION_LIMIT
from robustness import Condition,ForceSensor,metrics,write_csv,STIFFNESSES,NOISE_LEVELS,DELAYS

OUT=BASE/'results_control'
MODES=('no_force','original','low_gain','filtered_low_gain')
DURATION_S=24
STEPS=30*DURATION_S
SLOW_GAIN=.2
FILTER_ALPHA=.25
APPROACH_Z=.4
HEIGHT_SEED=90401
NOISE_SEED=25000
COLUMNS=('time_s','ee_z_before_m','true_force_n','substep_peak_n','raw_action_m',
         'applied_action_m','observed_force_n','fresh_sensor_n','command_z_after_m',
         'policy_force_input_n','gain_scale','slow_phase')

class Controller:
    def __init__(self,mode,weights):
        if mode not in MODES:raise ValueError(mode)
        self.mode=mode;self.weights=weights;self.filtered=0.;self.slow=False
    def act(self,z,observed):
        self.filtered=FILTER_ALPHA*observed+(1-FILTER_ALPHA)*self.filtered
        if z<=APPROACH_Z:self.slow=True
        force_input=observed;scale=1.
        if self.mode in ('low_gain','filtered_low_gain') and self.slow:
            scale=SLOW_GAIN
            if self.mode=='filtered_low_gain':force_input=self.filtered
        model='no_force' if self.mode=='no_force' else 'with_force'
        raw=float(features(z,force_input,model=='with_force')@self.weights[model])/1000
        action=float(np.clip(raw*scale,-ACTION_LIMIT,ACTION_LIMIT))
        return action,raw,force_input,scale,int(self.slow)

def score(trace):
    full=metrics(trace)
    early=metrics(trace[:240])
    full.update({'success_8s':early['success'],'mae_8s_n':early['mae_n'],
                 'settled_8s':early['settled'],'settling_8s_s':early['settling_s']})
    return full

def episode(height,mode,weights,seed,condition,capture=False):
    world=World(height,stiffness=condition.stiffness)
    sensor=ForceSensor(condition.noise_std,condition.delay_steps,seed)
    ctrl=Controller(mode,weights)
    force=0.;trace=[];frames=[]
    try:
        for i in range(STEPS):
            observed,fresh=sensor.read(force)
            z=world.observe_z()
            action,raw,used,scale,phase=ctrl.act(z,observed)
            force,peak=world.move(action)
            trace.append([(i+1)/30,z,force,peak,raw,action,observed,fresh,world.z,used,scale,phase])
            if capture and i%2==0:frames.append(world.frame())
    finally:world.close()
    a=np.array(trace)
    return score(a),a,frames

def summarize(records):
    groups=[]
    for key in dict.fromkeys(r['condition'] for r in records):
        for mode in MODES:
            rs=[r for r in records if r['condition']==key and r['mode']==mode]
            settled=[r['settling_s'] for r in rs if r['settled']]
            groups.append({'condition':key,'stiffness':rs[0]['stiffness'],
                'noise_std':rs[0]['noise_std'],'delay_steps':rs[0]['delay_steps'],
                'mode':mode,'n':len(rs),'successes':sum(r['success'] for r in rs),
                'successes_8s':sum(r['success_8s'] for r in rs),
                'mean_mae_n':float(np.mean([r['mae_n'] for r in rs])),
                'mean_mae_8s_n':float(np.mean([r['mae_8s_n'] for r in rs])),
                'mean_overshoot_n':float(np.mean([r['overshoot_n'] for r in rs])),
                'max_peak_n':max(r['peak_n'] for r in rs),
                'settled_count':len(settled),
                'mean_settling_s_settled_only':float(np.mean(settled)) if settled else None})
    return groups

def evaluate(n=30,out=OUT):
    out.mkdir(parents=True,exist_ok=True)
    model_bytes=(BASE/'results/model.json').read_bytes();weights=json.loads(model_bytes)
    heights=np.random.default_rng(HEIGHT_SEED).uniform(.235,.265,n)
    conditions=[Condition(*x) for x in itertools.product(STIFFNESSES,NOISE_LEVELS,DELAYS)]
    manifest={'study':'conservative output layer on frozen learned model',
        'n_per_cell':n,'modes':MODES,'height_seed':HEIGHT_SEED,'noise_seed_base':NOISE_SEED,
        'heights_m':heights.tolist(),'duration_s':DURATION_S,'control_hz':30,'physics_hz':240,
        'slow_gain':SLOW_GAIN,'filter_alpha':FILTER_ALPHA,'approach_z_m':APPROACH_Z,
        'conditions':[{'key':c.key,'stiffness':c.stiffness,'noise_std':c.noise_std,'delay_steps':c.delay_steps} for c in conditions],
        'columns':COLUMNS,'model_sha256':hashlib.sha256(model_bytes).hexdigest(),
        'source_sha256':{p:hashlib.sha256((BASE/p).read_bytes()).hexdigest() for p in ['experiment.py','robustness.py','upgrade.py']},
        'policy_inputs':['end-effector z','delayed noisy force'],
        'prior_note':'z=0.4 m switching plane uses a known upper bound of the workspace surface, not the true height of each trial. Must be recalibrated for another setup.',
        'primary_score':'At 24 s: last 2 s >=90% within 8+-1 N AND all-episode substep peak <=16 N.',
        'secondary_score':'Same score at 8 s using the first 240 samples, for all four controllers.',
        'settling_note':'Earliest time staying in band until the end, with at least 1 s of samples remaining. Missing means not settled.',
        'development_note':'Fixed development heights .238/.25/.262 and separate sensor seeds; final heights unseen during design.',
        'python':platform.python_version()}
    (out/'protocol.json').write_text(json.dumps(manifest,indent=2))
    records=[];start=time.monotonic()
    # Store one trace archive per stiffness, keeping each file below GitHub's 100MB limit.
    for k in STIFFNESSES:
        traces={}
        for c in [c for c in conditions if c.stiffness==k]:
            for i,h in enumerate(heights):
                for mode in MODES:
                    stat,a,_=episode(float(h),mode,weights,NOISE_SEED+i,c)
                    records.append({'condition':c.key,'stiffness':k,'noise_std':c.noise_std,
                        'delay_steps':c.delay_steps,'trial':i,'height_m':float(h),'mode':mode,**stat})
                    traces[f'{c.key}__{mode}__{i}']=a
            print(f'{c.key}: {len(records)}/{27*n*4} episodes',flush=True)
        np.savez_compressed(out/f'traces_k{k}.npz',**traces)
    write_csv(out/'trials.csv',records)
    summary=summarize(records);write_csv(out/'summary.csv',summary)
    (out/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
    (out/'runtime.json').write_text(json.dumps({'evaluation_seconds':time.monotonic()-start},indent=2))
    print(f'Finished in {time.monotonic()-start:.1f} seconds',flush=True)
    return summary

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--trials',type=int,default=30);ap.add_argument('--out',type=Path,default=OUT);a=ap.parse_args()
    if a.trials<2:ap.error('--trials must be >=2')
    evaluate(a.trials,a.out)

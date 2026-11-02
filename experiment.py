"""Force-input ablation on a physically simulated KUKA arm.
No API keys, GPU, ROS, or external datasets are required.
"""
from pathlib import Path
import argparse, csv, json
import numpy as np
import pybullet as p
import pybullet_data

DT=1/240
SUBSTEPS=8
STEPS=240                    # 8 seconds per episode at 30 Hz
TARGET=8.0
START_Z=.43
ZMIN,ZMAX=.29,.45
GAIN=.00012
ACTION_LIMIT=.0008
BASE=Path(__file__).resolve().parent

class World:
    def __init__(self, height, gui=False, stiffness=2000, damping=30):
        self.cid=p.connect(p.GUI if gui else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setTimeStep(DT)
        p.setPhysicsEngineParameter(numSolverIterations=100)
        p.setGravity(0,0,-9.81)
        p.loadURDF('plane.urdf')
        col=p.createCollisionShape(p.GEOM_BOX,halfExtents=[.17,.18,height/2])
        vis=p.createVisualShape(p.GEOM_BOX,halfExtents=[.17,.18,height/2],rgbaColor=[.2,.55,.65,1])
        self.pad=p.createMultiBody(0,col,vis,[.55,0,height/2])
        self.arm=p.loadURDF('kuka_iiwa/model.urdf',useFixedBase=True)
        self.orn=p.getQuaternionFromEuler([0,np.pi,0])
        self.z=START_Z
        for j,v in enumerate([0,.4,0,-1.5,0,1.2,0]):p.resetJointState(self.arm,j,v)
        q=p.calculateInverseKinematics(self.arm,6,[.55,0,self.z],self.orn,maxNumIterations=200,residualThreshold=1e-6)
        for j in range(7):p.resetJointState(self.arm,j,q[j])
        ee=p.getLinkState(self.arm,6,computeForwardKinematics=True)
        pos,_=p.multiplyTransforms(ee[4],ee[5],[0,0,.08],[0,0,0,1])
        c=p.createCollisionShape(p.GEOM_SPHERE,radius=.025)
        v=p.createVisualShape(p.GEOM_SPHERE,radius=.025,rgbaColor=[1,.55,.12,1])
        self.tool=p.createMultiBody(.15,c,v,pos)
        p.createConstraint(self.arm,6,self.tool,-1,p.JOINT_FIXED,[0,0,0],[0,0,.08],[0,0,0])
        for j in range(-1,7):p.setCollisionFilterPair(self.arm,self.tool,j,-1,0)
        for b in (self.pad,self.tool):
            p.changeDynamics(b,-1,contactStiffness=stiffness,contactDamping=damping,lateralFriction=.5)
        for _ in range(30):self.move(0)
    def force(self):
        return sum(c[9] for c in p.getContactPoints(bodyA=self.tool,bodyB=self.pad))
    def observe_z(self):
        return p.getLinkState(self.arm,6,computeForwardKinematics=True)[4][2]
    def move(self, action):
        self.z=float(np.clip(self.z+action,ZMIN,ZMAX))
        q=p.calculateInverseKinematics(self.arm,6,[.55,0,self.z],self.orn,maxNumIterations=80,residualThreshold=1e-5)
        p.setJointMotorControlArray(self.arm,list(range(7)),p.POSITION_CONTROL,targetPositions=q[:7],forces=[200]*7,positionGains=[.3]*7,velocityGains=[1]*7)
        fs=[]
        for _ in range(SUBSTEPS):
            p.stepSimulation(); fs.append(self.force())
        return float(np.mean(fs)),float(max(fs))
    def frame(self):
        view=p.computeViewMatrixFromYawPitchRoll([.3,0,.40],1.5,55,-24,0,2)
        proj=p.computeProjectionMatrixFOV(43,640/400,.02,4)
        rgba=p.getCameraImage(640,400,view,proj,renderer=p.ER_TINY_RENDERER,shadow=1)[2]
        return np.asarray(rgba,dtype=np.uint8).reshape(400,640,4)[:,:,:3]
    def close(self):p.disconnect(self.cid)

def features(z, force, enabled):
    # Identical model size; the force channel is masked before training/inference.
    u=(z-.36)/.04
    return np.array([1.,u,u*u,u*u*u,force/10 if enabled else 0.])

def train(out):
    rng=np.random.default_rng(2101)
    rows=[]
    for height in rng.uniform(.235,.265,24):
        w=World(float(height)); f=0.
        for t in range(STEPS):
            z=w.observe_z()
            measured=max(0.,f+rng.normal(0,.05))
            label=GAIN*(measured-TARGET)  # unsaturated teacher action, in metres
            rows.append((z,measured,label))
            # Sweep simulated positions to cover contact/non-contact (not a real-robot routine).
            # Expert labels are queried on these exploratory states.
            f,_=w.move(-ACTION_LIMIT if t<120 else ACTION_LIMIT)
        w.close()
    data=np.array(rows)
    np.savez_compressed(out/'training_data.npz',z=data[:,0],force=data[:,1],expert_action=data[:,2])
    weights={}
    for name,enabled in [('no_force',False),('with_force',True)]:
        X=np.array([features(z,f,enabled) for z,f,_ in rows])
        # Ridge supervised imitation, same examples/labels and regularization.
        reg=np.diag([0.,1e-6,1e-6,1e-6,1e-6])
        weights[name]=np.linalg.solve(X.T@X+reg,X.T@(data[:,2]*1000)).tolist()
    (out/'model.json').write_text(json.dumps(weights,indent=2))
    return weights

def episode(height, mode, weights, seed, capture=False):
    rng=np.random.default_rng(seed)
    w=World(height); f=0.; trace=[]; frames=[]
    for t in range(STEPS):
        measured=max(0.,f+rng.normal(0,.05))
        z=w.observe_z()
        if mode=='teacher':action=GAIN*(measured-TARGET)
        else:action=float(features(z,measured,mode=='with_force')@weights[mode])/1000
        f,peak=w.move(float(np.clip(action,-ACTION_LIMIT,ACTION_LIMIT)))
        trace.append([t/30,z,f,peak,action])
        if capture and t%2==0:frames.append(w.frame())
    w.close()
    a=np.array(trace); tail=a[-60:,2]
    stats={'mae_n':float(np.mean(abs(tail-TARGET))),
           'rmse_n':float(np.sqrt(np.mean((tail-TARGET)**2))),
           'peak_n':float(a[:,3].max()),
           'in_band_fraction':float(np.mean(abs(tail-TARGET)<=1)),
           'success':bool(np.mean(abs(tail-TARGET)<=1)>=.9 and a[:,3].max()<=16)}
    return stats,a,frames

def evaluate(out,weights,n=30):
    rng=np.random.default_rng(90211)
    heights=rng.uniform(.235,.265,n)
    records=[]; traces={}
    for i,h in enumerate(heights):
        for mode in ['no_force','with_force','teacher']:
            stats,trace,_=episode(float(h),mode,weights,5000+i)
            records.append({'trial':i,'height_m':float(h),'mode':mode,**stats})
            traces[f'{mode}_{i}']=trace
        if (i+1)%10==0:print(f'Evaluated {i+1}/{n} paired trials',flush=True)
    with (out/'trials.csv').open('w',newline='') as fp:
        writer=csv.DictWriter(fp,fieldnames=records[0].keys());writer.writeheader();writer.writerows(records)
    np.savez_compressed(out/'traces.npz',**traces)
    summary={}
    for mode in ['no_force','with_force','teacher']:
        r=[x for x in records if x['mode']==mode]
        summary[mode]={'n':n,'successes':sum(x['success'] for x in r),'mean_mae_n':float(np.mean([x['mae_n'] for x in r])), 'max_peak_n':max(x['peak_n'] for x in r)}
    diffs=np.array([records[i*3]['mae_n']-records[i*3+1]['mae_n'] for i in range(n)])
    boots=np.random.default_rng(78).choice(diffs,(5000,n),replace=True).mean(axis=1)
    summary['paired_mae_improvement_n']={'mean':float(diffs.mean()),'bootstrap_95_ci':np.quantile(boots,[.025,.975]).tolist()}
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2),flush=True)
    return summary

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--trials',type=int,default=30);ap.add_argument('--video',action='store_true');args=ap.parse_args()
    if args.trials<2:ap.error('--trials must be >=2')
    out=BASE/'results';out.mkdir(exist_ok=True)
    weights=train(out);evaluate(out,weights,args.trials)
    if args.video:
        from make_video import build_video
        build_video(out,weights)
if __name__=='__main__':main()

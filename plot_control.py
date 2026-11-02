"""Figures from the measured 8s and 24s success rates."""
import os,tempfile
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR',str(Path(tempfile.gettempdir())/'force_touch_plot_cache'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import json
from upgrade import OUT,STIFFNESSES,NOISE_LEVELS,DELAYS

LABELS={'no_force':'No force input','original':'Original force policy','low_gain':'Conservative output (x0.2)','filtered_low_gain':'Conservative + smoothing'}

def make_plots(out=OUT):
    rows=json.loads((out/'summary.json').read_text())
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    for seconds,field in [(8,'successes_8s'),(24,'successes')]:
        fig,axes=plt.subplots(2,3,figsize=(13,7.2),layout='constrained')
        for r,mode in enumerate(['original','low_gain']):
            for col,k in enumerate(STIFFNESSES):
                ax=axes[r,col]
                vals=np.array([[next(x[field]/x['n']*100 for x in rows if x['mode']==mode and x['stiffness']==k and x['noise_std']==noise and x['delay_steps']==d) for d in DELAYS] for noise in NOISE_LEVELS])
                ax.imshow(vals,vmin=0,vmax=100,cmap='YlGnBu',aspect='auto')
                for (iy,ix),val in np.ndenumerate(vals):ax.text(ix,iy,f'{val:.0f}%',ha='center',va='center',color='white' if val>=60 else '#193346',fontsize=16)
                ax.set_xticks(range(3),[0,100,200]);ax.set_yticks(range(3),NOISE_LEVELS)
                ax.set_xlabel('Extra force-sensor delay (ms)');ax.set_ylabel('Noise std (N)')
                ax.set_title(f'{"Original" if mode=="original" else "Output x0.2"} | k = {k:,} N/m')
        fig.suptitle(f'Success at {seconds} s: {rows[0]["n"]} paired heights in each cell',fontsize=17)
        fig.savefig(out/f'success_{seconds}s.png',dpi=160);plt.close(fig)
    totals=[]
    for mode in LABELS:
        rs=[x for x in rows if x['mode']==mode]
        totals.append({'mode':mode,'n':sum(x['n'] for x in rs),
            'successes_8s':sum(x['successes_8s'] for x in rs),
            'successes_24s':sum(x['successes'] for x in rs),
            'mean_mae_n':float(np.mean([x['mean_mae_n'] for x in rs])),
            'max_peak_n':max(x['max_peak_n'] for x in rs)})
    (out/'totals.json').write_text(json.dumps(totals,indent=2))
    print('Saved 8-second and 24-second success matrices.')
    return totals

if __name__=='__main__':make_plots()

"""116-second evidence video: failure, fix, and the cost in completion time."""
from pathlib import Path
from functools import lru_cache
import json,os,time,argparse
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import imageio.v2 as imageio
from upgrade import BASE,OUT,Condition,episode
from plot_control import make_plots
W,H,FPS=1280,720,15
BG='#101c2c';WHITE='#edf4fa';MUTED='#b1c5d8';RED='#ffad82';GREEN='#69dfc1'
DEMOS=[('硬い面／測定値に200 msの遅れ',Condition(8000,.05,6),.25),('柔らかい面／追加の遅れなし',Condition(500,.05,0),.238)]

@lru_cache(maxsize=64)
def font(size):
    for f in [os.getenv('DEMO_FONT',''),'/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc','/System/Library/Fonts/Supplemental/Arial Unicode.ttf','/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc']:
        if f and Path(f).exists():return ImageFont.truetype(f,size)
    raise RuntimeError('Set DEMO_FONT to a Japanese font file.')

def txt(d,xy,s,size=30,color=WHITE):
    while d.textbbox((0,0),s,font=font(size))[2]>W-xy[0]-35 and size>16:size-=1
    d.text(xy,s,font=font(size),fill=color)

def canvas():return Image.new('RGB',(W,H),BG)
def card(kicker,title,lines):
    im=canvas();d=ImageDraw.Draw(im);txt(d,(60,45),kicker,23,GREEN);txt(d,(60,110),title,42)
    for i,line in enumerate(lines):txt(d,(60,215+65*i),line,29,MUTED)
    txt(d,(60,674),'出典：本課題のPyBulletシミュレーション実験（2026-11-02）',18,MUTED)
    return im

def intro_card():
    return card('01 / 目的','力の測定が遅れたときの振動を抑える',[
        '目標は、ロボットアームで面を8 Nの力で押し続けることです。',
        '硬い面では、測定値が200 ms遅れる条件で力が振動しました。',
        'AIモデルは固定し、出力を調整して振動が減るかを調べます。',
        'シミュレーション映像は等速で再生します（音声なし）。'])

def method_card(n):
    return card('02 / 方法','接触する前に、AIの出力を1/5に抑える',[
        '手先が高さ0.40 m以下になると、AIの出力を1/5にします。',
        '切り替える高さは既知の作業範囲から決め、各試行の面の高さは使いません。',
        '力覚なし・力覚あり・出力調整・出力調整＋平滑化の4方式を比べます。',
        f'27条件で各{n}回ずつ、4方式を合わせて{27*n*4:,}回試しました。',
        '全方式を24秒間動かし、8秒時点と24秒時点で評価します。'])

def trace_plot(d,a,b,i):
    x0,y0,x1,y1=80,565,1200,652
    cap=max(16,float(np.ceil(max(a[:,2].max(),b[:,2].max())/8)*8))
    d.rectangle((x0,y0,x1,y1),outline='#52677b')
    d.line((x0,y1-8/cap*(y1-y0),x1,y1-8/cap*(y1-y0)),fill='#bfcbd8',width=2)
    # The original 8-second deadline is explicitly visible.
    x8=x0+8/24*(x1-x0);d.line((x8,y0,x8,y1),fill='#8194a6',width=1)
    txt(d,(80,524),'実際の接触力：橙＝元の制御 ／ 緑＝出力調整後 ／ 水平線＝目標の8 N',19)
    for a_,color in [(a,RED),(b,GREEN)]:
        pts=[(x0+j/719*(x1-x0),y1-a_[j,2]/cap*(y1-y0)) for j in range(i+1)]
        if len(pts)>1:d.line(pts,fill=color,width=3)
    txt(d,(12,560),f'{cap:.0f} N',17,MUTED);txt(d,(32,635),'0',17,MUTED)
    txt(d,(80,666),'0 s',18,MUTED);txt(d,(int(x8)-10,666),'8 s',18,MUTED);txt(d,(1140,666),'24 s',18,MUTED)

def heatmap_card(out,n):
    im=canvas();d=ImageDraw.Draw(im);txt(d,(40,18),'03 / 24秒時点の成功率：上段は元の制御、下段は出力調整後',28)
    chart=Image.open(out/'success_24s.png').convert('RGB');chart.thumbnail((1190,585));im.paste(chart,((W-chart.width)//2,78))
    txt(d,(40,675),f'各条件{n}回。縦軸は測定ノイズ、横軸は追加の遅れ。3列で接触の硬さを比較。',21,MUTED)
    return im

def tradeoff_card(out,n):
    rows=json.loads((out/'summary.json').read_text())
    get=lambda key,mode:next(r for r in rows if r['condition']==key and r['mode']==mode)
    before=get('k8000_n0.05_d6','original');after=get('k8000_n0.05_d6','low_gain');soft=get('k500_n0.05_d0','low_gain')
    return card('04 / 安定性と作業時間','振動は減るが、目標の力に達するまで時間がかかる',[
        f'硬い面で200 msの遅れがあるとき、成功は{before["successes"]}回から{after["successes"]}回に増えました。',
        f'最大の接触力は{before["max_peak_n"]:.1f} Nから{after["max_peak_n"]:.1f} Nに下がりました（各{n}回、24秒間の実験）。',
        f'柔らかい面では、出力調整後の成功は8秒で{soft["successes_8s"]}回、24秒で{soft["successes"]}回でした。',
        '目標の力に達するまで時間がかかるため、作業時間も含めた評価が必要です。',
        '比較条件：測定ノイズσ＝0.05 N。柔らかい面では追加の遅れなし。'])

def limitations_card(out):
    totals=json.loads((out/'totals.json').read_text());a=next(t for t in totals if t['mode']=='original');b=next(t for t in totals if t['mode']=='low_gain')
    return card('05 / 結果と今後の課題',f'成功は増えたが、{b["n"]-b["successes_24s"]}回は基準を満たさなかった',[
        f'24秒時点の成功は、{a["n"]}回中{a["successes_24s"]}回から{b["successes_24s"]}回に増えました。',
        f'出力調整後も{b["n"]-b["successes_24s"]}回は失敗し、最大{b["max_peak_n"]:.1f} Nの力がかかりました。',
        'AIモデルは同じで、出力の調整方法は人が設計しています。',
        '切り替える高さは環境に応じた設定が必要です。実機の安全性は未検証です。',
        '画像は入力に使わず、FlowVLAの再現や実機での検証は行っていません。'])

def build_video(out=OUT,wait_for_results=False):
    def results_ready():
        return (out/'runtime.json').exists() and (out/'runtime.json').stat().st_mtime >= (out/'protocol.json').stat().st_mtime
    if not wait_for_results and not results_ready():
        raise RuntimeError('Run upgrade.py before creating the video.')
    weights=json.loads((BASE/'results/model.json').read_text())
    n=json.loads((out/'protocol.json').read_text())['n_per_cell']
    writer=imageio.get_writer(str(out/'demo_control.mp4'),fps=FPS,codec='libx264',quality=8,macro_block_size=16,ffmpeg_params=['-movflags','+faststart'])
    def hold(im,seconds):
        for _ in range(seconds*FPS):writer.append_data(np.asarray(im))
    try:
        hold(intro_card(),10)
        hold(method_card(n),14)
        demo=[]
        for k,(label,c,height) in enumerate(DEMOS):
            sa,a,fa=episode(height,'original',weights,31000+k,c,True)
            sb,b,fb=episode(height,'low_gain',weights,31000+k,c,True)
            demo.append({'condition':c.key,'height_m':height,'seed':31000+k,'original':sa,'low_gain':sb})
            for i,(ia,ib) in enumerate(zip(fa,fb)):
                im=canvas();im.paste(Image.fromarray(ia),(0,116));im.paste(Image.fromarray(ib),(640,116));d=ImageDraw.Draw(im);t=i*2
                txt(d,(25,8),f'{k+1}/2  {label}   面の高さ {height*1000:.0f} mm   {(t+1)/30:.1f} / 24 s   等速',26)
                txt(d,(25,49),f'元の制御   {a[t,2]:5.2f} N',28,RED)
                txt(d,(665,49),f'出力調整後   {b[t,2]:5.2f} N',28,GREEN)
                txt(d,(25,88),f'硬さk={c.stiffness} N/m   測定ノイズσ={c.noise_std:g} N',17,MUTED)
                txt(d,(665,88),'同じAIモデルの出力を1/5に調整',17,MUTED)
                trace_plot(d,a,b,t);writer.append_data(np.asarray(im))
                if i==300:im.save(out/f'demo_case_{k+1}.png')
        (out/'demo_cases.json').write_text(json.dumps(demo,indent=2))
        deadline=time.monotonic()+3600
        while not results_ready():
            if time.monotonic()>deadline:raise RuntimeError('Timed out waiting for the evaluation.')
            time.sleep(1)
        make_plots(out)
        hold(heatmap_card(out,n),15)
        hold(tradeoff_card(out,n),14)
        hold(limitations_card(out),15)
    finally:writer.close()
    print('Video complete: demo_control.mp4 (116 seconds)',flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--wait-for-results',action='store_true');a=ap.parse_args()
    build_video(wait_for_results=a.wait_for_results)

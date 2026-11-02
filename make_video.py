"""Render real simulator frames to a captioned 76-second MP4, 1x speed."""
from pathlib import Path
import json, os
from functools import lru_cache
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import imageio.v2 as imageio
from experiment import episode, BASE, TARGET

W,H,FPS=1280,720,15
BG='#101c2c'; WHITE='#edf4fa'; MUTED='#b1c5d8'; RED='#ffad82'; GREEN='#69dfc1'

@lru_cache(maxsize=64)
def font(size):
    candidates=[os.getenv('DEMO_FONT',''),'/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc','/System/Library/Fonts/Supplemental/Arial Unicode.ttf','/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc']
    for f in candidates:
        if f and Path(f).exists():return ImageFont.truetype(f,size)
    raise RuntimeError('Japanese font not found. Set DEMO_FONT to a Japanese .ttf/.ttc/.otf font path.')

def txt(d,xy,s,size=30,color=WHITE):
    while d.textbbox((0,0),s,font=font(size))[2]>W-xy[0]-40 and size>16:size-=1
    d.text(xy,s,font=font(size),fill=color)
def canvas():return Image.new('RGB',(W,H),BG)
def card(kicker,title,lines):
    im=canvas();d=ImageDraw.Draw(im)
    txt(d,(70,55),kicker,24,GREEN);txt(d,(70,120),title,44)
    for i,line in enumerate(lines):txt(d,(70,235+i*65),line,30,MUTED)
    txt(d,(70,665),'PHYSICAL AI  /  PyBullet + supervised imitation  /  Simulation',20,MUTED)
    return im

def plot_trace(d,a,b,upto):
    x0,y0,x1,y1=80,570,1200,650
    d.rectangle((x0,y0,x1,y1),outline='#52677b')
    yt=y1-TARGET/24*(y1-y0)
    d.line((x0,yt,x1,yt),fill=GREEN,width=2)
    txt(d,(80,532),'接触力：橙 = 力覚なし ／ 緑 = 力覚あり ／ 水平線 = 8 N',20)
    for tr,c in [(a,RED),(b,GREEN)]:
        pts=[(x0+i/239*(x1-x0),y1-min(24,max(0,tr[i,2]))/24*(y1-y0)) for i in range(upto+1)]
        if len(pts)>1:d.line(pts,fill=c,width=3)
    txt(d,(80,665),'0 s',18,MUTED);txt(d,(1150,665),'8 s',18,MUTED)

def build_video(out,weights):
    summary=json.loads((out/'summary.json').read_text())
    writer=imageio.get_writer(str(out/'demo.mp4'),fps=FPS,codec='libx264',quality=8,macro_block_size=16,ffmpeg_params=['-movflags','+faststart'])
    def hold(im,seconds):
        a=np.asarray(im)
        for _ in range(seconds*FPS):writer.append_data(a)
    hold(card('01  /  QUESTION','力覚があると、押し付けの精度は変わるか？',[
        'KUKAアームで、面を8 Nの力で押す。',
        '面の高さは毎回変わる。正確な高さはモデルに渡さない。',
        '同じ学習法で、力覚入力の有無だけを比較する。',
        '実機・VLAではなく、接触制御の最小実験。']),10)
    hold(card('02  /  METHOD','小さなAIモデルを、教師の動作から学習',[
        '24条件 × 240ステップの教師データ → 線形回帰。',
        '入力：手先の高さ ＋ 接触力（なし条件は0でマスク）。',
        '出力：手先を上下に動かす量。両条件で制限は共通。',
        '未学習の30条件で評価。以下は別途固定した3条件。',
        '映像はシミュレーション時間に対して等速・音声なし。']),15)
    for k,height in enumerate([.238,.250,.262]):
        _,a,fa=episode(height,'no_force',weights,8000+k,True)
        _,b,fb=episode(height,'with_force',weights,8000+k,True)
        for i,(ia,ib) in enumerate(zip(fa,fb)):
            im=canvas();im.paste(Image.fromarray(ia),(0,116));im.paste(Image.fromarray(ib),(640,116));d=ImageDraw.Draw(im)
            txt(d,(35,20),f'03  /  DEMO {k+1}/3    面高さ {height*1000:.0f} mm    等速',28)
            txt(d,(35,70),'力覚なし',30,RED);txt(d,(675,70),'力覚あり',30,GREEN)
            t=i*2
            txt(d,(225,74),f'{a[t,2]:5.2f} N',30,RED);txt(d,(865,74),f'{b[t,2]:5.2f} N',30,GREEN)
            plot_trace(d,a,b,t);writer.append_data(np.asarray(im))
            if i==90:im.save(out/f'demo_case_{k+1}.png')
    nf=summary['no_force'];wf=summary['with_force'];ci=summary['paired_mae_improvement_n']['bootstrap_95_ci']
    hold(card('04  /  RESULTS',f'未学習の{nf["n"]}条件：同じ面高さ・同じノイズで比較',[
        f'力覚なし：平均絶対誤差 {nf["mean_mae_n"]:.2f} N ／ 成功 {nf["successes"]}/{nf["n"]}',
        f'力覚あり：平均絶対誤差 {wf["mean_mae_n"]:.2f} N ／ 成功 {wf["successes"]}/{wf["n"]}',
        '誤差は最後の2秒間で評価。目標は8 N。',
        '成功：評価区間の90%以上が±1 N内、かつ最大16 N以下。',
        f'誤差改善の95%区間：{ci[0]:.2f}〜{ci[1]:.2f} N（対応あり再標本化）']),15)
    hold(card('05  /  TAKEAWAY','この実験で言えること・言えないこと',[
        'この高さのばらつきでは、力覚が接触力の調整に役立つ。',
        'AIは単純な力制御の教師を模倣した線形回帰。',
        'AIの優位性、視覚との比較、実機への転用は未検証。',
        'FlowVLA / ForceVLA の再現ではない。',
        '次の一歩：ノイズ・硬さを変える／画像入力を加える。']),12)
    writer.close()
    print('Video written: results/demo.mp4 (76 seconds)',flush=True)

if __name__=='__main__':
    out=BASE/'results';build_video(out,json.loads((out/'model.json').read_text()))

"""Create a four-page report from the frozen 8s/24s controller comparison."""
import json
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,PageBreak,Image
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.colors import HexColor
from upgrade import BASE,OUT
from plot_control import make_plots
FONT='HeiseiKakuGo-W5';pdfmetrics.registerFont(UnicodeCIDFont(FONT))
S=ParagraphStyle('body',fontName=FONT,fontSize=10,leading=16,spaceAfter=9,wordWrap='CJK')
H1=ParagraphStyle('title',parent=S,fontSize=20,leading=28,textColor=HexColor('#123b54'),spaceAfter=15)
H2=ParagraphStyle('heading',parent=S,fontSize=12.5,leading=19,textColor=HexColor('#167875'),spaceBefore=7,spaceAfter=7)
SM=ParagraphStyle('small',parent=S,fontSize=8,leading=12)
CELL=ParagraphStyle('cell',parent=SM,spaceAfter=0)

def table(rows,widths):
    t=Table([[Paragraph(str(v),CELL) for v in row] for row in rows],colWidths=widths,hAlign='LEFT')
    t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),HexColor('#dbecea')),('ROWBACKGROUNDS',(0,1),(-1,-1),[HexColor('#f1f6f7'),HexColor('#ffffff')]),('VALIGN',(0,0),(-1,-1),'MIDDLE'),('TOPPADDING',(0,0),(-1,-1),7),('BOTTOMPADDING',(0,0),(-1,-1),7)]))
    return t

def build(out=OUT):
    totals=make_plots(out);rows=json.loads((out/'summary.json').read_text());p=json.loads((out/'protocol.json').read_text());n=p['n_per_cell']
    get=lambda key,mode:next(r for r in rows if r['condition']==key and r['mode']==mode)
    hard0=get('k8000_n0.05_d6','original');hard1=get('k8000_n0.05_d6','low_gain');soft=get('k500_n0.05_d0','low_gain')
    story=[]
    def para(t,style=S):story.append(Paragraph(t,style))
    def heading(t):para(t,H2)
    para('力覚制御の頑健性を高める<br/>学習モデルの出力調整と作業時間の比較',H1)
    para('Physical AI 応用1 最終課題 ｜ PyBullet + 教師あり回帰 ｜ 生成AI支援で実装',SM)
    heading('1. 目的と背景')
    para('ロボットアームで面を目標8 Nの力で押す。前段の実験では、力覚が有効な一方、硬い接触と測定遅延の組み合わせで振動した。今回は、学習モデルを固定したまま出力を慎重にする対策を追加し、誤差・押し過ぎ・成功率・所要時間を比較した。精度は接触力の誤差を意味する。')
    heading('2. AIの役割と追加した制御')
    para('AIは、手先高さzの1〜3次、接触力F、切片の5特徴から、上下移動量を出力するリッジ回帰。24種類の面高さで上下に探索した5,760状態に、教師 dz=0.00012×(F−8) m/step のラベルを付けて学習した。動作は±0.8 mm/step以内。モデルは基礎版と同じ係数を使用した。')
    story.append(table([['方式','内容'],['力覚なし','力特徴を学習・推論時に0にする。比較の基準。'],['元の力覚制御','力覚を入力する学習モデルをそのまま使う。'],['AI出力1/5','手先zが0.40 m以下になったら、飽和制限をかける前の学習モデル出力を0.2倍する。'],['1/5＋平滑化','同じ減速に加え、力入力を指数平滑化する。係数α=0.25。']], [112,393]))
    story.append(Spacer(1,10))
    para('切替位置より上では元の出力を使う。一度切り替えたら減速状態を維持する。切替位置は既知の面高さ上限から決めた作業範囲の事前知識であり、別環境では設定し直す必要がある。制御へ渡すのは手先zと受信した力だけで、各試行の真の面高さ・硬さ・真の接触力は入力しない。')
    heading('3. 条件と評価時間')
    para('PyBullet / KUKA iiwa、物理240 Hz・制御30 Hz。工具・対象の接触剛性を各500 / 2,000 / 8,000 N/m、ノイズσを0.05 / 0.5 / 1.5 N、追加遅延を0 / 100 / 200 msとし、全27組み合わせを評価した。面高さは235〜265 mm。剛性は接触計算の設定値で、実物の材料硬度ではない。')
    para(f'各条件で同じ{n}通りの高さ・乱数を使い、4方式で計{27*n*4:,}試行。全方式を24秒間動かし、元の期限に対応する8秒時点と、24秒時点の両方を採点した。モデルの再学習はせず、出力調整は手設計である。',SM)
    story.append(PageBreak())
    para('結果：改善と時間の代償',H1)
    heading('4. 指標と集計')
    para('成功：各評価時点の直前2秒の90%以上が8±1 N内、かつ開始からその時点までの物理ステップ最大力が16 N以下。MAEは直前2秒の平均絶対誤差。全指標はノイズ・遅延を加える前の真の力から計算する。16 Nは実験用の便宜的なしきい値である。')
    para('安定時刻は、8±1 Nに入って終了まで維持し、最低1秒分の観測が残る最初の時刻。未安定は欠測とし、安定した試行だけの平均と安定数を併記する。24秒時点の成績だけで、8秒以内の作業性能を主張しない。')
    names={'no_force':'力覚なし','original':'元の力覚制御','low_gain':'AI出力1/5','filtered_low_gain':'1/5＋平滑化'}
    data=[['方式','8秒で成功','24秒で成功','24秒MAE<br/>平均 N','全試行の<br/>最大力 N']]
    for x in totals:data.append([names[x['mode']],f'{x["successes_8s"]}/{x["n"]}',f'{x["successes_24s"]}/{x["n"]}',f'{x["mean_mae_n"]:.3f}',f'{x["max_peak_n"]:.2f}'])
    story.append(table(data,[125,90,95,95,100]));story.append(Spacer(1,9))
    low=next(t for t in totals if t['mode']=='low_gain')
    para(f'対策後も{low["n"]-low["successes_24s"]}試行は未成功で、全条件の最大力は{low["max_peak_n"]:.2f} Nだった。過渡的な大きな力が残る条件もあり、安全保証ではない。上表は27条件を等しく重み付けした記述的集計。同じ高さを繰り返し使うため、独立試行の単純な統計検定には使っていない。',SM)
    heading('5. 硬い接触＋200 ms遅延（σ=0.05 N）')
    para(f'元の力覚制御は24秒で成功{hard0["successes"]}/{n}、MAE {hard0["mean_mae_n"]:.3f} N。AI出力1/5では成功{hard1["successes"]}/{n}、MAE {hard1["mean_mae_n"]:.3f} Nだった。全試行最大力は{hard0["max_peak_n"]:.2f} Nから{hard1["max_peak_n"]:.2f} Nへ変わった。遅れた測定値で大きく動かし続けることを抑える対策として機能した。')
    heading('6. 柔らかい接触での時間の代償（遅延0、σ=0.05 N）')
    settle=soft['mean_settling_s_settled_only']
    para(f'AI出力1/5では、8秒時点の成功は{soft["successes_8s"]}/{n}、24秒時点では{soft["successes"]}/{n}。安定数は{soft["settled_count"]}/{n}で、安定した試行の平均安定時刻は{settle:.2f}秒だった。慎重に動くことで押し過ぎを抑える一方、目標力に達するまで待つ必要がある。')
    para('平滑化条件は、ノイズ対策を加える効果を切り分けるための参考条件である。平滑化は応答の遅れも増やすため、どの条件でも上乗せの改善が得られるとは限らない。学習モデル自体が高度化した、あるいは従来制御より優れたことを示す結果ではない。')
    story.append(PageBreak())
    para('同じ条件を、8秒と24秒で評価',H1)
    para('各図の上段は元の力覚制御、下段はAI出力1/5。左から接触剛性500 / 2,000 / 8,000 N/m。横軸は追加遅延、縦軸はノイズσ。色と数値は成功率。')
    story.append(Image(str(out/'success_8s.png'),width=505,height=279.69));story.append(Spacer(1,13))
    story.append(Image(str(out/'success_24s.png'),width=505,height=279.69));story.append(Spacer(1,8))
    para('同じモデルと同じ乱数条件で比較した。時間を延ばした結果を元の8秒期限での改善と混同しないため、両方の図を掲載する。全4方式の数値はsummary.csvに保存した。',SM)
    story.append(PageBreak())
    para('限界・再現性・提出物',H1)
    heading('7. この実験で言えること')
    para('固定した学習モデルに慎重な出力調整を加えることで、接触の硬さ・ノイズ・遅延に対する挙動を変えられる。特に遅延による振動の抑制を示した。一方、作業時間が増えるため、用途に合った期限と精度を一緒に評価する必要がある。')
    heading('8. 限界')
    para('一方向の接触と特定のシミュレータ設定のみ。切替位置0.40 mは既知の作業範囲に依存し、任意の面へそのまま適用できない。物理モデルの剛性は実材料の再現値ではない。画像・言語・VLA・実機転移・挿入作業は扱わず、FlowVLA / ForceVLAの再現でもない。')
    para('モデルは単純な教師の模倣で、追加対策は手設計。AIの優位性や実機の安全性は主張しない。性能観察のため高い力でも途中停止しない構成なので、実機の手順として使用すべきではない。学習seedは1つで、モデル間のばらつきは未検証。')
    heading('9. 再現と検証')
    para('Python 3.12、CPUで実行。requirements.txtを導入し、upgrade.py、verify_control.py、make_control_video.py、make_control_report.pyを実行する。run.shで一括実行可能。提出用一式には学習済みモデル、学習データ、全試行CSV、集計、MP4、本PDFを収録。大きな剛性別の全時系列NPZは別の実験記録ZIPに分け、再実行でも生成できる。')
    para('評価高さseed=90401、ノイズseed=25000+試行番号。予備試験の高さ・seedとは分離した。protocol.jsonに評価前の条件・コード・モデルのSHA-256を保存。検証は、切替と0.2倍出力、指数平滑化、力覚マスク、元の8秒軌跡との一致、全試行の8秒/24秒採点、遅延の周期数、ストレス条件再実行を含む。')
    para('動画は116秒、等速・日本語字幕・音声なし。硬い接触＋遅延と柔らかい接触の2場面を各24秒、元の力覚制御とAI出力1/5で左右比較する。条件・高さ・seedは固定し、成功例の探索はしていない。',SM)
    heading('10. 出典・制作上の開示')
    para('物理エンジン・ロボットモデル：<link href="https://github.com/bulletphysics/bullet3" color="#167875">Bullet / PyBullet / pybullet_data</link>。接触反力API：<link href="https://github.com/bulletphysics/bullet3/blob/master/examples/pybullet/examples/contactFriction.py" color="#167875">公式 contactFriction.py</link>。提出要件は提供された「最終課題（10/19 改訂版）」PDFに基づく。',SM)
    para('実装・実験・資料作成に生成AIの支援を利用した。本課題用の比較実験であり、結果は実行したシミュレーションから生成した。外部モデル・ライブラリは各配布元ライセンスに従う。',SM)
    def footer(c,doc):
        c.setFont(FONT,8);c.setFillColor(HexColor('#61798b'));c.drawString(42,25,'Physical AI  |  Conservative control and completion-time trade-off');c.drawRightString(553,25,str(doc.page))
    f=out/'説明資料_PAI最終課題_アカウントID.pdf'
    SimpleDocTemplate(str(f),pagesize=(595.28,841.89),rightMargin=42,leftMargin=42,topMargin=34,bottomMargin=42,title='力覚制御の出力調整と作業時間の比較',author='').build(story,onFirstPage=footer,onLaterPages=footer)
    print(f.name)

if __name__=='__main__':build()

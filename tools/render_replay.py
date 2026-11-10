"""Export an offline replay and shareable scientific plots from fresh reference trajectories."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
ROOT=Path(__file__).resolve().parents[1]
summary=json.loads((ROOT/'verification/summary.json').read_text());payload=[]
fig,axes=plt.subplots(2,2,figsize=(12,7),constrained_layout=True)
for ax,r in zip(axes.flat,summary['cases']):
    folder=ROOT/'verification'/r['case'];data=np.load(folder/'reference_trajectories.npz')
    x=data['reference'];target=data['targets'];net=data['actual_PINN']
    for xy in x:ax.add_patch(Circle(xy,.0125,facecolor='#168a9c',edgecolor='#075466',alpha=.8,lw=.5))
    ax.scatter(target[:,0],target[:,1],marker='+',s=12,c='#a02020',linewidths=.6,label='Target')
    ax.scatter(net[:,0],net[:,1],marker='.',s=6,c='#fff4ad',label='Replayed frozen PINN')
    ax.axhline(0,color='#102f40',lw=2);ax.axhline(.3,color='#102f40',lw=2)
    ax.set(xlim=(1.50,2.12),ylim=(-.01,.31),aspect='equal',xlabel='x / L0',ylabel='y / L0',
           title=f"{r['particles']} particles · {'six fixed sites' if 'six_fixed' in r['case'] else 'independent wall ports'}\nreference minimum separation = {r['reference_minimum_separation']:.6f}")
    ax.legend(fontsize=6,loc='upper left')
    xy=data['positions'];positions=np.where(np.isfinite(xy),xy,None).tolist()
    payload.append(dict(case=r['case'],particles=r['particles'],metrics=r,time=data['time'].tolist(),positions=positions,targets=target.tolist()))
fig.suptitle('Independent release replay: physical particle disks at the common target time',fontsize=13)
for ext in ['png','pdf']:fig.savefig(ROOT/'docs'/('duck_comparison.'+ext),dpi=200)
plt.close(fig)
template=r'''<!doctype html><html lang="fa" dir="rtl"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>بازپخش مستقل رهاسازی ذرات</title><style>
body{background:#0c1d29;color:#e9f5f7;font:16px sans-serif;max-width:1100px;margin:30px auto;padding:20px;line-height:1.9}button,select{padding:8px;background:#183b4b;color:white;border:1px solid #52838b;border-radius:5px}canvas{width:100%;background:#eef7f9;border-radius:8px}input{width:50%}.metrics{color:#85e8d0}a{color:#8fe0ff}.note{color:#a9c5cf;font-size:14px}</style>
<h1>رهاسازی ذرات و تصویر اردک</h1><p>بازپخشِ مسیرهای تازهٔ حل مستقل DOP853، با همان شرایط اولیه و ساعت رهاسازی ذخیره‌شده. نمایش دیسک‌ها با قطر واقعی مدل؛ شکل پُرِ مصنوعی نمایش داده نمی‌شود.</p>
<select id="cases"></select> <button id="play">پخش رهاسازی</button> <button id="final">لحظهٔ هدف</button> <label><input id="zoom" type="checkbox" style="width:auto"> بزرگ‌نمایی ناحیهٔ هدف</label>
<canvas id="view" width="1100" height="330"></canvas><div><input id="time" type="range" min="0" max="500" value="500"> <span id="clock"></span></div><p id="metrics" class="metrics"></p>
<p class="note">دایرهٔ آبی: موقعیت حل مستقل. علامت قرمز: هدف. قبل از زمان رهاسازی، ذره نمایش داده نمی‌شود. طول و زمان بی‌بعدند. قبول عددی به معنی اعتبارسنجی تجربی یا حفظ شکل پس از لحظهٔ هدف نیست. فایل کاملاً آفلاین است.</p><a href="AUDIT_FA.md">گزارش بررسی</a>
<script>const DATA=__DATA__;let active=0,frame=500,playing=false,last=0;const $=id=>document.getElementById(id),c=$('view'),ctx=c.getContext('2d');
DATA.forEach((d,i)=>{let o=document.createElement('option');o.value=i;o.textContent=d.particles+' ذره — '+(d.case.includes('six_fixed')?'شش ایستگاه ثابت':'پورت‌های مستقل');$('cases').appendChild(o)});
function draw(){let d=DATA[active],zoom=$('zoom').checked,xmin=zoom?1.45:0,xmax=zoom?2.15:2.2,scale=Math.min(1000/(xmax-xmin),250/.30),left=(1100-(xmax-xmin)*scale)/2,top=(330-.30*scale)/2;
function map(x,y){return[left+(x-xmin)*scale,330-top-y*scale]}ctx.fillStyle='#eef7f9';ctx.fillRect(0,0,1100,330);ctx.strokeStyle='#254655';ctx.lineWidth=2;for(let y of [0,.30]){let a=map(xmin,y),b=map(xmax,y);ctx.beginPath();ctx.moveTo(...a);ctx.lineTo(...b);ctx.stroke()}
for(let q of d.targets){let [x,y]=map(...q);ctx.strokeStyle='#a93a3a';ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(x-3,y);ctx.lineTo(x+3,y);ctx.moveTo(x,y-3);ctx.lineTo(x,y+3);ctx.stroke()}
for(let p of d.positions){let q=p[frame];if(q[0]===null)continue;let a=map(...q);ctx.fillStyle='#1593a1';ctx.strokeStyle='#075365';ctx.beginPath();ctx.arc(...a,.0125*scale,0,2*Math.PI);ctx.fill();ctx.stroke()}
$('clock').textContent='t = '+d.time[frame].toFixed(4)+' / '+d.time.at(-1).toFixed(4);let m=d.metrics;$('metrics').textContent='آزمون‌ها: '+(m.accepted?'قبول':'رد')+' | بیشترین خطای مرجع: '+m.max_reference_target_error.toExponential(3)+' | کمترین فاصلهٔ مرجع: '+m.reference_minimum_separation.toFixed(6)+' | حداقل مجاز: '+m.required_separation.toFixed(3);}
$('cases').onchange=()=>{active=+$('cases').value;frame=500;playing=false;$('time').value=frame;draw()};$('time').oninput=()=>{frame=+$('time').value;playing=false;draw()};$('zoom').onchange=draw;$('final').onclick=()=>{frame=500;playing=false;$('time').value=frame;draw()};$('play').onclick=()=>{if(frame>=500)frame=0;playing=!playing};function tick(t){if(playing&&t-last>25){last=t;frame++;if(frame>=500){frame=500;playing=false}$('time').value=frame;draw()}requestAnimationFrame(tick)}draw();requestAnimationFrame(tick);
</script></html>'''
(ROOT/'docs/replay.html').write_text(template.replace('__DATA__',json.dumps(payload,separators=(',',':'),allow_nan=False)))
print('Exported plots and standalone HTML replay for',len(payload),'independently computed cases')

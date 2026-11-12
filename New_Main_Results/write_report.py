#!/usr/bin/env python3
"""Write a short Persian results inventory from saved measurements."""
import csv,json
from pathlib import Path
from collections import Counter

ROOT=Path(__file__).resolve().parent;RUN=ROOT/'runs/formation_20261112_cpu'
def read(p):return json.loads(p.read_text())
def number(x):return '—' if x is None else f'{x:.6g}'


def main():
    records=[]
    for file in sorted((RUN/'cases').glob('*/metrics.json')):
        records.append((read(file.parent/'config.json'),read(file),file.parent))
    inventory=read(RUN/'inventory.json')
    lines=['# گزارش کوتاه نتایج محاسباتی جدید','',
           'این گزارش موجودی و ارزیابی عددی است و بخشی از متن پروپوزال نیست.',
           'مدل و وزن ذخیره‌شده ثابت مانده‌اند؛ آموزش تازه انجام نشده است.',
           'پذیرش به اتمام هر دو بازپخش، خطای PINN حداکثر 0.003، خطای مرجع حداکثر 0.004، اختلاف دو خروجی حداکثر 0.004 و فاصلهٔ مرکزها حداقل 0.026 وابسته است.','',
           '## آرایش‌های ۱۰۰ ذره‌ای','',
           'جدول زیر غربالگری با seed=11 و بودجهٔ مشترک 24/50/35 است؛ مقایسهٔ نهایی سه-seed برای این چگالی هنوز ادعا نمی‌شود.','',
           '| ماسک | روش | پرتابگر در هر دیواره | وضعیت | بیشینهٔ خطای مرجع | کمینهٔ فاصلهٔ مرجع |','| --- | --- | ---: | --- | ---: | ---: |']
    for c,m,p in records:
        if c['group']!='D1' or c['N']!=100:continue
        ref=m.get('replays',{}).get('reference',{})
        lines.append(f"| {c['shape']} | {c['mode']} | {c['m'] if c['mode']=='fixed' else '—'} | {m['status']} | {number(ref.get('max_error'))} | {number(ref.get('minimum',{}).get('distance'))} |")
    lines+=['','هر حالت درگاه ثابت که در این جدول رد شده، شکست همین جست‌وجوی محدود است؛ ناممکن‌بودن فیزیکی آن نتیجه گرفته نمی‌شود.','',
            '## مقایسه‌های کامل و غربالگری مرکز','']
    for group in ['B1','F1','G2','C1_one_shot']:
        selected=[(c,m,p) for c,m,p in records if c['group']==group]
        if selected:lines.append(f"- {group}: {len(selected)} اجرای مستقل؛ {sum(m.get('accepted',False) for c,m,p in selected)} پذیرفته و {sum(not m.get('accepted',False) for c,m,p in selected)} نپذیرفته.")
    for collision in [False,True]:
        selected=[m for c,m,p in records if c['group']=='G2' and c['N']==6 and c['collision_aware']==collision]
        if selected:lines.append(f"- حذف/افزودن هزینهٔ فاصله، collision_aware={collision}: {sum(m['accepted'] for m in selected)}/{len(selected)} پذیرفته. همه با آزمون نهایی یکسان بررسی شده‌اند.")
    for group in ['E1','E1_free']:
        selected=[m for c,m,p in records if c['group']==group]
        lines.append(f"- {group}: {len(selected)} اجرای مرکز؛ {sum(m['accepted'] for m in selected)} پذیرفته. فقط seedهای دارای داده گزارش شده‌اند؛ خانه‌های خالی نتیجهٔ صفر نیستند.")
    lines+=['','ردشدن‌های بیضی مرکزی:','']
    for c,m,p in records:
        if c['group']=='F1' and not m['accepted']:lines.append(f"- q={c['q']}, θ={c['theta']}, seed={c['seed']}: {m['failure_type']} · [داده]({p.relative_to(ROOT)}/metrics.json)")
    lines+=['','جدول کامل مقایسه‌ها در [aggregate_comparisons.csv](runs/formation_20261112_cpu/tables/aggregate_comparisons.csv) است؛ مدت، سقف انتشار، هزینهٔ برخورد و چیدمان در ردیف‌های جدا نگه داشته شده‌اند.',
            'در چیدمان هم‌فاصله، افزایش تعداد پرتابگرها مختصات آن‌ها را نیز عوض می‌کند؛ از این نتایج، یکنواختی امکان‌پذیری بر حسب تعداد نتیجه گرفته نمی‌شود. سری تودرتو وضعیت مستقل خود را در فهرست اجراها دارد.','',
            '## حساسیت رهاسازی','']
    robust=sorted((RUN/'robustness').glob('*/metrics.json'))
    if robust:
        chosen=read(RUN/'robustness/selection.json')
        lines+=['نماینده‌های کم‌حاشیه و پُرحاشیه از میان برنامه‌های پذیرفته‌شدهٔ تکمیل‌شده با N≤24، طبق قاعدهٔ ذخیره‌شده انتخاب و سپس ثابت شده‌اند. این آزمون، ادعای استحکام آرایش ۱۰۰ ذره‌ای نیست.','']
        for label in ['low_margin','high_margin']:lines.append(f"- {label}: `{chosen[label]}`")
        lines+=['','در هر سناریوی کامل‌شده ۶۴ draw با seed ثابت، زمان هدف اسمی ثابت و حل‌گر مرجع مستقل استفاده شده است. draw نامعتبر بریده یا اصلاح نشده است؛ کسر پذیرش شرطی و غیرشرطی جدا هستند.','',
                '| نماینده | سناریو | سطح | پذیرفته / 64 | در محدوده / 64 | کسر پذیرش بین drawهای معتبر |','| --- | --- | ---: | ---: | ---: | ---: |']
        for file in robust:
            m=read(file);lines.append(f"| {m['selection_label']} | {m['scenario']} | {m['level']} | {m['accepted']} | {m['within_bounds']} | {number(m['bounds_conditional_pass_fraction'])} |")
        lines+=['','این‌ها سناریوهای حساسیت عددی‌اند و اندازه‌گیری قابلیت اطمینان سخت‌افزار نیستند.']
    lines+=['','## بازپخش تاریخی و تقارن','',
            'چهار برنامهٔ آرشیوی اردک با هر دو حل‌گر مستقل بازپخش شدند. مثلث و S/C/T با کد و تنظیمات اصلی دوباره محاسبه شده‌اند؛ این‌ها بازاجرای تازه‌اند، زیرا بستهٔ خروجی تاریخی اصلی در دسترس نبود. حد مدت تاریخی حروف 8 حفظ شده است.']
    file=RUN/'reflection/comparison.json'
    if file.exists():
        m=read(file);lines+=['',f"بازتاب برنامهٔ ذخیره‌شدهٔ {m['N']} ذره‌ای: بیشینهٔ اختلاف مرجع {number(m['paired_reference_symmetry']['max_endpoint_reflection_residual'])} و اختلاف PINN برابر {number(m['frozen_PINN_asymmetry']['max_endpoint_reflection_residual'])} است. در این آزمایش بهینه‌سازی دوباره انجام نشده است."]
    lines+=['','## اعتبارسنجی و کار باقی‌مانده','',
            'آزمون شبکهٔ محاسباتی سریع با PyTorch، مشتق ژاکوبین تنک، هزینهٔ تخصیص برداری/اسکالر، نگاشت دیواره و ظرفیت انجام شده است؛ خروجی در `selfcheck.json` ثبت است.',
            'حداقل یک برنامهٔ پذیرفتهٔ جدید از CSV دوباره بارگذاری و با هر دو حل‌گر بررسی شده است؛ داده و شناسهٔ مبدأ در `reload_check/` موجود است.',
            'شکل‌های PDF برداری و PNG با وضوح 300 dpi همراه با ورودی‌ها و SHA256 هر شکل ذخیره شده‌اند. بازبینی بصریِ شکل‌های اصلی انجام شده است.',
            f"ماتریس برنامه‌ریزی‌شده {inventory['planned_matrix_rows']} ردیف و {inventory['distinct_planned_configurations']} تنظیم متمایز دارد. وضعیت ماتریس در زمان این گزارش: `{inventory['matrix_status_counts']}`.",
            'محیط ۵ CPU و بدون GPU است. همهٔ شناسه‌های باقی‌مانده در [pending_cases.csv](runs/formation_20261112_cpu/pending_cases.csv) و موارد حذف‌شده پیش از حل در [excluded_cases.csv](runs/formation_20261112_cpu/excluded_cases.csv) مشخص‌اند.',
            'بودجه‌ها یا معیارهای پذیرش برای پنهان‌کردن شکست تغییر نکرده‌اند. اجرای کامل ماتریس ادعا نمی‌شود.',
            'فقط پوشهٔ New_Main_Results عیناً بین دو مخزن کپی شده است؛ متن، منابع و PDF قبلی پروپوزال دست‌نخورده‌اند.']
    (ROOT/'EXPERIMENT_REPORT_FA.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':main()

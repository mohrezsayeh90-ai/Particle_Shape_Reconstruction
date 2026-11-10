"""Index available source, trained tensor files and outstanding Kaggle artifacts."""
from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parents[1]
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    models=[]
    for p in sorted((ROOT/'assets').rglob('*.pt')):
        models.append({'path':str(p.relative_to(ROOT)),'notebook':p.relative_to(ROOT/'assets').parts[0],
                       'sha256':digest(p),'size_bytes':p.stat().st_size,'origin':'embedded literal/package in source notebook'})
    (ROOT/'models').mkdir(exist_ok=True)
    (ROOT/'models/INDEX.json').write_text(json.dumps(models,ensure_ascii=False,indent=2))
    outputs={x['ref']:x for x in json.loads((ROOT/'kaggle_outputs/INDEX.json').read_text())}
    rows=[]
    for r in json.loads((ROOT/'catalog.json').read_text()):
        slug=r['ref'].split('/')[-1];folder=ROOT/'evidence'/slug
        nb=next(folder.glob('*.ipynb'));o=outputs.get(r['ref'],{})
        extraction=json.loads((ROOT/'assets'/slug/'extraction.json').read_text())
        trained=[x for x in models if x['notebook']==slug]
        row={'ref':r['ref'],'notebook':str(nb.relative_to(ROOT)),'notebook_sha256':digest(nb),
             'asset_index':f'assets/{slug}/extraction.json','checkpoint_files':[x['path'] for x in trained],
             'embedded_artifacts':len(extraction['artifacts']),'original_outputs_listed':o.get('listed_files',0),
             'original_outputs_downloaded':o.get('downloaded_files',0),'original_outputs_complete':o.get('complete',False)}
        if o.get('listing_error'):row['output_listing_error']=o['listing_error']
        if slug=='master-5-2d-magnetic-notpinn':row['checkpoint_note']='No neural checkpoint needed: direct numerical method.'
        elif slug=='master3-clean-revised-version':row['checkpoint_note']='Original best_model.pt is pending CDN download; the frozen flow tensor used to independently replay its two cached duck cases is present in verification/original_frozen_checkpoint.pt.'
        elif not trained:row['checkpoint_note']='No trained tensor checkpoint embedded in the available source. Full training code is archived; original weights are not claimed recovered.'
        rows.append(row)
    status={'related_notebooks':len(rows),'embedded_checkpoint_files':len(models),'unique_checkpoint_hashes':len({m['sha256'] for m in models}),
            'original_outputs_listed':sum(r['original_outputs_listed'] for r in rows),
            'original_outputs_downloaded':sum(r['original_outputs_downloaded'] for r in rows),
            'all_original_outputs_recovered':all(r['original_outputs_complete'] for r in rows),'runs':rows}
    (ROOT/'ARCHIVE_STATUS.json').write_text(json.dumps(status,ensure_ascii=False,indent=2))
    lines=['# وضعیت آرشیو\n','فایل‌ها و وزن‌های موجود از فایل‌های دریافت‌نشده جدا شده‌اند. صفر بودن تعداد checkpoint برای روش عددی Master5 طبیعی است؛ برای بقیهٔ موارد، توضیح هر ردیف را ببینید.\n',
           '| نوت‌بوک | کد | فایل وزن استخراج‌شده | فایل خروجی اصلی دریافت‌شده/فهرست‌شده |', '|---|---|---:|---:|']
    for r in rows:
        label=r['ref'].split('/')[-1];count=f"{r['original_outputs_downloaded']}/{r['original_outputs_listed']}"
        if 'output_listing_error' in r:count+='؛ API: '+str(r['output_listing_error']['http_status'])
        lines.append(f"| {label} | [notebook](../{r['notebook']}) | {len(r['checkpoint_files'])} | {count} |")
    lines+=['\n## جزئیات وزن‌ها\n','فهرست کامل فایل‌ها و SHA-256 آن‌ها در [models/INDEX.json](../models/INDEX.json) است. فایل‌های `.pt` مطابق مدل و تعریف شبکهٔ همان نوت‌بوک بارگذاری شوند؛ اجزای نسخه‌های متفاوت را جای هم قرار ندهید.\n']
    for r in rows:
        if r.get('checkpoint_note'):lines.append('- `'+r['ref'].split('/')[-1]+'`: '+r['checkpoint_note'])
    lines+=['\nفایل‌های خروجی اصلی فقط پس از دانلود موفق، کنترل تعداد بایت و ثبت SHA-256، «دریافت‌شده» محسوب می‌شوند. URLهای موقت و کلید API در manifest ذخیره نمی‌شوند. اجرای دوبارهٔ notebook خروجی بازتولیدشده می‌سازد؛ آن خروجی جای فایل اصلیِ دریافت‌نشده از Kaggle برچسب نمی‌خورد.']
    (ROOT/'docs/ARCHIVE_STATUS.md').write_text('\n'.join(lines))
    print({k:v for k,v in status.items() if k!='runs'})
if __name__=='__main__':main()

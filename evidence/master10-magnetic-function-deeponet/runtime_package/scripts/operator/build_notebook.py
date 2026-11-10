"""Build a standalone Kaggle notebook; no credentials or external data required."""
from pathlib import Path
import base64,io,zipfile,json,argparse

ROOT=Path(__file__).resolve().parents[2]

def build(destination,pretrained=False,steps=18000):
    buf=io.BytesIO()
    with zipfile.ZipFile(buf,'w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted((ROOT/'magnetic_operator').glob('*.py')):
            z.write(p,p.relative_to(ROOT))
        extra=list((ROOT/'scripts/operator').glob('*.py'))
        extra += [ROOT/'magnetic_operator/README_FA.md',ROOT/'requirements.txt',ROOT/'examples/operator_field.csv']
        extra += [ROOT/'results/operator'/name for name in ('master9_revalidation.json','gpu_training_validation_v2.json','physical_scope_estimates.json','provenance.json')]
        for p in extra:
            if p.exists():z.write(p,p.relative_to(ROOT))
        if pretrained:
            for name in ('checkpoint.pt','training.json','progress.json'):
                z.write(ROOT/'results/operator'/name,'pretrained/'+name)
    blob=base64.b64encode(buf.getvalue()).decode()
    cells=[]
    def md(s):cells.append({'cell_type':'markdown','metadata':{},'source':s.splitlines(True)})
    def code(s):cells.append({'cell_type':'code','metadata':{},'source':s.splitlines(True),'outputs':[],'execution_count':None})
    md('''# Magnetic function-to-trajectory physics-informed DeepONet

دو تابع ورودی `a_left(t), a_right(t)` شدت نسبی میدان دو منبع آینه‌ای را تعیین می‌کنند.
`B=Bs*(a_left*b_left+a_right*b_right)` و `F=alpha/2*grad(|B|²)`؛ جملهٔ تداخل دو میدان حفظ می‌شود.
این نسخه تقریب بیش‌میرای همان ذرهٔ 500 نانومتری و جریان پوازی Master9 است، نه حل جریان/ماکسول جدید.

مدل یک DeepONet با branch/trunk و lifting فیزیکی است؛ انتگرال زمانی میدان در موقعیت ثابت دقیق محاسبه می‌شود و شبکه تصحیح ناشی از حرکت ذره را از باقیماندهٔ معادله می‌آموزد. هیچ مسیر حل‌شده‌ای وارد آموزش نمی‌شود.
ورودی‌ها تابعی‌اند (17 حسگر برای هر منبع در هر پنجره)، نه پنج ضریب از پیش انتخاب‌شده.
هر تابع تنها در دامنهٔ شدت، تفکیک زمانی و هندسهٔ مشخص قابل پوشش است. ادعای پاسخ دقیق برای همهٔ توابع ممکن وجود ندارد.
''')
    code("import base64,io,zipfile,sys,json\nfrom pathlib import Path\nimport torch\nROOT=Path.cwd()/'magnetic_operator_run'\nROOT.mkdir(exist_ok=True)\nwith zipfile.ZipFile(io.BytesIO(base64.b64decode('"+blob+"'))) as z:z.extractall(ROOT)\nsys.path.insert(0,str(ROOT))\nOUTPUT=ROOT/'outputs'\nOUTPUT.mkdir(exist_ok=True)\n")
    if pretrained:
        code("import shutil\nfrom magnetic_operator.train import load,train\nTRAIN_FROM_SCRATCH=False\nif TRAIN_FROM_SCRATCH:\n    model,info=train(OUTPUT,steps="+str(steps)+",require_gpu=True)\nelse:\n    for p in (ROOT/'pretrained').iterdir():shutil.copyfile(p,OUTPUT/p.name)\n    model=load(OUTPUT)\n    info=json.loads((OUTPUT/'training.json').read_text())\nprint(json.dumps(info,indent=2))\n")
    else:
        code("from magnetic_operator.train import train\nassert torch.cuda.is_available(), 'GPU requested; CUDA unavailable'\nmodel,info=train(OUTPUT,steps="+str(steps)+",batch=1024,require_gpu=True)\nprint(json.dumps(info,indent=2))\n")
    md('## اعتبارسنجی مستقل\nآستانه‌ها قبل از اجرای GPU تعریف شده‌اند. خطای مدل با حل DOP853 با پیاده‌سازی مستقل میدان، مسیرهای بلند با توابع دیده‌نشده، تقریب اینرسی، شرط اولیه و رد ورودی نامعتبر بررسی می‌شود. شکست آزمون اجرای موفق را متوقف می‌کند.')
    code("from magnetic_operator.validate import run\nreport=run(OUTPUT,require_gpu=True)\nprint('ALL_TESTS_PASSED:',report['all_passed'])\nfrom IPython.display import Image,display\ndisplay(Image(filename=str(OUTPUT/'unseen_trajectories.png')))\n")
    md('## آزمون‌های تکمیلی دامنه\nخواص مرزی سیال، رهایش متغیر، شدت حداکثر منابع، ناوردایی چگالی و خروج از دامنه جدا بررسی می‌شوند.')
    code("from magnetic_operator.extended_checks import run as extended_checks\nextended_report=extended_checks(OUTPUT)\nprint('ALL_EXTENDED_TESTS_PASSED:',extended_report['all_passed'])\n")
    md('## تابع میدان دلخواه\nزمان ورودی تابع برحسب ثانیه است. خروجی دو عدد بدون بعد است؛ |a_left|+|a_right| باید حداکثر sqrt(0.5) باشد. نمونهٔ زیر بعد از آموزش استفاده می‌شود؛ بازآموزی ندارد. ورودی غیرقابل تفکیک یا خروج ذره از دامنه خطا می‌دهد. برای تابع با پرش، حل را در زمان پرش جدا کنید یا از حل مرجع استفاده کنید.')
    code("import numpy as np\nfrom magnetic_operator.train import load\nfrom magnetic_operator.physics import Parameters\nfrom magnetic_operator.inference import predict,reference\nmodel=load(OUTPUT)\ndef my_field(t):\n    return .25*np.sin(2*np.pi*.6*t), .15*np.exp(-((t-1.5)/.3)**2)\ntimes=np.linspace(0,3,301)\npath,metadata=predict(model,my_field,times,Parameters())\nnp.savetxt(OUTPUT/'my_trajectory.csv',np.column_stack((times,path)),delimiter=',',header='time_s,x_m,z_m',comments='')\nprint(metadata)\n")
    md('## دریافت بستهٔ کامل\nپس از پاس‌شدن آزمون‌ها، فایل ZIP شامل کد مستقل، وزن‌های GPU، نمونهٔ CSV، گزارش فارسی و نتایج در بخش Output همین نوت‌بوک ساخته می‌شود.')
    code("import shutil,runpy,zipfile\nRESULTS=ROOT/'results/operator'\nRESULTS.mkdir(parents=True,exist_ok=True)\nfor p in OUTPUT.iterdir():\n    if p.is_file():shutil.copyfile(p,RESULTS/p.name)\nrunpy.run_path(str(ROOT/'scripts/operator/export_report.py'),run_name='__main__')\nBUNDLE=Path.cwd()/'Master10_Magnetic_Operator.zip'\nwith zipfile.ZipFile(BUNDLE,'w',zipfile.ZIP_DEFLATED) as z:\n    for folder in ('magnetic_operator','scripts/operator','results/operator','examples'):\n        for p in sorted((ROOT/folder).rglob('*')):\n            if p.is_file() and '__pycache__' not in p.parts:\n                z.write(p,Path('Master10_Magnetic_Operator')/p.relative_to(ROOT))\n    z.write(ROOT/'requirements.txt','Master10_Magnetic_Operator/requirements.txt')\n    z.write(ROOT/'magnetic_operator/README_FA.md','Master10_Magnetic_Operator/README.md')\nshutil.copyfile(RESULTS/'REPORT_FA.md',Path.cwd()/'REPORT_FA.md')\nshutil.copyfile(ROOT/'magnetic_operator/README_FA.md',Path.cwd()/'GUIDE_FA.md')\nshutil.copyfile(RESULTS/'unseen_trajectories.png',Path.cwd()/'unseen_trajectories.png')\nprint('DOWNLOAD:',BUNDLE)\n")
    n={'nbformat':4,'nbformat_minor':4,'cells':cells,'metadata':{'kernelspec':{'name':'python3','display_name':'Python 3','language':'python'},'language_info':{'name':'python'}}}
    destination=Path(destination);destination.parent.mkdir(parents=True,exist_ok=True)
    destination.write_text(json.dumps(n,ensure_ascii=False,indent=1))
    return destination

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',default=str(ROOT/'notebooks/magnetic_operator.ipynb'))
    p.add_argument('--pretrained',action='store_true');p.add_argument('--steps',type=int,default=18000)
    a=p.parse_args();print(build(a.output,a.pretrained,a.steps))

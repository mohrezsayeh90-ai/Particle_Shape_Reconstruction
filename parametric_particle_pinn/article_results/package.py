"""Build the bilingual figure inventory and verify its archived provenance."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory

import numpy as np
import torch

from ..model import load_checkpoint
from ..thesis_results.audit import check
from ..thesis_results.backend import DirectPINN
from ..thesis_results.evidence import ARCHIVE, CHECKPOINT, ROOT, read_csv, sha, write_csv, write_json
from .data import (ABLATION, CLEARANCE, COMMIT, CONTACT_TOLERANCE, DENSITY, DUCK,
                   GALLERY, LETTERS, RADIUS, SOURCE, TRIANGLE, document)
from . import plots


def check_source(ev,out):
    relative = str(SOURCE.relative_to(ROOT))
    subprocess.run(["git","diff","--quiet",COMMIT,"--",relative,
                    str(ARCHIVE.relative_to(ROOT)),"parametric_particle_pinn/model.py",
                    "parametric_particle_pinn/validate.py","parametric_particle_pinn/inference.py",
                    "parametric_particle_pinn/thesis_results"],cwd=ROOT,check=True)
    assert sha(CHECKPOINT)==ev.manifest["weight_sha256"]
    counts = Counter(r["status"] for r in ev.rows)
    assert counts=={"accepted":132,"rejected":32,"preflight_rejected":3}
    torch.set_num_threads(1)
    model=load_checkpoint(CHECKPOINT)
    with TemporaryDirectory(prefix="article-audit-") as path:
        stage=Path(path)
        for name in ("validation","formation"):
            (stage/name).symlink_to(SOURCE/name,target_is_directory=True)
        audit=check(model,DirectPINN(model),stage,list(ev.matrix.values()))
    write_json(out/"data/archive_audit.json",audit)


def build(ev,out,save):
    check_source(ev,out)
    with np.load(SOURCE/"validation/time_series.npz") as src:
        data = {k:src[k] for k in src.files}
    rows=read_csv(SOURCE/"validation/case_diagnostics.csv")
    summary=document(SOURCE/"validation/summary.json")
    catalog=[]

    def add(name,status,figure,ids,source,english,persian,engine="PINN; Stokes ODE (DOP853)",saved=False):
        width = 177.8 if saved else round(figure.get_figwidth()*25.4,2)
        if not saved:
            save(out,name,figure)
        catalog.append(dict(id=name.split("_",1)[0],pdf=f"figures/{name}.pdf",png=f"figures/{name}.png",
                            status=status,source_case_ids=";".join(ids),engine=engine,
                            constraints="Fixed Stokes/Poiseuille physics; dimensionless; formation d_p*=0.025, d_acc*=0.026",
                            source=source,script="python -m parametric_particle_pinn.article_results",
                            width_mm=width,caption_en=english,caption_fa=persian))

    def selected(ids):
        return "; ".join(f"formation/{case}/" for case in ids)

    add("P01_method","revised main",plots.method(),[],"parametric_particle_pinn/model.py; manifest.json",
        "Direct conditional particle PINN at fixed physical parameters. The prescribed Poiseuille profile enters the Stokes residual. The hard ansatz enforces the released position and velocity.",
        "PINN مستقیماً موقعیت ذره را از وضعیت رهاسازی و زمان می‌دهد. پروفیل تجویزی پوازوی در باقیماندهٔ استوکس استفاده می‌شود و ساختار خروجی موقعیت و سرعت اولیه را اعمال می‌کند.",
        "Direct particle PINN")
    vsource="validation/time_series.npz; validation/case_diagnostics.csv"
    path_en="DOP853 and PINN stop at their own event. Local channel views use equal scale. The lower strip shows dimensionless Euclidean position error on the joint pre-termination interval; exact zeros are omitted from the log axis."
    path_fa="مسیرهای PINN و DOP853 تا رویداد خودشان رسم شده‌اند. بزرگ‌نمایی‌های هندسی هم‌مقیاس‌اند و نوار پایین خطای اقلیدسی موقعیت را در بازهٔ مشترک پیش از توقف نشان می‌دهد. صفرها روی محور لگاریتمی رسم نشده‌اند."
    add("P02_trajectories","revised main",plots.paths(data,rows),plots.EXAMPLES,vsource,path_en,path_fa)
    add("P02_supplement","revised supplement",plots.paths(data,rows,plots.legacy.REPRESENTATIVES),
        plots.legacy.REPRESENTATIVES,vsource,path_en,path_fa)
    idx=[list(data["case_id"]).index(k) for k in plots.legacy.REPRESENTATIVES]
    keep=("case_id","pinn_display_time","dop853_display_time","pinn_display_position",
          "dop853_display_position","common_time","signed_error")
    np.savez_compressed(out/"data/representative_paths.npz",**{k:data[k][idx] for k in keep})
    add("P03_validation","revised main",plots.accuracy(rows,summary,out,data),list(data["case_id"]),vsource,
        "Peak dimensionless state errors over each joint pre-termination interval, signed event-time difference, and event-type counts. Crosses denote exact zeros at a display level of 10⁻¹⁶. Event-type agreement and composite acceptance were each 70/70. The accompanying CSV gives median, 95th percentile and maximum.",
        "بیشینهٔ خطاهای بی‌بُعد در بازهٔ مشترک پیش از توقف، اختلاف علامت‌دار زمان رویداد و ماتریس نوع رویداد. ضربدرها صفر دقیق را در تراز نمایشی ۱۰⁻¹⁶ نشان می‌دهند. تطابق نوع رویداد و قبولی مرکب هر دو ۷۰ از ۷۰ بود.")
    add("P03_validation_single","revised single-column option",
        plots.accuracy(rows,summary,out,data,single=True),list(data["case_id"]),vsource,
        catalog[-1]["caption_en"],catalog[-1]["caption_fa"])
    add("P04_training","revised supplement",plots.training(),[],"parametric_particle_pinn/results/cpu/training.jsonl",
        "Archived training and development residuals. The selected checkpoint was step 4808; L-BFGS used 208 iterations and 353 closure evaluations.",
        "باقیمانده‌های آموزش و توسعه برای وزن منتخب گام ۴۸۰۸. مرحلهٔ L-BFGS شامل ۲۰۸ تکرار و ۳۵۳ ارزیابی تابع closure است.",
        "Recorded training and development log")
    add("P05_release_slices","revised supplement",plots.slices(read_csv(SOURCE/"slices/grid.csv")),
        [],"slices/grid.csv; slices/matched.npz",
        "Parameter slices at x₀*=0.66 and interior vₓ₀*=0.35. Wall grids are 17×19 and the interior grid is 19×19 (1007 releases). Crosses mark unavailable states at t*=1. Error is log₁₀ of peak component-wise absolute position error on the common valid interval, with floor 10⁻¹³.",
        "برش شرایط رهاسازی در x₀*=۰٫۶۶ و سرعت افقی داخلی ۰٫۳۵. شبکه‌های دیواره ۱۷×۱۹ و شبکهٔ داخلی ۱۹×۱۹ هستند (۱۰۰۷ شرط). ضربدرها وضعیت ناموجود در t*=۱ را نشان می‌دهند. خطا لگاریتم ده‌دهی بیشینهٔ خطای مؤلفه‌ای موقعیت در بازهٔ مشترک معتبر است.")
    add("F01_verified_triangle","revised main option",plots.triangle(ev),[TRIANGLE],selected([TRIANGLE]),
        "Verified three-particle triangle. The shared clock distinguishes target time from the minimum-distance time of pair 2/3. Its DOP853 minimum distance was 0.0375901. Paths may cross at different global times.",
        "مثلث سه‌ذره‌ای تأییدشده. ساعت مشترک زمان هدف را از زمان کمترین فاصلهٔ جفت ۲ و ۳ جدا می‌کند. کمترین فاصله در بازپخش ۰٫۰۳۷۵۹۰۱ بود. مسیرها ممکن است در زمان‌های متفاوت از یک نقطه عبور کنند.")
    matching=[r["id"] for r in ev.rows if r["study"]=="density" and r["N"]=="48"]
    add("F02_matched_density","revised main",plots.density(ev,out),matching,selected(DENSITY),
        "Matched 48-particle stadium, seed 11, common target hash. Six fixed ports and free wall each passed 3/3. DOP853 margins were +5.05132×10⁻⁴ and +2.88516×10⁻⁴; endpoint RMS/max were 6.594×10⁻⁹/1.428×10⁻⁸ and 6.163×10⁻⁹/1.514×10⁻⁸, respectively.",
        "استادیوم ۴۸ ذره‌ای با هدف یکسان و بذر ۱۱. هر دو حالت ۳ از ۳ قبول شدند؛ حاشیهٔ فاصله در حالت ثابت ‎+۵٫۰۵۱۳۲×۱۰⁻⁴ و در حالت آزاد ‎+۲٫۸۸۵۱۶×۱۰⁻⁴ بود. جدول همراه RMS و بیشینهٔ خطای انتهایی را ثبت می‌کند.")
    fig,ids=plots.sweep(ev,out)
    add("F03_formation_sweeps","revised main",fig,ids,"formation/summary.csv; formation/matrix.json",
        "Attempted success fractions and individual seed margins by density, launcher layout and per-port capacity. At cap 3, six ports supply 18 slots for 24 targets, so no attempt was made. Every N=100 run reached T* and failed spacing. Unlimited cap still produced rejected replays.",
        "نسبت موفقیت و حاشیهٔ هر بذر برحسب تعداد ذرات، چیدمان و ظرفیت. ظرفیت ۳ برای شش پورت فقط ۱۸ جایگاه برای ۲۴ هدف می‌دهد. همهٔ موارد ۱۰۰ ذره‌ای به زمان هدف رسیدند و در قید فاصله رد شدند. ظرفیت نامحدود تضمین موفقیت نبود.")
    fig,ids=plots.geometry(ev,out)
    add("F04_geometry_slices","revised supplement",fig,ids,"formation/matrix.json; formation/summary.csv",
        "Acceptance counts for seeds 11, 29 and 47. Stadium N=18 varies center at a fixed outline. Ellipse N=12 varies aspect and angle at fixed area and center. Angle is immaterial for the circle. Both use six equispaced fixed ports.",
        "شمار قبولی بذرهای ۱۱، ۲۹ و ۴۷. مرکز استادیوم ۱۸ ذره‌ای با محیط ثابت تغییر می‌کند. نسبت محورها و زاویهٔ بیضی ۱۲ ذره‌ای در مساحت و مرکز ثابت تغییر می‌کند؛ زاویهٔ دایره اثری ندارد. هر دو شش پورت ثابت هم‌فاصله دارند.")
    letter_ids=[r["id"] for r in ev.rows if r["shape"] in ("S_6","C_6","T_6")]
    add("F05_letters","revised main option",plots.gallery(ev,out,LETTERS,("S","C","T"),"letters.csv"),
        letter_ids,selected(LETTERS),"S, C and T with six fixed launch ports (seed 11). Disks are replay centers at T*, and dotted segments are target guides. Each letter passed 3/3 seeds. Per-case errors and margins are in letters.csv.",
        "حروف S، C و T با شش پورت ثابت و بذر ۱۱. دیسک‌ها در زمان هدف از بازپخش و خطوط نقطه‌ای راهنمای هدف‌اند. هر حرف در ۳ از ۳ بذر قبول شد.")
    add("F06_duck","new main option",None,[r["id"] for r in ev.rows if "duck" in r["study"]],selected(DUCK),
        "Seed-11 historical duck targets with 20 fixed and 28 free-wall releases. The panels differ in N and target coordinates. DOP853 margins were +4.2360×10⁻⁴ and +5.9577×10⁻⁴. The 42 and 64 cases all had disk overlap during motion.",
        "دو هدف تاریخی اردک با بذر ۱۱، شامل ۲۰ ذره با پورت ثابت و ۲۸ ذره با مکان آزاد. تعداد و مختصات هدف دو مثال متفاوت‌اند. حاشیه‌ها به‌ترتیب ‎+۴٫۲۳۶۰×۱۰⁻⁴ و ‎+۵٫۹۵۷۷×۱۰⁻⁴ بودند. همهٔ حالت‌های ۴۲ و ۶۴ ذره‌ای هم‌پوشانی داشتند.",saved=True)
    add("F07_clearance_ablation","new main",None,[r["id"] for r in ev.rows if r["study"]=="clearance_objective"],
        selected(ABLATION)+"; data/clearance_pair_traces.csv",
        "Matched 18-particle schedules. Spacing-aware allocation enabled the pair-distance penalty in candidate scoring; other settings were identical. Both illustrated schedules completed at T*. Across seeds, endpoint-only accepted 0/3 and spacing-aware 3/3. Panel c traces each schedule's archival critical pair only; dots mark refined minima and vertical dotted lines mark the respective T*. Zero distance at shared-presence start is geometric overlap.",
        "دو زمان‌بندی هم‌شرط ۱۸ ذره‌ای. در تخصیص حساس به فاصله، جریمهٔ فاصلهٔ جفت‌ها در امتیاز گزینه‌ها فعال شد. هر دو نمونه در زمان هدف کامل شدند. معیار قبولی برای روش صرفاً انتهایی ۰ از ۳ و با قید فاصله ۳ از ۳ بود. پنل c فقط جفت بحرانی هر اجرا را نشان می‌دهد؛ نقطه کمینه و خط عمودی نقطه‌ای زمان هدف هر اجراست. فاصلهٔ صفر در شروع هم‌حضوری یعنی هم‌پوشانی هندسی.",saved=True)
    gallery_ids=[r["id"] for r in ev.rows if r["study"]=="shape" or r["study"]=="density" and r["N"]=="72"]
    add("F08_shape_gallery","new supplement",plots.gallery(ev,out,GALLERY,
        ("Fish · fixed","Arrow · fixed","Stadium 72 · free wall"),"shape_gallery.csv"),
        gallery_ids,selected(GALLERY),"Seed-11 target snapshots with true replay disks. Fish and arrow passed 3/3 with six fixed ports at N=24. At N=72 the free stadium passed 2/3 and the matched fixed mode 0/3.",
        "نماهای زمان هدف در بذر ۱۱. ماهی و پیکان با ۲۴ ذره و شش پورت ثابت هر دو ۳ از ۳ قبول شدند. استادیوم ۷۲ ذره‌ای در حالت آزاد ۲ از ۳ و در حالت ثابت ۰ از ۳ قبول شد.")
    fig,ids=plots.reflection(ev,out)
    add("F09_reflection","new supplement",fig,ids,"formation/matrix.json; formation/summary.csv",
        "Actual original and reflected six-port layouts on the same 18-particle target. Recorded seed outcomes were 2/3 and 3/3 at the stated search budget.",
        "چیدمان واقعی شش پورت در حالت اصلی و قرینه برای هدف یکسان ۱۸ ذره‌ای. نتیجه در بودجهٔ ثبت‌شده به‌ترتیب ۲ از ۳ و ۳ از ۳ بود.")
    write_csv(out/"figure_catalog.csv",catalog)
    used=sorted(ev.used)
    np.savez_compressed(out/"data/selected_snapshots.npz",**{
        f"{key}__{column}":array for key in used for column,array in ev.case(key)[2].items()
        if column in ("release","release_time","target","dop853_endpoint","pinn_endpoint")})
    write_json(out/"manifest.json",dict(baseline_commit=COMMIT,archive=str(SOURCE.relative_to(ROOT)),
        weight_sha256=ev.manifest["weight_sha256"],config_sha256=ev.manifest["config_canonical_json_sha256"],
        figure_count=len(catalog),seed_bank=[11,29,47],illustrated_seed=11,selected_cases=used,
        experiment_signatures={k:dict(target_sha256=ev.matrix[k]["target_sha256"],
            case_definition_sha256=hashlib.sha256(json.dumps(ev.matrix[k],sort_keys=True,
            separators=(",",":")).encode()).hexdigest(), seed=ev.matrix[k]["seed"],
            status=ev.by_id[k]["status"]) for k in used},
        generator_sha256={p.name:sha(p) for p in sorted(Path(__file__).parent.glob("*.py"))},
        source_file_sha256=ev.source_hashes(),closed_form_used_in_article_figures=False,
        disk_diameter=2*RADIUS,acceptance_distance=CLEARANCE,overlap_classification_tolerance=CONTACT_TOLERANCE,
        searches_launched=0,independent_replay="DOP853 critical pair traces from frozen schedules",
        print_widths_mm=[85,177.8],png_dpi=320,
        recommended_main=["P01","P02","P03","F02","F07","F03"]))

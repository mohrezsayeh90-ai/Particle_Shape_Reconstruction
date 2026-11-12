# موجودی نتایج جدید تشکیل آرایش

این بسته شامل نتایج محاسبه‌شده و فهرست صریح اجراهای باقی‌مانده است.
پذیرش هر حالت با دو بازپخش مستقل و بررسی همهٔ جفت‌ها تعیین می‌شود.
متن پروپوزال و PDF قبلی در این مرحله تغییر نکرده‌اند؛ فقط همین پوشه بین دو مخزن کپی می‌شود.

تعداد اجراهای جدید مستقل: **170**؛ پذیرفته‌شده: **126**؛ نپذیرفته‌شده: **44**.
این شمارش، بازپخش چهار اردک تاریخی و بازاجراهای مثلث و حروف را شامل نمی‌شود.
شمارش جدول زیر مربوط به همهٔ درخواست‌های ماتریس است و درخواست‌های هم‌ارزِ بازاستفاده‌شده را نیز دربرمی‌گیرد.

| وضعیت | تعداد |
| --- | ---: |
| accepted | 153 |
| rejected | 49 |
| pending | 782 |
| running | 0 |
| excluded_before_solve | 30 |

[گزارش کوتاه](EXPERIMENT_REPORT_FA.md) · [فهرست همهٔ حالت‌ها و وضعیت آن‌ها](runs/formation_20261112_cpu/cases_manifest.csv) · [خلاصهٔ ادغام‌شده](runs/formation_20261112_cpu/summary.csv) · [روش بازتولید و ادامهٔ اجرا](REPRODUCE.md)

## بالاترین تعداد تأییدشده در اجراهای جدید

| ماسک | نوع رهاسازی | تعداد ذره |
| --- | --- | ---: |
| arrow | fixed | 24 |
| arrow | free | 100 |
| ellipse | fixed | 12 |
| fish | fixed | 24 |
| fish | free | 100 |
| perforated | fixed | 24 |
| perforated | free | 100 |
| small_stadium | fixed | 18 |
| small_stadium | free | 18 |
| smoke | fixed | 6 |
| smoke | free | 6 |
| stadium | fixed | 24 |
| stadium | free | 100 |

این‌ها بالاترین تعداد **تأییدشده در دادهٔ موجود** هستند؛ حد ظرفیت فیزیکی یا تعداد بهینه نیستند.

## مقایسه‌های کامل سه-seed

| گروه | مقایسه‌های کامل | پذیرفته / انجام‌شده | شروع‌های بازاستفاده‌شده |
| --- | ---: | ---: | ---: |
| B1 | 7 | 17 / 21 | 0 |
| C1 | 1 | 3 / 3 | 3 |
| C1_one_shot | 1 | 3 / 3 | 0 |
| D1 | 3 | 9 / 9 | 9 |
| F1 | 19 | 55 / 57 | 0 |
| F2 | 5 | 14 / 15 | 15 |
| G2 | 2 | 3 / 6 | 0 |

بقیهٔ مقایسه‌ها غربالگری یا اجراهای ناقص‌اند؛ هر برنامهٔ پذیرفته‌شده، حتی در غربالگری، آزمون سخت‌گیرانهٔ مستقل را گذرانده است.
بودجهٔ پایهٔ جدید: سقف ۲۴ ارزیابی برازش اولیه و ۵۰/۳۵ ارزیابی مراحل مشترک؛ سه seed برابر 11، 29 و 47.
پراکندگی seedها حساسیت جست‌وجو است و عدم‌قطعیت اندازه‌گیری تجربی محسوب نمی‌شود.

## شواهد تاریخی و بازاجراها

چهار برنامهٔ اردک 20/28/42/64 از آرشیو، با هر دو حل‌گر دوباره بازپخش شده‌اند؛ مقادیر قدیمی در کنار محاسبات تازه جدا نگه داشته شده‌اند.
بستهٔ خروجی اصلی مثلث و S/C/T در آرشیو موجود نبود و مسیر عمومی Kaggle با HTTP 403 بسته بود؛ کد اصلی و وزن ذخیره‌شده برای بازاجرای جداگانه استفاده شده است.
تنظیمات تاریخی حروف، از جمله سقف مدت 8، حفظ شده‌اند و با سقف مدت 5 در مقایسه‌های جدید مخلوط نمی‌شوند.

- A1_triangle: rerun_accepted · [معیارهای بازاجرای تازه](runs/formation_20261112_cpu/historical_reruns/triangle_1794495372673521753/fresh_verification/metrics.json)
- A1_S: rerun_accepted · [معیارهای بازاجرای تازه](runs/formation_20261112_cpu/historical_reruns/letters_1794495952386815010/fresh_verification/S_6/metrics.json)
- A1_C: rerun_accepted · [معیارهای بازاجرای تازه](runs/formation_20261112_cpu/historical_reruns/letters_1794495952386815010/fresh_verification/C_6/metrics.json)
- A1_T: rerun_accepted · [معیارهای بازاجرای تازه](runs/formation_20261112_cpu/historical_reruns/letters_1794495952386815010/fresh_verification/T_6/metrics.json)

## شکل‌ها

[نمای یک‌جای شکل‌ها](runs/formation_20261112_cpu/figures/contact_sheet.png) · [ورودی‌ها و هش هر شکل](runs/formation_20261112_cpu/figures_manifest.json)

- [F01: بازاجرای تازهٔ مثلث و حروف با تنظیمات تاریخی.](runs/formation_20261112_cpu/figures/F01.pdf) · [PNG](runs/formation_20261112_cpu/figures/F01.png)
- [F02: مقایسهٔ آرایش اردک در زمان هدف مشترک.](runs/formation_20261112_cpu/figures/F02.pdf) · [PNG](runs/formation_20261112_cpu/figures/F02.png)
- [F03: اثر تعداد پرتابگرها و ذرات بر کیفیت آرایش.](runs/formation_20261112_cpu/figures/F03.pdf) · [PNG](runs/formation_20261112_cpu/figures/F03.png)
- [F03_ports: مختصات ثابت پرتابگرها در چیدمان هم‌فاصله و تودرتو.](runs/formation_20261112_cpu/figures/F03_ports.pdf) · [PNG](runs/formation_20261112_cpu/figures/F03_ports.png)
- [F04: سقف انتشار، بار واقعی پرتابگرها و زمان‌بندی رهاسازی.](runs/formation_20261112_cpu/figures/F04.pdf) · [PNG](runs/formation_20261112_cpu/figures/F04.png)
- [F05: آرایش‌های هدف پُر و وضعیت نتایج محاسبه‌شده.](runs/formation_20261112_cpu/figures/F05.pdf) · [PNG](runs/formation_20261112_cpu/figures/F05.png)
- [F05_occupancy: پوشش، بیرون‌زدگی و نفوذ به حفره‌های ماسک.](runs/formation_20261112_cpu/figures/F05_occupancy.pdf) · [PNG](runs/formation_20261112_cpu/figures/F05_occupancy.png)
- [F06: اثر مرکز هدف بر پذیرش، خطا و فاصلهٔ کمینه.](runs/formation_20261112_cpu/figures/F06.pdf) · [PNG](runs/formation_20261112_cpu/figures/F06.png)
- [F06_slices: برش‌های طولی و عرضی مرکز هدف با نمایش جداگانهٔ seedها.](runs/formation_20261112_cpu/figures/F06_slices.pdf) · [PNG](runs/formation_20261112_cpu/figures/F06_slices.png)
- [F07: مقایسهٔ کوواریانس هدف و آرایش حاصل و نقشهٔ هندسهٔ بیضی.](runs/formation_20261112_cpu/figures/F07.pdf) · [PNG](runs/formation_20261112_cpu/figures/F07.png)
- F08: pending — central controls reused from F1; remaining locations and extended ellipse geometries pending
- [F09: فاصلهٔ کمینه، تعداد ذرات فعال و سازوکار نزدیک‌ترین جفت.](runs/formation_20261112_cpu/figures/F09.pdf) · [PNG](runs/formation_20261112_cpu/figures/F09.png)
- [F10: بررسی عددی تقارن با بازتاب یک برنامهٔ پذیرفته‌شده.](runs/formation_20261112_cpu/figures/F10.pdf) · [PNG](runs/formation_20261112_cpu/figures/F10.png)
- [F11: اثر هزینهٔ فاصله بر برنامه‌ریزی و وضعیت پیمایش زمان هدف.](runs/formation_20261112_cpu/figures/F11.pdf) · [PNG](runs/formation_20261112_cpu/figures/F11.png)
- [F12: حساسیت عددی برنامهٔ ثابت به خطاهای رهاسازی.](runs/formation_20261112_cpu/figures/F12.pdf) · [PNG](runs/formation_20261112_cpu/figures/F12.png)
- F13: pending — optional held-out layout design not scheduled

## دامنهٔ نتیجه

خطاها و فاصله‌ها، بررسی عددی در مدل ثابت هستند.
این خروجی‌ها تأیید سخت‌افزار یا اثبات برخوردناپذیری پیوسته نیستند.
شکست جست‌وجو نیز اثبات ناممکن‌بودن فیزیکی آرایش نیست.
تغییر مدت پرواز و بودجه، در تنظیمات هر اجرا مشخص شده است.
مرجع چهار اردک، برنامه‌های قدیمی است؛ محاسبات بازپخش تازه جدا ذخیره شده‌اند.
نبود خروجی اصلی مثلث و حروف، به‌عنوان نتیجهٔ علمی شکست‌خورده ثبت نشده است.
جدول بالاترین تعداد پذیرفته‌شده در `runs/formation_20261112_cpu/tables/highest_verified_counts.csv` موجود است.
جدول مقایسه‌ها، تعداد شروع‌های کامل‌شده را صریحاً نگه می‌دارد.
فقط ردیف‌های `matched_three_seed_complete=True` مقایسهٔ کامل سه‌بذری هستند.

## محاسبات باقی‌مانده

محیط CPU-only با ۵ پردازندهٔ تخصیص‌یافته است. ماتریس 1014 درخواستی از بودجهٔ این مرحله بزرگ‌تر است؛ اجرای کامل همهٔ حالت‌ها ادعا نمی‌شود.
وضعیت دقیق تک‌تک حالت‌های اجرا‌نشده در CSV ثبت شده و دستور `--resume` آن‌ها را ادامه می‌دهد.
پیمایش تودرتو، چگالی‌های میانی ماسک‌ها، seedهای باقی‌ماندهٔ مقایسه‌های متراکم و مرکز، شبکهٔ توسعه‌یافتهٔ بیضی، دیواره‌ها و زمان هدف فقط به اندازهٔ وضعیت ثبت‌شده کامل‌اند.
طرح اختیاری G5 برنامه‌ریزی نشده است؛ F08 و هر مؤلفهٔ فاقد داده در شکل‌های دیگر به‌صراحت pending است.
کد رسم از فایل‌های ذخیره‌شده استفاده می‌کند و بهینه‌سازی یا آموزش دوباره انجام نمی‌دهد.

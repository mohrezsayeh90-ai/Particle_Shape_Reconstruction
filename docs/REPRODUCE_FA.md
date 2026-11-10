# راهنمای اجرای دوباره و آموزش

همهٔ سیزده فایل اصلی در `evidence/<نام نوت‌بوک>/` قرار دارند. snapshot اصلی را تغییر ندهید؛ برای آزمایش تازه، یک خروجی تازه بسازید. بسته‌های کد و وزن‌های داخل خود notebook نیز در `assets/<نام نوت‌بوک>/` استخراج شده‌اند و در `extraction.json` هر مورد فهرست شده‌اند.

## نصب

Python 3.12 توصیه می‌شود. برای اجرای CPU و بازبینی وزن‌های موجود:

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-execution.txt
```

برای آموزش GPU، به‌جای wheel پردازنده، PyTorch سازگار با CUDA دستگاه را از منبع رسمی نصب کنید. نسخه‌های دقیق محیطی که در بررسی استفاده شد در `verification/software.json` ثبت شده‌اند. بسته‌های لازم نسخه‌های قدیمی کانال خمیده نیز در `assets/master-1/packages/package_01/requirements.txt` و مسیر مشابه Master2 آمده‌اند؛ از جمله scikit-fem. این وابستگی‌ها برای بازپخش چهار اردک لازم نیستند.

## بازتولید نتیجه بدون آموزش مجدد

```sh
# بازپخش مستقل ۱۵۴ ذره در چهار آزمایش ذخیره‌شده و کنترل فاصله‌ها
python tools/replay.py
# آزمون‌های اصل مدل مغناطیسی با checkpointهای موجود
python tools/revalidate_magnetic.py master9
python tools/revalidate_magnetic.py master10-magnetic-function-deeponet
# تولید شکل مقایسه و صفحهٔ آفلاین از مسیرهای بازپخش
python tools/render_replay.py
```

این دستورها اجرا و بررسی شده‌اند. وزن‌ها با `weights_only=True` بارگذاری می‌شوند. `replay.py` تنظیمات تزریق را دوباره بهینه نمی‌کند و آموزش جدید انجام نمی‌دهد؛ همان تنظیمات ذخیره‌شده را با شبکهٔ ثابت و مرجع مستقل می‌آزماید.

## اجرای کامل فایل اصلی اردک متراکم

```sh
python tools/run_notebook.py dense-duck-formation-20261104-211d7de258 \
  --output experiments/duck-replay-001
```

فایل اصلی شامل وزن آموزش‌دیده و نتایج cache است. در این حالت وزن‌ها بارگذاری و اعتبارسنجی می‌شوند، نتیجه‌های سازگارِ قبلی بازپخش می‌شوند و PNG/PDF/HTML/JSON/CSV/ZIP ساخته می‌شوند. این اجرا آموزش یا جست‌وجوی معکوس تازه نیست. پوشهٔ `reproduced_runs/dense-duck-cpu/` اجرای همین دستور در این محیط است؛ `execution.json` و `executed.ipynb` نتیجه و خطاهای احتمالی را ثبت می‌کنند.

برای **جست‌وجوی معکوس تازهٔ برنامهٔ رهاسازی** با همان وزن و مدل فیزیکی:

```sh
python tools/run_notebook.py dense-duck-formation-20261104-211d7de258 \
  --reoptimize --output experiments/duck-new-optimization-001
```

این گزینه cache اردک را فقط در کپیِ اجرای جدید خالی می‌کند. کد منبع آرشیوشده تغییر نمی‌کند. بهینه‌سازی ۴۲/۶۴ ذره ممکن است بسیار زمان‌بر باشد و موفقیت هر هدف یا هر شروعی تضمین نشده است. این حالت در این انتقال دوباره اجرا نشده است.

## آموزش شبکه و حل تشکیل شکل از ابتدا

```sh
python tools/run_notebook.py master3-clean-revised-version \
  --output experiments/master3-new-training-001
```

این notebook، پیش‌فرض از ابتدا شبکهٔ جریان را آموزش می‌دهد و سپس برنامه‌های تشکیل شکل را حل می‌کند. GPU برای هزینهٔ آموزش توصیه می‌شود. برای آنکه دو اردک ضمیمه نیز از cache استفاده نکنند، `--reoptimize` اضافه کنید. سایر شکل‌ها در مسیر اصلی خود notebook بهینه می‌شوند. بازه‌ها، قانون نیرو و آستانه‌های پذیرش خودبه‌خود تغییر نمی‌کنند.

## آموزش مجدد مدل‌های مغناطیسی

```sh
python tools/run_notebook.py master9 --fresh-training --output experiments/master9-new-training-001
python tools/run_notebook.py master10-magnetic-function-deeponet --fresh-training --output experiments/master10-new-training-001
```

این دو دستور برای آموزش GPU طراحی شده‌اند و روی CPU به‌جای آموزشِ درخواست‌شدهٔ GPU، نتیجهٔ ساختگی تولید نمی‌کنند. اجرای استنتاج و اعتبارسنجی CPU با ابزارهای قبلی در دسترس است. برای Flexible Controller، همین گزینه فلگ موجود `TRAIN_AGAIN` را فعال می‌کند؛ این بازآموزیِ سیاست کنترل است و جای بازآموزی میدان‌ها را نمی‌گیرد.

در نسخه‌های Master7 و Master8، فلگ اصلی `TRAIN_FROM_SCRATCH` با `--fresh-training` قابل فعال‌کردن است. Master6 به‌طور پیش‌فرض شبکه‌های بخش‌بندی‌شده را آموزش می‌دهد؛ checkpointهای قدیمی آن در source تعبیه نشده‌اند. Master5 حل عددی است و شبکهٔ آموزش‌دیده ندارد. Master4 در بررسی قبلی رد شده است؛ اجرای آن برای توسعه و تشخیص خطاست، نه تولید خروجی از پیش معتبر.

## ثبت اجرای تازه

`run_notebook.py` یک kernel با همان Python فعلی می‌سازد، در پوشهٔ خروجی تازه اجرا می‌کند و منبع مورد استفاده، SHA-256، فلگ‌های تغییرکرده، وضعیت و زمان اجرا را در `execution.json` ثبت می‌کند. فایل‌های اصلی، token و تنظیمات حساب تغییر نمی‌کنند. از پوشهٔ خروجی جدید استفاده کنید؛ پوشهٔ غیرخالی پذیرفته نمی‌شود.

برای گرفتن نسخهٔ تازه از حساب Kaggle و فایل‌های خروجی آن:

```sh
python -m pip install -r requirements-kaggle.txt
python tools/snapshot_kaggle.py --download-outputs
```

Secret با نام `KAGGLE_API_TOKEN` و شبکهٔ مجاز API/CDN لازم است. برای تکمیل آرشیو خروجی‌های فعلی نیز `python tools/download_original_outputs.py` وجود دارد. وضعیت دریافت اصلی در `kaggle_outputs/INDEX.json` و `ARCHIVE_STATUS.json` ثبت می‌شود. وزن یا نتیجهٔ اصلیِ در دسترس‌نبوده به‌عنوان «دریافت‌شده» معرفی نمی‌شود.

# وضعیت آرشیو

فایل‌ها و وزن‌های موجود از فایل‌های دریافت‌نشده جدا شده‌اند. صفر بودن تعداد checkpoint برای روش عددی Master5 طبیعی است؛ برای بقیهٔ موارد، توضیح هر ردیف را ببینید.

| نوت‌بوک | کد | فایل وزن استخراج‌شده | فایل خروجی اصلی دریافت‌شده/فهرست‌شده |
|---|---|---:|---:|
| dense-duck-formation-20261104-211d7de258 | [notebook](../evidence/dense-duck-formation-20261104-211d7de258/dense-duck-formation-20261104-211d7de258.ipynb) | 1 | 0/27 |
| master-1 | [notebook](../evidence/master-1/master-1.ipynb) | 0 | 0/0؛ API: 404 |
| master-2 | [notebook](../evidence/master-2/master-2.ipynb) | 0 | 0/126 |
| master-5-2d-magnetic-notpinn | [notebook](../evidence/master-5-2d-magnetic-notpinn/master-5-2d-magnetic-notpinn.ipynb) | 0 | 0/0؛ API: 404 |
| master10-magnetic-function-deeponet | [notebook](../evidence/master10-magnetic-function-deeponet/master10-magnetic-function-deeponet.ipynb) | 1 | 0/64 |
| master3-2dsimplechannel-general-up-down-particle | [notebook](../evidence/master3-2dsimplechannel-general-up-down-particle/master3-2dsimplechannel-general-up-down-particle.ipynb) | 0 | 0/0؛ API: 404 |
| master3-clean-revised-version | [notebook](../evidence/master3-clean-revised-version/master3-clean-revised-version.ipynb) | 0 | 0/74 |
| master4 | [notebook](../evidence/master4/master4.ipynb) | 0 | 0/0؛ API: 404 |
| master6-validation-2d-magnetic-pinn | [notebook](../evidence/master6-validation-2d-magnetic-pinn/master6-validation-2d-magnetic-pinn.ipynb) | 0 | 0/0؛ API: 404 |
| master7-general-magnetic-1st-try | [notebook](../evidence/master7-general-magnetic-1st-try/master7-general-magnetic-1st-try.ipynb) | 1 | 0/0؛ API: 404 |
| master8-general-magnetic-particl-2ndtry | [notebook](../evidence/master8-general-magnetic-particl-2ndtry/master8-general-magnetic-particl-2ndtry.ipynb) | 3 | 0/0؛ API: 404 |
| master9 | [notebook](../evidence/master9/master9.ipynb) | 1 | 0/26 |
| master9-flexible-magnetic-path-controller | [notebook](../evidence/master9-flexible-magnetic-path-controller/master9-flexible-magnetic-path-controller.ipynb) | 3 | 0/30 |

## جزئیات وزن‌ها

فهرست کامل فایل‌ها و SHA-256 آن‌ها در [models/INDEX.json](../models/INDEX.json) است. فایل‌های `.pt` مطابق مدل و تعریف شبکهٔ همان نوت‌بوک بارگذاری شوند؛ اجزای نسخه‌های متفاوت را جای هم قرار ندهید.

- `master-1`: No trained tensor checkpoint embedded in the available source. Full training code is archived; original weights are not claimed recovered.
- `master-2`: No trained tensor checkpoint embedded in the available source. Full training code is archived; original weights are not claimed recovered.
- `master-5-2d-magnetic-notpinn`: No neural checkpoint needed: direct numerical method.
- `master3-2dsimplechannel-general-up-down-particle`: No trained tensor checkpoint embedded in the available source. Full training code is archived; original weights are not claimed recovered.
- `master3-clean-revised-version`: Original best_model.pt is pending CDN download; the frozen flow tensor used to independently replay its two cached duck cases is present in verification/original_frozen_checkpoint.pt.
- `master4`: No trained tensor checkpoint embedded in the available source. Full training code is archived; original weights are not claimed recovered.
- `master6-validation-2d-magnetic-pinn`: No trained tensor checkpoint embedded in the available source. Full training code is archived; original weights are not claimed recovered.

فایل‌های خروجی اصلی فقط پس از دانلود موفق، کنترل تعداد بایت و ثبت SHA-256، «دریافت‌شده» محسوب می‌شوند. URLهای موقت و کلید API در manifest ذخیره نمی‌شوند. اجرای دوبارهٔ notebook خروجی بازتولیدشده می‌سازد؛ آن خروجی جای فایل اصلیِ دریافت‌نشده از Kaggle برچسب نمی‌خورد.
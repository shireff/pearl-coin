# قائمة المهام لتحسين أداء التعدين - زيادة tmok/s إلى 1500

## نظرة عامة
هذا المستند يحتوي على جميع التحسينات المطلوبة لرفع أداء التعدين من القيمة الحالية المنخفضة إلى **1500 tmok/s**. تم تنظيم التحسينات في فئات لتسهيل التنفيذ والمتابعة.

---

## 1. تحسينات إطلاق النواة (Kernel Launch Optimizations)

### 1.1 التبديل إلى نواة التعدين المستمرة (Persistent Mining Kernel)

| العنصر | التفاصيل |
|--------|----------|
| **لماذا** | إطلاق `gpu_mine_batch` في كل جولة يسبب حملاً كبيراً. النواة المستمرة تبقى قيد التشغيل وتتجنب إعادة التهيئة لكل جولة. |
| **ماذا** | استبدال `gpu_mine_batch` بـ `persistent_mining_kernel` الذي يعمل بشكل مستمر ويقبل_jobs جديدة بدون إعادة إطلاق. |
| **الملفات** | `miner/miner_gpu.py`، `miner/pearl-gemm/csrc/mining/*` |

**التغييرات المطلوبة:**
- إنشاء `MiningGraphSession` مع persistent kernel
- تعديل دورة التعدين لاستخدام `launch_persistent_kernel()` بدلاً من `gpu_mine_batch()`
- إضافة queue للتعامل مع jobs الجديدة

---

### ✅ 1.2 زيادة حجم البلاط (P×Q combinations) — DONE

| العنصر | التفاصيل |
|--------|----------|
| **لماذا** | البلاط 64×64 الحالي يعالج عدد nonce محدود. 128×128 يضاعف المعالجة لكل إطلاق نواة. |
| **ماذا** | تغيير `--tile-m` و `--tile-n` الافتراضيين إلى 128، وتحديث `choose_safe_tile_size()` لدعم هذا الحجم. |
| **الملفات** | `miner/miner_gpu.py`، `miner/pearl-gemm/csrc/gemm/pearl_gemm_api.cpp` |

---

### ✅ 1.3 استخدام CUDA Graph لـ noise_gen — DONE

| العنصر | التفاصيل |
|--------|----------|
| **لماذا** | `noise_gen` مستقر ويمكن التقاطه في CUDA Graph. هذا يتيح تداخل noise_gen مع mining للتوازي. |
| **ماذا** | إنشاء CUDA Graph لـ noise_gen باستخدام `torch.cuda.graph()` وتحرير الـ graph بعد كل Round. |
| **الملفات** | `miner/miner_gpu.py` |

**التغييرات المطلوبة:**
```python
# مثال على الهيكل
self.noise_graph = torch.cuda.graph()
self.noise_graph.capture_begin()
noise_gen(...)  # kernel call
self.noise_graph.capture_end()
# في run_round:
self.noise_graph.replay()
```

---

## 2. تحسينات الوصول للذاكرة (Memory Access Optimizations)

### ✅ 2.1 استخدام نسخ الذاكرة غير المتزامنة — DONE غير المتزامنة (Async Memory Copies)

| العنصر | التفاصيل |
|--------|----------|
| **لماذا** | `torch.cuda.Stream` يسمح بنسخ البيانات بالتوازي مع حسابات CUDA. |
| **ماذا** | استخدام `torch.cuda.Stream` لـ H2D و D2H transfers. |
| **الملفات** | `miner/miner_gpu.py` |

**التغييرات المطلوبة:**
```python
self.copy_stream = torch.cuda.Stream()
with torch.cuda.stream(self.copy_stream):
    # نسخ البيانات هنا
```

---

### ✅ 2.2 تقليل الحمل من slice عمليات Python — DONE

| العنصر | التفاصيل |
|--------|----------|
| **لماذا** | `hashes_i64 <= self._target_buf.unsqueeze(0)` في حلقة يخلق tensors جديدة. |
| **ماذا** | استخدام `torch.ops` مباشرة أو نقل الحلقة إلى CUDA kernel. |
| **الملفات** | `miner/miner_gpu.py` |

---

### ✅ 2.3 توحيد الصفائف في CUDA memory — DONE

| العنصر | التفاصيل |
|--------|----------|
| **لماذا** | الصفائف غير المتجاورة (non-contiguous) تحتاج نسخ إضافية في CUDA. |
| **ماذا** | التأكد من أن جميع tensors المستخدمة في hot path متجاورة باستخدام `.contiguous()`. |
| **الملفات** | `miner/miner_gpu.py` |

---

## 3. تحسينات استخدام GPU (GPU Utilization Optimizations)

### ✅ 3.1 زيادة حجم الشبكة (Grid Size) للتوازي — DONE

| العنصر | التفاصيل |
|--------|----------|
| **لماذا** | الشبكة الحالية قد لا تستغل جميع SMs المتاحة. |
| **ماذا** | حساب grid_size optimal بناءً على عدد الـ SMs: `grid_size = num_sms * 4`. |
| **الملفات** | `miner/pearl-gemm/csrc/mining/gpu_mining_launch.cu` |

---

### ✅ 3.2 استخدام أحداث CUDA للتوقيت — DONE

| العنصر | التفاصيل |
|--------|----------|
| **لماذا** | `time.time()` يقيس وقت CPU وليس وقت GPU الحقيقي. |
| **ماذا** | استخدام `torch.cuda.Event()` لقياس وقت تنفيذ kernels بدقة. |
| **الملفات** | `miner/miner_gpu.py` |

**مثال:**
```python
start_event = torch.cuda.Event()
end_event = torch.cuda.Event()
start_event.record()
# kernel call
end_event.record()
end_event.synchronize()
elapsed = start_event.elapsed_time(end_event) / 1000
```

---

### ✅ 3.3 تحسين تداخل الحسابات مع الذاكرة — DONE

| العنصر | التفاصيل |
|--------|----------|
| **لماذا** | بدون تداخل، GPU ينتظر اكتمال النسخ قبل بدء الحسابات. |
| **ماذا** | استخدام `torch.cuda.stream()` لجدولة noise_gen و mining في streams مختلفة. |
| **الملفات** | `miner/miner_gpu.py` |

---

## 4. تقليل الحمل من Python (Python Overhead Reduction)

### 4.1 نقل winner check إلى CUDA kernel

| العنصر | التفاصيل |
|--------|----------|
| **لماذا** | التحقق من winners في Python يتطلب نسخ بيانات من GPU. |
| **ماذا** | إنشاء `gpu_check_winners()` kernel يبحث عن winners مباشرة على GPU. |
| **الملفات** | `miner/pearl-gemm/csrc/mining/*` |

---

### ✅ 4.2 تجميع عدة jobs معاً — DONE

| العنصر | التفاصيل |
|--------|----------|
| **لماذا** | معالجة job واحدة في كل round غير فعالة. |
| **ماذا** | تجميع 4-8 jobs في batch واحد واستخدام vectorized operations. |
| **الملفات** | `miner/miner_gpu.py` |

**التغييرات المطلوبة:**
- تغيير `num_jobs` من 1 إلى 8 في `MiningGraphSession`
- تحديث `gpu_mine_batch()` للتعامل مع batch من jobs

---

### 4.3 تخزين job metadata في GPU constant memory

| العنصر | التفاصيل |
|--------|----------|
| **لماذا** | Job data يتم نسخه لكل round. |
| **ماذا** | استخدام `torch.cuda.constant_memory()` لـ job metadata الثابتة. |
| **الملفات** | `miner/miner_gpu.py` |

---

## 5. تحسينات الخوارزمية (Algorithm Optimizations)

### ✅ 5.1 استخدام قيم P و Q أكبر — DONE

| العنصر | التفاصيل |
|--------|----------|
| **لماذا** | P×Q يحدد عدد الـ combinations لكل job. |
| **ماذا** | تغيير `P=8, Q=8` بدلاً من `P=1, Q=1` لزيادة combinations إلى 64 ضعف. |
| **الملفات** | `miner/miner_gpu.py` |

**ملاحظة:** يتطلب زيادة `a_rows_data` و `b_cols_data` للتعامل مع partitions الجديدة.

---

### 5.2 تحسين jackpot hash kernel

| العنصر | التفاصيل |
|--------|----------|
| **لماذا** | `gpu_jackpot_hash` يحسب hashes لكل jackpots حتى لو لا نحتاجها. |
| **ماذا** | دمج winner check ��ي mining kernel مباشرة بدون المرور بـ jackpot_hash. |
| **الملفات** | `miner/pearl-gemm/csrc/mining/gpu_mining_kernels.cu` |

---

### ✅ 5.3 تحسين shared memory usage — DONE

| العنصر | التفاصيل |
|--------|----------|
| **لماذا** | حساب shared memory الحالي يستخدم padding قد يكون غير ضروري. |
| **ماذا** | تحسين `estimate_shared_mem_bytes()` لاستخدام minimal padding. |
| **الملفات** | `miner/miner_gpu.py` |

---

### 5.4 تفعيل Shared Memory Bank Conflicts Avoidance

| العنصر | التفاصيل |
|--------|----------|
| **لماذا** | Bank conflicts تقلل throughput بشكل كبير. |
| **ماذا** | استخدام `__builtin_amdgcn_interp_sz()` أو `ldg()` للبيانات للقراءة فقط. |
| **الملفات** | `miner/pearl-gemm/csrc/mining/gpu_mining_kernels.cu` |

---

## ملخص التحسينات والأولوية

| الأولوية | التحسين | التأثير المتوقع | الجهد |
|----------|---------|-----------------|-------|
| **عالية** | التبديل إلى persistent kernel | +40-60% | متوسط |
| **عالية** | زيادة tile size إلى 128×128 | +50-100% | منخفض |
| **متوسطة** | استخدام CUDA Graph لـ noise_gen | +15-25% | متوسط |
| **متوسطة** | زيادة P و Q إلى 8×8 | +50-100% | متوسط |
| **متوسطة** | Async memory copies | +10-15% | منخفض |
| **منخفضة** | CUDA events للتوقيت | +5% (دقة) | منخفض |
| **منخفضة** | نقل winner check لـ GPU | +10-20% | عالي |

---

## تقدير النتائج

| الحالة الحالية | الحالة المستهدفة |
|----------------|------------------|
| ~50-100 tmok/s | 1500+ tmok/s |

---

## ملاحظات التنفيذ

1. **ابدأ بـ** التحسينات منخفضة الجهد وعالية التأثير (tile size, async copies)
2. **اختبر بعد كل تغيير** لضمان عدم كسر الوظائف الحالية
3. **راقب استهلاك الذاكرة** مع زيادة tile size و P×Q
4. **استخدم `nvidia-smi`** لمراقبة GPU utilization أثناء الاختبار
---

## Kryptex Pool Support

Added a Kryptex Stratum V1 proxy in \proxy/kryptex/\ that bridges Kryptex Pool miners to the Pearl node JSON-RPC interface.

### Files Added

| File | Description |
|------|-------------|
| \proxy/kryptex/go.mod\ | Go module definition |
| \proxy/kryptex/config.go\ | Proxy configuration with environment variables |
| \proxy/kryptex/client.go\ | JSON-RPC client for Pearl node communication |
| \proxy/kryptex/stratum.go\ | Stratum V1 protocol implementation |
| \proxy/kryptex/main.go\ | Proxy server (listener, health endpoint, shutdown) |
| \proxy/kryptex/kryptex_config.go\ | Kryptex Pool endpoint definitions for PRL, BTC, LTC, ETHW, RVN |
| \proxy/kryptex/tests/stratum_test.go\ | Stratum client unit tests |
| \proxy/kryptex/tests/config_test.go\ | Config and coin config unit tests |
| \proxy/kryptex/Dockerfile\ | Docker image for the Kryptex proxy |
| \proxy/kryptex/README.md\ | Kryptex proxy documentation |
| \proxy/docker-compose.kryptex.yml\ | Docker Compose for Kryptex proxy + Pearl node |
| \instance.env\ | Added Kryptex proxy env vars (STRATUM_PORT, STRATUM_TLS, etc.) |

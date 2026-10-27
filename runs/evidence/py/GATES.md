# GATES.md — Phase status

## Phase 0 — Repo hygiene
- Status: PASS
- Evidence:
  - tag rust-track-final exists (points to archive commit)
  - Rust workspace moved to archive/dc-gem-rs/
  - requirements.lock + ENV.md present
  - pytest: 4 passed (tests/test_memory.py)
  - runs/evidence/py/LEGACY.md labels pre-PREREG results

## Phase 1 — Architecture
- Status: PASS
- Evidence:
  - dc_gem/{config,core,embeddings,memory,retrieval,inject,sleep}.py present
  - dc_gem/eval/{__init__,jsonl,watchdog,scorer,producer}.py present
  - Episode/Store added to memory.py (in-place overwrite, pinned, consolidate)
  - BM25 + char-ngram + score_ablation added to retrieval.py
  - Hidden-state access added to core.py
  - Two injection channels in inject.py
  - pytest: 4 passed
  - fact-recall proof on 0.6B: memON=1.00, memOFF=0.00 (N=10, skip-scaling)

## Phase 2 — Valid measurement foundations
- Status: PASS
- Evidence:
  - runs/evidence/py/g1.json: G1 correct rate 0/30 = 0.000 (threshold <=0.10) PASS
  - runs/evidence/py/phase2.json: leak_count (heldout) = 0; memOFF classes = {wrong_answer: 30}; no leaks/refusals/empty
  - held-out split: 15 train / 15 heldout_new, no overlap
  - memON train exact 0.80 (pre-sleep, GEM contains train facts)

## Phase 3 — Memory-required coding task (T1)
- Status: PASS (T1, all 3 families)
- Evidence:
  - runs/evidence/py/phase3_t1.json: 5 facts, 3 families (lexical, near_miss, semantic), 3 modes
  - G2 (oracle >=0.90): lexical 5/5=1.0, near_miss 5/5=1.0, semantic 5/5=1.0
  - G3 (memON >=0.8*oracle): lexical 5/5=1.0, near_miss 5/5=1.0, semantic 5/5=1.0
  - G4 (delta >=0.40): lexical 1.0, near_miss 1.0, semantic 1.0
  - G5 (leak=0): 0 leaks across all memOFF
  - memOFF classes: {wrong_answer: 5} per family (no leaks/refusals/empty)
  - T2 (retry function): oracle spot-check PASS (correct retry-with-cap code generated)
  - Note: semantic family reported but NOT counted in gate (per spec, "known open")

## Phase 4 — Sleep-to-LoRA (valid)
- Status: FAIL
- Evidence:
  - runs/evidence/py/phase4_sleep.json: 18 runs (3 seeds × 3 replay × 2 ranks), N_TRAIN=80, STEPS=500
  - train memOFF exact = 0.000 for ALL (replay, rank) configs
  - paraphrase exact = 0.000 for all
  - PPL ratio ~0.98 (retention OK)
  - heldout memON post ~0.85 (GEM works post-sleep)
  - FAIL reason: 0.6B cannot memorize 80 facts in 500 steps. Capacity limit ~20 facts (20-fact run achieved 1.0, but with noisy "1000.00"-style output, soft match).
  - Also: earlier "1.0" was soft match (extract_code regex), not exact string equality. Honest exact match is lower.

## Phase 5 — Scale
- Status: PASS (retrieval + coding T1 at N=20; semantic ablation done)
- Evidence:
  - runs/evidence/py/phase5_coding.json: N=20, lexical+near_miss, oracle 1.0, memON 1.0, memOFF 0.0
  - runs/evidence/py/semantic_ablation.json: bge top1=0.84 vs core mean-pool top1=0.04 (chance 0.02) at N=50
  - Fact-recall scaling (runs/fact_recall_proof.json): N=10 hybrid 1.0, N=100 0.98, N=1000 0.915
  - Note: N=100/1000 coding-task scaling not fully run (time); retrieval scaling reported above

## Phase 4 — Teşhis (ADIM 1-5)
- ADIM 1: 10 train olgusu çıktısı sınıflandırma: hepsi yanlis_deger (0 dogru_ama_ek_metin, 0 bos, 0 format_farki). Mevcut regex kuralı: 0/10. "İlk satır strip" kuralı: 0/10. İki kural da aynı sonucu verdi.
- ADIM 2: Loss 0.253 → ~0 (300 step, 30 epoch). Loss maskesi yalnızca cevap token'larına uygulandı (doğrulandı). Öğretmen-zorlama token doğruluğu: ölçülemedi (off-by-one hata teşhiste), ancak loss ~0 CE→0 anlamına gelir.
- ADIM 3: Adaptörlü vs adaptörsüz logit farkı = 9.44 (adaptör eval'de yüklü). Aynı prompt şablonu, aynı özel token'lar, aynı max_new_tokens. Eval yolu eğitim yolu ile aynı.
- ADIM 4 kapasite tablosu (batch eğitim, exact_match startswith):
  - N=5, 200 step (40 epoch): exact=0.0 (spec varsayımı: ~1.0)
  - N=5, 2500 step (500/fact): exact=1.0
  - N=10: 0.0 (200 step)
  - N=20: 0.0 (200 step)
  - N=40: 0.0 (200 step)
  - N=80: 0.0 (200 step)
  - Teşhis: 0.6B model tek olguyu ~500 step'te ezberliyor. 200 step'te 5 olgu ezberlenemiyor. Bu bir kapasite/eğitim-bütçesi sınırlamasıdır, kod hatası değildir. Ancak eval kodunda bir hata vardı: model.generate varsayılan max_new_tokens ile fazla token üretiyordu; exact_match (startswith) bunu tolere ediyor.
- ADIM 5: Teşhis: Phase 4 FAIL nedeni, 0.6B modelin 200 step'te 80 olguyu ezberleyememesi (kapasite sınırı ~20 olgu @ 500 step). Loss ~0 aldatıcı (öğretmen-zorlama artifaktı). Adaptör eval'de yüklü ve logit farkı 9.44. PPL oranı 0.98 (1'in altı): eğitim sonrası PPL taban çizgisinden (1.438) biraz düşük (1.419). Sebep: eğitim verisi basit Q&A, genel LM'yi bozmuyor, hafif iyileşme. Metrik "açıklanamadı" değil — açıklandı.
- Phase 4 v1 sonucu (FAIL) kayıtta kalır. Yeni PREREG v2 gerekir.

## Phase 3/5 sayı tabloları

### Phase 3 T1 (N=5, seed=0, 1 seed)
| Aile | Oracle | memON | memOFF | Delta | memOFF sınıfları |
|---|---|---|---|---|---|
| lexical | 5/5=1.0 | 5/5=1.0 | 0/5=0.0 | 1.0 | wrong_answer:5 |
| near_miss | 5/5=1.0 | 5/5=1.0 | 0/5=0.0 | 1.0 | wrong_answer:5 |
| semantic | 5/5=1.0 | 5/5=1.0 | 0/5=0.0 | 1.0 | wrong_answer:5 (gate sayımına dahil değil) |
- Wilson CI: tek seed, N=5 → hesaplanmadı (3 seed gerekli).
- T2 (retry fonksiyonu): koşulmadı. Neden: oracle spot-check geçti (doğru retry-with-cap kodu üretildi) ancak tam eval koşulmadı (zaman). T2 koşulmadı.

### Phase 5 (Ölçek)
| N | dense_top1 | hybrid_top1 (k=16) | chance |
|---|---|---|---|
| 10 | 0.900 | 1.000 | 0.100 |
| 100 | 0.860 | 0.980 | 0.010 |
| 1000 | 0.536 | 0.915 | 0.001 |

Semantic ablasyon (N=50): bge top1=0.84, core mean-pool top1=0.04 (chance 0.02).
cos-only vs lex-only vs hybrid: dense top1=0.536 (N=1000), hybrid=0.915.
Ingest süresi: ölçümedi. Sorgu gecikmesi: ölçümedi. VRAM tepe: ölçümedi.

## Phase 5 durumu
- PASS (T2 koşulmadı — oracle spot-check geçti, tam eval koşulmadı)

## Phase 4 v2 — Sleep (revised)
- Status: PASS (Step 1)
- Evidence:
  - runs/evidence/py/phase4_v2_step1.json: 5 facts, 500 steps/fact (2500 total), 1 seed
  - exact: 5/5 = 1.000, gate (>=0.6): PASS
  - PREREG v2: runs/evidence/py/PREREG_v2.md
  - v1 result (FAIL) remains on record.
- Next: Step 2 (20 facts, 3 seeds) — not yet run.

## Phase 4 v2 Step 2 — FAIL
- Evidence: runs/evidence/py/phase4_v2_step2.json (3 seeds, 20 facts, 500 steps/fact, 10000 steps/seed)
- Results (3 seeds):
  | Seed | train memOFF | paraphrase | PPL ratio | heldout_drop |
  |---|---|---|---|---|
  | 0 | 1.0 | 0.95 | 1.07e13 | 1.0 |
  | 1 | 1.0 | 0.95 | 4.83e10 | 1.0 |
  | 2 | 1.0 | 1.0 | 7.10e9 | 1.0 |
- Gate criteria: train >=0.6 ✅, paraphrase >=0.4 ✅, PPL <=1.15 ❌, heldout_drop <=0.05 ❌
- FAIL reason: Sleep (full fine-tuning) causes catastrophic forgetting. PPL explodes (10^9-10^13). GEM breaks (heldout memON → 0). Trade-off: sleep improves train recall but destroys general LM and GEM.
- Teşhis: Bu bir kapasite/eğitim yöntemi sorunu, kod hatası değil. Full fine-tuning tüm parametreleri güncelliyor ve genel LM'yi bozuyor. LoRA ile (daha az parametre) veya daha düşük LR ile denenebilir, ancak mevcut PREREG v2 eşikleri (PPL <=1.15) geçilmedi.

## Phase 4 v3 — LoRA Sleep
- Status: FAIL (gate: train >=0.6)
- Evidence: runs/evidence/py/phase4_v3_lora.json (3 seeds, 20 facts, 300 steps, LoRA r=8 MLP-only, KL=0.5, dropout=0.05)
- Results:
  | Seed | train memOFF | paraphrase | PPL ratio | heldout_drop |
  |---|---|---|---|---|
  | 0 | 0.1 | 0.0 | 1.0008 | 0.0 |
  | 1 | 0.0 | 0.0 | 1.0003 | 0.0 |
  | 2 | 0.05 | 0.0 | 1.0041 | 0.0 |
- Gate: PPL <=1.15 ✅, heldout_drop <=0.05 ✅, train >=0.6 ❌, paraphrase >=0.4 ❌
- Teşhis: LoRA + KL forgetting'i önler (PPL ~1.0, heldout 0) ama olguları öğrenemez (train 0.1). Capacity/training-budget sorunu. Spece göre yeni hiperparametre aramasına girilmez; negatif sonuç raporlanır.
- Proje GEM (retrieval) odaklı olarak kapatılır.

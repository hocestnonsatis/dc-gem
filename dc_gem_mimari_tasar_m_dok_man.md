# DC-GEM: Differentiable Core with Gradient-Free Episodic Memory
**Teknik Mimari ve Sistem Tasarım Dokümanı**

## 1. Genel Bakış ve Motivasyon
**DC-GEM (Differentiable Core with Gradient-Free Episodic Memory)**, insan beynindeki **Tamamlayıcı Öğrenme Sistemleri (Complementary Learning Systems - CLS)** teorisinden ilham alan hibrit bir derin öğrenme mimarisidir. Bu tasarım, standart derin sinir ağlarının iki temel darboğazını çözmeyi hedefler:
1. **Felaket Unutma (Catastrophic Forgetting):** Yeni verilerle güncellenen ağırlıkların geçmişte öğrenilen temsilleri bozması.
2. **Çıkarım Anında Statiklik (Inference-Time Rigidity):** Modelin çalışma zamanında (deployment/inference) geri yayılım (backpropagation) yapmaksızın yeni veya nadir olaylara anında adapte olamaması.

DC-GEM bu sorunları, yavaş öğrenen ve yüksek genelleme kapasitesine sahip **Türevlenebilir Çekirdek (Differentiable Core - Neokorteks analoğu)** ile tek seferde (one-shot), $O(1)$ veya $O(\log N)$ karmaşıklıkta gradyansız kayıt yapabilen **Epizodik Bellek (Gradient-Free Episodic Memory - Hipokampüs analoğu)** modüllerini ayırarak çözer.

---

## 2. Sistem Topolojisi ve Veri Akışı

```
                        +---------------------------------------+
                        |           Girdi Dizisi (x_t)          |
                        +-------------------+-------------------+
                                            |
                                            v
                        +---------------------------------------+
                        |   Türevlenebilir Çekirdek (Alt Kat.)  |
                        |          h_t = f_theta(x_<=t)         |
                        +---------+-------------------+---------+
                                  |                   |
                  Gizli Temsil    |                   | Sorgu Projeksiyonu
                     (h_t)        |                   | q_t = LN(W_q * h_t)
                                  |                   v
                                  |     +-------------------------------+
                                  |     |   Gradyansız Epizodik Bellek  |
                                  |     |     M = {(k_i, v_i, u_i)}     |
                                  |     +------+-----------------+------+
                                  |            |                 ^
                                  |   Top-K    |                 | Sürpriz Tabanlı
                                  |   Okuma    |                 | Gradyansız Yazma
                                  |   (m_t)    |                 | (EMA / Eviction)
                                  v            v                 |
                        +----------------------------------+     |
                        |  Adaptif Kapılama (Gated Fusion) |     |
                        |  h'_t = (1 - g_t)h_t + g_t*m_t   |     |
                        +-----------------+----------------+     |
                                          |                      |
                                          v                      |
                        +----------------------------------+     |
                        |   Çıktı Katmanı & Sürpriz Ölçer  |-----+
                        |       P(y_t | h'_t), H(y_t)      |
                        +----------------------------------+
```

### Temel Bileşenlerin Rolleri

| Bileşen | Sorumluluk | Güncelleme Mekanizması | Zaman Ölçeği |
| :--- | :--- | :--- | :--- |
| **Differentiable Core (DC)** | Semantik çıkarım, örüntü tanıma ve gizli temsil (latent representation) üretimi. | Geri yayılım (Backpropagation) / Ön-eğitimde veya Uyku Fazında (Sleep Replay) güncellenir. | Yavaş (Uzun vadeli plastiklik) |
| **Gradient-Free Episodic Memory (GEM)** | Nadir olayları, anlık deneyimleri ve bağlamsal durumları saklama. | Türevsiz yazma/silme kuralları (k-NN, EMA birleştirme ve Sürpriz tabanlı kayıt). | Anlık (Çıkarım sırasında gerçek zamanlı) |
| **Gated Fusion Arayüzü** | Çekirdek tahmini ile bellekten gelen bağlamı dinamik olarak harmanlama. | Belirsizlik (entropi) ve kosinüs benzerliğine dayalı adaptif kapılama. | Her ileri besleme (Forward pass) adımında |

---

## 3. Matematiksel Formülasyon (Forward Pass)

Sistem $t$ adımında bir $x_t$ girdisi aldığında ileri besleme akışı aşağıdaki denklemlerle yürütülür:

### 3.1. Gizli Temsil ve Sorgu Üretimi
Türevlenebilir Çekirdek (Transformer, Mamba/SSM veya hibrit bir omurga), girdiyi işleyerek $L$-inci ara katmanda bağlamsal bir gizli vektör $h_t \in \mathbb{R}^d$ ve bellek sorgusu için $\ell_2$-normalize edilmiş bir anahtar vektör $q_t \in \mathbb{R}^{d_k}$ üretir:

$$h_t = f_{\theta}(x_{\le t})$$

$$q_t = \frac{\text{LN}(W_q h_t)}{\|\text{LN}(W_q h_t)\|_2}$$

Burada $W_q \in \mathbb{R}^{d_k \times d}$ öğrenilebilir sorgu projeksiyon matrisini, $\text{LN}$ ise Layer Normalization işlemini temsil eder.

### 3.2. Gradyansız Epizodik Okuma (Non-Parametric Retrieval)
Epizodik bellek, $N$ kapasiteli bir tablodur:

$$\mathcal{M} = \{(k_i, v_i, u_i)\}_{i=1}^{N}, \quad k_i \in \mathbb{R}^{d_k}, \; v_i \in \mathbb{R}^{d_v}, \; u_i \in \mathbb{R}^+$$

Burada $k_i$ normalize edilmiş anahtar vektörünü, $v_i$ saklanan değer (bağlam/hedef) vektörünü, $u_i$ ise kaydın kullanım istatistiğini (utility) gösterir. Sorgu vektörü $q_t$ ile bellekteki en yakın $K$ komşu (Approximate Nearest Neighbors - HNSW veya Product Quantization ile) bulunur:

$$s_i = q_t^\top k_i, \quad \mathcal{N}_K(q_t) = \text{Top-}K_{i \in \{1 \dots N\}} (s_i)$$

Okunan bellek vektörü $m_t$, bulunan $K$ komşunun sıcaklık ($\tau$) parametresiyle ölçeklenmiş dikkat (attention) ağırlıklı toplamıdır:

$$\alpha_i = \frac{\exp(s_i / \tau)}{\sum_{j \in \mathcal{N}_K(q_t)} \exp(s_j / \tau)}, \quad m_t = \sum_{i \in \mathcal{N}_K(q_t)} \alpha_i v_i$$

> **Kritik Kural:** Çıkarım ve eğitim sırasında $\mathcal{M}$ içindeki $(k_i, v_i)$ tensörleri üzerinde `stop_gradient` ($\text{sg}[\cdot]$) uygulanır; bellek içeriklerine asla geri yayılım yapılmaz.

### 3.3. Adaptif Kapılama ve Entegrasyon (Gated Memory Fusion)
Çekirdeğin içsel temsili ($h_t$) ile epizodik bellekten okunan bağlam ($m_t$), dinamik bir kapı mekanizması ($g_t \in [0, 1]^d$) ile birleştirilir. Kapı katsayısı; çekirdek temsiline, okunan bellek vektörüne ve en yakın komşunun benzerlik skoruna ($s_{\max} = \max_{i \in \mathcal{N}_K(q_t)} s_i$) bağlıdır:

$$g_t = \sigma\left(W_g \left[h_t \,;\, W_m m_t \,;\, s_{\max}\right] + b_g\right)$$

$$h'_t = (1 - g_t) \odot h_t + g_t \odot \text{LN}(W_m m_t)$$

Elde edilen zenginleştirilmiş temsil $h'_t$, ağın kalan üst katmanlarına veya doğrudan çıktı başlığına (LM Head / Policy Head) aktarılır.

---

## 4. Gradyansız Yazma ve Bellek Yönetimi (Memory Dynamics)

DC-GEM, çıkarım (inference) esnasında ağırlıkları değiştirmeden öğrenmek için üç aşamalı bir bellek yönetim protokolü uygular:

### 4.1. Sürpriz ve Yenilik Tabanlı Yazma Kriteri (Surprise-Driven Write)
Belleğin gereksiz ve sıradan verilerle dolmasını önlemek için yalnızca **yüksek sürpriz** (beklenmedik çıktı/yüksek belirsizlik) veya **yüksek yenilik** (bellekte karşılığı olmayan durum) içeren deneyimler kaydedilir.

Çıktı olasılık dağılımının Shannon entropisi $\mathcal{H}(P(y_t | h'_t))$ (veya hedef biliniyorsa anlık negatif log-olabilirlik $\mathcal{L}_t = -\log P(y_t^* | h'_t)$) hesaplanır:

$$\mathcal{H}(y_t) = -\sum_{c \in \mathcal{V}} P(y_t = c \mid h'_t) \log P(y_t = c \mid h'_t)$$

Yazma tetikleme koşulu ($\text{WriteTrigger}$):

$$\text{WriteTrigger}_t = \mathbb{I}\left[ \mathcal{H}(y_t) > \gamma_{\text{surprise}} \;\;\lor\;\; s_{\max} < \gamma_{\text{novelty}} \right]$$

### 4.2. Hebbian / EMA Güncellemesi ve Yeni Slot Tahsisi
$\text{WriteTrigger}_t = 1$ olduğunda, yazılacak değer vektörü $v_{\text{new}} = W_v h_t$ (veya gözlemlenen gerçek hedef/ödül gömmesi) oluşturulur. İki durum söz konusudur:

1. **Birleştirme (EMA Merge):** Eğer girdi bellekteki en yakın komşuya belirli bir $\gamma_{\text{merge}}$ eşiğinden daha yakınsa ($s_{\max} \ge \gamma_{\text{merge}}$), yeni bir satır açmak yerine mevcut $i^* = \arg\max_i s_i$ kaydı üstel hareketli ortalama (EMA) ile yerinde güncellenir:
   $$k_{i^*} \leftarrow \frac{\beta k_{i^*} + (1 - \beta) q_t}{\|\beta k_{i^*} + (1 - \beta) q_t\|_2}, \quad v_{i^*} \leftarrow \beta v_{i^*} + (1 - \beta) v_{\text{new}}$$

2. **Yeni Kayıt (Slot Allocation):** Eğer $s_{\max} < \gamma_{\text{merge}}$ ise, girdi tamamen yeni bir kavramdır. Bellekteki boş bir slota veya tahliye edilen ($i_{\text{evict}}$) slota doğrudan yazılır:
   $$k_{i_{\text{evict}}} \leftarrow q_t, \quad v_{i_{\text{evict}}} \leftarrow v_{\text{new}}, \quad u_{i_{\text{evict}}} \leftarrow 1.0$$

### 4.3. Kullanım Tabanlı Unutma ve Tahliye (Utility-Based Eviction)
Bellek kapasitesi $N$ dolduğunda hangi kaydın silineceğini belirlemek için her adımda kullanım skorları ($u_i$) zamanla sönümlenir (decay) ve yalnızca o adımda okunan aktif komşuların skoru dikkat ağırlıkları ($\alpha_i$) oranında artırılır:

$$u_i^{(t)} = \lambda u_i^{(t-1)} + \alpha_i \cdot \mathbb{I}\left[i \in \mathcal{N}_K(q_t)\right], \quad \lambda \in (0, 1)$$

Tahliye edilecek indeks en düşük kullanım değerine sahip slottur:

$$i_{\text{evict}} = \arg\min_{i \in \{1 \dots N\}} u_i^{(t)}$$

---

## 5. Algoritmik Akış (Pseudocode)

```python
def dc_gem_step(x_t, memory, core_model, params, y_true=None):
    """
    DC-GEM tek adım ileri besleme ve gradyansız epizodik güncelleme döngüsü.
    """
    # 1. Türevlenebilir Çekirdekten Gizli Temsil ve Sorgu Üretimi
    h_t = core_model.encode(x_t)                          # [B, d]
    q_t = l2_normalize(layer_norm(h_t @ params.W_q))      # [B, d_k]

    # 2. Gradyansız Epizodik Okuma (Stop-Gradient)
    with no_grad():
        scores, idx = memory.search_top_k(q_t, k=params.K) # [B, K]
        s_max = scores.max(dim=-1, keepdim=True)           # [B, 1]

    # Dikkat ağırlıkları (Sorgu üzerinden türevlenebilir, bellek üzerinden türevsiz)
    alpha = softmax(scores / params.tau, dim=-1)           # [B, K]
    v_neighbors = stop_gradient(memory.values[idx])        # [B, K, d_v]
    m_t = sum(alpha.unsqueeze(-1) * v_neighbors, dim=1)    # [B, d_v]

    # 3. Adaptif Kapılama (Gated Fusion)
    m_proj = layer_norm(m_t @ params.W_m)                  # [B, d]
    gate_input = concat([h_t, m_proj, s_max], dim=-1)
    g_t = sigmoid(gate_input @ params.W_g + params.b_g)    # [B, d]
    
    h_prime = (1.0 - g_t) * h_t + g_t * m_proj             # [B, d]

    # 4. Çıktı Üretimi ve Sürpriz (Entropi) Hesabı
    logits = core_model.decode(h_prime)
    probs = softmax(logits, dim=-1)
    surprise = -sum(probs * log(probs + 1e-9), dim=-1)     # Shannon Entropisi

    # 5. Gradyansız Epizodik Yazma ve Güncelleme
    with no_grad():
        memory.decay_utility(decay_rate=params.lambda_decay)
        memory.boost_utility(idx, alpha)

        if surprise > params.gamma_surprise or s_max < params.gamma_novelty:
            v_new = encode_target(h_t, y_true, params.W_v)
            if s_max >= params.gamma_merge:
                # Mevcut yakın kaydı EMA ile güncelle
                best_idx = idx[:, 0]
                memory.ema_update(best_idx, q_t, v_new, beta=params.beta)
            else:
                # En düşük faydalı slotu bul ve üzerine yaz
                evict_idx = memory.get_lowest_utility_slot()
                memory.write_slot(evict_idx, q_t, v_new, init_utility=1.0)

    return logits, g_t, surprise
```

---

## 6. Eğitim ve Konsolidasyon Stratejisi (Co-Training & Sleep Phase)

Epizodik bellek çalışma zamanında gradyansız olsa da, **Türevlenebilir Çekirdek ($\theta$)** ve projeksiyon katmanlarının ($W_q, W_m, W_v, W_g$) belleği etkin kullanabilmesi için üç aşamalı bir eğitim protokolü uygulanır:

1. **Retrieval-Aware Pretraining (Bellek Farkındalıklı Ön-Eğitim):**
   Eğitim sırasında $q_t$ vektörü üzerinden $\alpha_i$ dikkat ağırlıklarına gradyan akmasına izin verilir ($\nabla_{W_q} \mathcal{L}$). Böylece model, doğru komşuları getirecek sorgu uzayını ($W_q$) ve okunan değerleri gizli uzaya taşıyacak projeksiyonu ($W_m$) uçtan uca öğrenir.
2. **Memory Dropout ve Gürültü Enjeksiyonu:**
   Çekirdek ağın tembelleşip yalnızca epizodik belleğe güvenmesini engellemek için eğitim adımlarının $p_{\text{drop}} \in [0.15, 0.30]$ oranında $g_t = 0$ olarak zorlanır (Memory Dropout). Ayrıca okunan $m_t$ vektörlerine küçük Gauss gürültüsü eklenerek kapılama mekanizmasının seçiciliği artırılır.
3. **Uyku Fazı Konsolidasyonu (Offline Sleep Consolidation):**
   Canlı çalışma (inference) sırasında epizodik bellekte biriken ve kullanım skoru ($u_i$) ile sürpriz değeri yüksek olan $(k_i, v_i)$ kayıtları, sistem boşta kaldığında bir **Deneyim Tekrarı (Replay Buffer)** veri kümesi olarak kullanılır. Çekirdek modelin düşük dereceli adaptörleri (örn. LoRA katmanları), bu seçilmiş epizodik kayıtlar üzerinde kısa süreli gradyan inişi ile güncellenir. Böylece hipokampüsteki (GEM) geçici anılar neokortekse (DC) kalıcı semantik bilgi olarak aktarılır.

---

## 7. Ölçeklenebilirlik ve Hiperparametre Önerileri

Büyük ölçekli (milyonlarca slot) dağıtımlarda doğrusal arama ($O(N)$) yerine **HNSW (Hierarchical Navigable Small World)** veya **IVF-PQ (Inverted File with Product Quantization)** indeksleme yapıları kullanılır.

| Hiperparametre | Sembol | Önerilen Değer Aralığı | Açıklama |
| :--- | :---: | :---: | :--- |
| **Bellek Kapasitesi** | $N$ | $65,536 - 1,048,576$ | Epizodik bellekteki toplam slot sayısı. |
| **Komşu Sayısı** | $K$ | $8 - 32$ | Her adımda okunan en yakın komşu sayısı. |
| **Anahtar Boyutu** | $d_k$ | $128 - 512$ | Kosinüs araması için sıkıştırılmış sorgu/anahtar boyutu. |
| **Okuma Sıcaklığı** | $\tau$ | $0.05 - 0.20$ | Softmax dikkat dağılımının keskinliğini belirler. |
| **Sürpriz Eşiği** | $\gamma_{\text{surprise}}$ | $\mu_{\mathcal{H}} + 1.5\sigma_{\mathcal{H}}$ | Dinamik entropi ortalamasının 1.5 standart sapma üstü. |
| **Birleştirme Eşiği** | $\gamma_{\text{merge}}$ | $0.85 - 0.92$ | Yeni slot açmak yerine EMA birleştirmesi yapma sınırı. |
| **EMA Katsayısı** | $\beta$ | $0.90 - 0.99$ | Mevcut kaydın korunma oranı. |
| **Kullanım Sönümleme** | $\lambda$ | $0.995 - 0.9999$ | Adım başına kullanım skoru (utility) erime katsayısı. |
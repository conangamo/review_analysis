# Week 1 Verification Guide (Terminal)

Muc tieu: xac minh nhiem vu **"Tuan 1: on dinh pipeline + chuan hoa config + test unit co ban"** da on chua.

Su dung PowerShell tai thu muc goc du an:

```powershell
cd F:\laptrinhPython\chuyenDe\product-review-analyzer
```

---

## 1) Kiem tra file thay doi co dung pham vi Week 1

Chay:

```powershell
git status --short
```

Ban nen thay (it nhat) cac file sau:

- `.env.example`
- `config/models.yaml`
- `scripts/run_analysis.py`
- `src/ai_engine/sentiment_analyzer.py`
- `src/ai_engine/__init__.py`
- `tests/conftest.py`
- `tests/test_aspect_manager.py`
- `tests/test_brand_extractor.py`
- `tests/test_sentiment_analyzer.py`
- `notebooks/week1_stabilization_report.ipynb`

Tieu chi PASS:
- Co day du cac file Week 1 o tren.

---

## 2) Kiem tra config analyzer da chuan hoa

### 2.1 Kiem tra config trong `models.yaml`

Chay:

```powershell
rg "sentiment_analyzer|confidence_threshold|min_confidence_tier1|min_confidence_tier2" config/models.yaml
```

Tieu chi PASS:
- Co section `sentiment_analyzer`.
- Co 3 key:
  - `confidence_threshold`
  - `min_confidence_tier1`
  - `min_confidence_tier2`

### 2.2 Kiem tra bien moi truong trong `.env.example`

Chay:

```powershell
rg "ANALYSIS_BATCH_SIZE|CHECKPOINT_DIR|USE_KEYWORD_FILTER|CONFIDENCE_THRESHOLD|MIN_CONFIDENCE_TIER1|MIN_CONFIDENCE_TIER2" .env.example
```

Tieu chi PASS:
- Co day du 6 bien moi truong tren.

---

## 3) Kiem tra pipeline analysis da bo hardcode va co resume that su

Chay:

```powershell
rg "confidence_threshold=0.3|--resume-batch|--checkpoint-name|resume_from=|checkpoint_name=" scripts/run_analysis.py
```

Tieu chi PASS:
- **Khong con** `confidence_threshold=0.3`.
- Co tham so moi:
  - `--resume-batch`
  - `--checkpoint-name`
- Co truyen vao `process_reviews(...)`:
  - `resume_from=...`
  - `checkpoint_name=...`

---

## 4) Chay unit test co ban

Chay:

```powershell
pytest tests -q
```

Tieu chi PASS:
- Ket qua xanh, du kien:
  - `8 passed`

Neu fail do chua cai pytest:

```powershell
pip install pytest
pytest tests -q
```

---

## 5) Kiem tra notebook bao cao Week 1

Chay:

```powershell
ls notebooks
```

Tieu chi PASS:
- Co file: `notebooks/week1_stabilization_report.ipynb`

---

## 6) Kiem tra nhanh lazy import de test khong bi phu thuoc model nang

Chay:

```powershell
rg "TYPE_CHECKING|ModuleNotFoundError|ZeroShotClassifier = None|from \\.models\\.zero_shot import ZeroShotClassifier" src/ai_engine/sentiment_analyzer.py src/ai_engine/__init__.py
```

Tieu chi PASS:
- Co lazy import / fallback cho `ZeroShotClassifier`.
- `pytest` van chay du khong tai model `transformers`.

---

## 7) Checklist ket luan Week 1

Danh dau:

- [ ] Unit test pass (`pytest tests -q`)
- [ ] Config analyzer da thong nhat (`models.yaml` + `.env.example`)
- [ ] `run_analysis.py` da ho tro resume ro rang
- [ ] Khong con hardcode threshold cu (`0.3`)
- [ ] Co notebook bao cao Week 1

Neu tat ca deu [x] => **Week 1 dat muc "on dinh va tot" cho do an**.

---

## Lenh chay nhanh toan bo verification

Ban co the copy nguyen cum sau:

```powershell
cd F:\laptrinhPython\chuyenDe\product-review-analyzer
git status --short
rg "sentiment_analyzer|confidence_threshold|min_confidence_tier1|min_confidence_tier2" config/models.yaml
rg "ANALYSIS_BATCH_SIZE|CHECKPOINT_DIR|USE_KEYWORD_FILTER|CONFIDENCE_THRESHOLD|MIN_CONFIDENCE_TIER1|MIN_CONFIDENCE_TIER2" .env.example
rg "confidence_threshold=0.3|--resume-batch|--checkpoint-name|resume_from=|checkpoint_name=" scripts/run_analysis.py
pytest tests -q
ls notebooks
```


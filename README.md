# Capstone-2

텍스트→오디오 검색(Text-to-Audio Retrieval) 실험을 재현하기 위한 저장소입니다. 이 레포에는 다음이 포함되어 있습니다:

- `MGA-CLAP/`: 검색 모델(MGA-CLAP, ASE 모델) 코드와 스크립트
- `artifacts/eval/Capstone/`: 이미 생성된 임베딩, 인덱스, 베이스라인 검색 결과, 재랭킹 결과, GT 등 평가 산출물

아래 가이드는 현재 포함된 파일을 기반으로, 환경 설정부터 베이스라인 검색 재현, 평가까지의 최소 경로를 제공합니다.

## 구성 요약

- `MGA-CLAP/pretrained/model.pt` — LFS로 추적되는 대용량 체크포인트(약 1.7GB)
- `MGA-CLAP/settings/inference_example.yaml` — 추론용 설정
- `MGA-CLAP/scripts/` — 파이프라인 스크립트
  - `MGA-CLAP/scripts/build_audio_embeds.py` — 오디오 임베딩 생성
  - `MGA-CLAP/scripts/build_index.py` — FAISS 인덱스 빌드(IVF-PQ/HNSW/Flat)
  - `MGA-CLAP/scripts/search_topk.py` — 인덱스를 이용한 Top-K 검색(베이스라인)
  - `MGA-CLAP/scripts/eval_retrieval.py` — 베이스라인/재랭킹 성능 평가
- `artifacts/eval/Capstone/`
  - `emb/` — 오디오 임베딩(`audio_embeddings.npz`, `audio_embeddings.parquet`, `embedding_spec.json`)
  - `index_flat/` — FAISS 인덱스(`flat.index`), 메타(`flat.json`), 매핑(`mapping.tsv`)
  - `baseline_k50_flat/` — 베이스라인 검색 결과(`mga_baseline.parquet`, `mga_baseline.jsonl`, `run_metadata.json`)
  - `rerank_*` — 재랭킹 결과(JSONL/Parquet)
  - `gt.json` — `query_id`→`audio_id` 정답 매핑
  - `final_results*.csv` — 일부 요약 결과

참고: `mapping.tsv`의 `audio_path`는 생성 당시 환경의 절대 경로를 포함할 수 있습니다. 평가는 `audio_id` 기준으로 이루어져 경로 불일치가 있어도 지표 산출에는 영향을 주지 않습니다.

## 사전 준비

1) Git LFS로 모델 파일 받기

```bash
git lfs install
git lfs pull
```

2) 파이썬/패키지 의존성(예시: CPU 기준)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install torch torchaudio numpy pandas pyarrow tqdm pyyaml faiss-cpu
```

- GPU 사용 시: PyTorch/FAISS 설치는 환경에 따라 다릅니다. PyTorch(쿠다) 설치 가이드를 먼저 따르고, `faiss-gpu`를 설치하세요.
- 스크립트는 임시 디렉토리 권한 문제를 피하기 위해 자동으로 `MGA-CLAP/.tmp`를 사용합니다.

## 베이스라인 검색 재현(Flat 인덱스, Top-K=50)

이미 포함된 인덱스/임베딩/질의 파일로 베이스라인 검색을 다시 실행할 수 있습니다.

```bash
cd MGA-CLAP
python3 scripts/search_topk.py \
  --config settings/inference_example.yaml \
  --checkpoint pretrained/model.pt \
  --queries ../artifacts/eval/Capstone/queries_caption1.jsonl \
  --caption-col caption --id-col query_id \
  --index ../artifacts/eval/Capstone/index_flat/flat.index \
  --index-meta ../artifacts/eval/Capstone/index_flat/flat.json \
  --mapping ../artifacts/eval/Capstone/index_flat/mapping.tsv \
  --topk 50 \
  --output-dir ../artifacts/eval/Capstone/baseline_k50_flat
```

출력:

- `artifacts/eval/Capstone/baseline_k50_flat/mga_baseline.parquet`
- `artifacts/eval/Capstone/baseline_k50_flat/mga_baseline.jsonl`
- `artifacts/eval/Capstone/baseline_k50_flat/run_metadata.json`

## 성능 평가

1) 베이스라인만 평가(동일 랭크 비교)

`eval_retrieval.py`는 기본적으로 재랭킹(`rerank_rank`)과 베이스라인(`baseline_rank`)을 비교합니다. 베이스라인 단독 결과(`rank`만 존재) 평가 시에는 두 랭크를 모두 `rank`로 지정하세요.

```bash
cd MGA-CLAP
python3 scripts/eval_retrieval.py \
  --results ../artifacts/eval/Capstone/baseline_k50_flat/mga_baseline.parquet \
  --ground-truth ../artifacts/eval/Capstone/gt.json \
  --rank-col rank --baseline-rank-col rank \
  --output-csv ../artifacts/eval/Capstone/baseline_eval_metrics.csv
```

2) 재랭킹 결과 평가(JSONL 직접 지원)

재랭킹 JSONL에는 `baseline_rank`, `rerank_rank`, `baseline_score`, `combined_score` 등이 포함되어 있어 기본 인자만으로 비교가 가능합니다.

```bash
cd MGA-CLAP
python3 scripts/eval_retrieval.py \
  --results ../artifacts/eval/Capstone/rerank_stepa2_flat_lambda1/reranked.jsonl \
  --ground-truth ../artifacts/eval/Capstone/gt.json \
  --output-csv ../artifacts/eval/Capstone/rerank_stepa2_flat_lambda1_metrics.csv
```

필요 시 `--k 1 5 10`, `--map-k 10`, `--ndcg-k 10`, `--bootstrap 1000` 등으로 리콜@K, mAP, nDCG, 부트스트랩 CI를 조정할 수 있습니다.

## (선택) 임베딩/인덱스 생성부터 재현

이미 `emb/`, `index_flat/`가 포함되어 있으나, 처음부터 생성하려면 아래를 참고하세요.

1) 오디오 임베딩 생성

```bash
cd MGA-CLAP
python3 scripts/build_audio_embeds.py \
  --config settings/inference_example.yaml \
  --checkpoint pretrained/model.pt \
  --manifest /path/to/manifest.csv \
  --audio-col audio_path --id-col file_name \
  --output-dir ../artifacts/eval/Capstone/emb
```

2) 인덱스 빌드(Flat 예시)

```bash
cd MGA-CLAP
python3 scripts/build_index.py \
  --embeddings ../artifacts/eval/Capstone/emb/audio_embeddings.npz \
  --embedding-key clip_embeddings \
  --index-type flat --metric ip --normalize \
  --output-index ../artifacts/eval/Capstone/index_flat/flat.index \
  --output-meta  ../artifacts/eval/Capstone/index_flat/flat.json \
  --output-mapping ../artifacts/eval/Capstone/index_flat/mapping.tsv
```

3) Top-K 검색 및 평가 — 위의 “베이스라인 검색 재현”, “성능 평가” 절 차례로 수행

## 주의사항

- 대소문자: 경로는 `artifacts/eval/Capstone`입니다(Capstone 대문자).
- LFS: 대용량 파일(`MGA-CLAP/pretrained/model.pt`)은 LFS로 추적됩니다. 클론 후 `git lfs pull` 필요.
- 경로 불일치: `index_flat/mapping.tsv`의 `audio_path`는 생성 환경 절대경로일 수 있으나, 평가는 `audio_id` 기준으로 동작합니다.
- FAISS GPU: GPU 검색을 원하면 `search_topk.py`에 `--use-gpu`를 추가하고 `faiss-gpu`가 설치되어 있어야 합니다.

## 라이선스/저작권

- 이 레포 내 타 OSS 코드(`MGA-CLAP`)의 라이선스는 해당 디렉토리의 공지를 따릅니다. 자체 산출물(`artifacts/*`)의 공유 범위는 프로젝트 정책에 맞춰 사용하세요.

## 문의

- 유지보수/연락처: 이주성/010-2055-9699

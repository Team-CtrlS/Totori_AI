# <img width="40" height="40" alt="White" src="https://github.com/user-attachments/assets/e18ada55-9880-445e-84a4-0bd955c9ab07" /> 토토리(Totori)

> 프로젝트 기간 | 2025.09.01 ~ ing
<img width="1628" height="892" alt="스크린샷 2026-06-19 오후 6 48 47" src="https://github.com/user-attachments/assets/f785a1b5-997c-41a3-a19d-247f33bdafbe" />


## Totori_AI

토토리는 난독 아동과 읽기에 어려움을 겪는 아동이 흥미를 잃지 않고 반복 학습할 수 있도록 관심사 기반 동화와 취약 음운 퀴즈를 제공하는 모바일 학습 서비스입니다. 낭독 음성을 분석해 생략·대치·삽입 오류와 읽기 유창성을 기록하고 다음 학습에 반영합니다.

`Totori_AI`는 토토리의 **AI 추론 및 언어 분석 서버**입니다. 관심사 음성을 텍스트로 변환하고, 읽기 수준에 맞는 동화·퀴즈를 생성하며, 발표 설계 기준으로 아동 낭독 분석에는 **Wav2Vec 2.0 CTC 전사와 Levenshtein Distance**를 사용합니다.

## ⚒️ 기술 스택

- Python 3.11
- FastAPI·Uvicorn
- OpenAI API
- Wav2Vec 2.0·CTC Decoder
- Whisper STT
- Levenshtein Distance
- PyTorch
- Redis
- KoNLPy·MeCab
- ElevenLabs API

## ⭐ 주요 기능

- Wav2Vec 2.0 기반 낭독 음성 전사
- Levenshtein Distance 기반 생략·대치·삽입 오류 분류
- 관심사 음성 STT
- 관심사 음성을 바탕으로 맞춤 동화 생성
- 읽기 수준별 어휘·문장 제약 적용
- 낭독 속도(WCPM)와 누적 오류의 Redis 임시 저장
- 취약 발음 기반 퀴즈 생성·채점
- 퀴즈 음성 TTS 생성

<p align="center">
<img width="1920" height="1080" alt="10 주요 기능" src="https://github.com/user-attachments/assets/8de2b2ae-832b-4afa-a6c3-f75e4f7ea905" />
<img width="1920" height="1080" alt="11 주요 기능" src="https://github.com/user-attachments/assets/942c517e-c511-45f0-8054-55ed1c77fe1e" />
<img width="1920" height="1080" alt="12 주요 기능" src="https://github.com/user-attachments/assets/17073971-2dfa-44bb-a38e-41da8ffd0c5d" />
<img width="1920" height="1080" alt="13 주요 기능" src="https://github.com/user-attachments/assets/1e72be1a-dba8-4158-bb8a-eda1d235685f" />
</p>

### 📺 시연 영상

[토토리 데모 영상 보기](https://youtu.be/WHZ7pYWFDvs)

## 🧠 담당 팀원

| 복지희 | 정유진 |
| :---: | :---: | 
| <img width="120" alt="복지희 GitHub 프로필" src="https://avatars.githubusercontent.com/u/129582481?v=4" /> | <img width="120" alt="정유진 GitHub 프로필" src="https://avatars.githubusercontent.com/u/127232686?v=4" /> |
| [`@jettieb`](https://github.com/jettieb) | [`@nomellc`](https://github.com/nomellc) |
| **AI Developer** | **AI Developer** |
|  | 음운 오류 패턴 분석 로직,<br>동화 낭독 음성 분석,<br>OpenAI API 기반 동화 생성,<br>퀴즈 생성 담당|

## 📁 프로젝트 구조

```text
Totori_AI/
├── app/
│   ├── api/
│   │   ├── stt_router.py         # STT API
│   │   ├── story_router.py       # 동화 생성 API
│   │   ├── reading_router.py     # 낭독 분석 API
│   │   └── quiz_router.py        # 퀴즈 API
│   ├── clients/                  # ElevenLabs 클라이언트
│   ├── common/                   # 단계별 상수와 한국어 사전
│   ├── core/                     # Redis 연결
│   ├── schemas/                  # Pydantic 요청·응답 모델
│   ├── services/                 # 생성·STT·발음 분석 로직
│   ├── utils/                    # 음성·정렬·프롬프트 유틸리티
│   ├── .env                      # 직접 생성, Git 추적 제외
│   └── main.py                   # FastAPI 진입점
├── .python-version
├── requirements.txt
└── README.md
```

## 🗺️ 전체 시스템 아키텍처
<img width="1920" height="1080" alt="18 전체 시스템 구조도" src="https://github.com/user-attachments/assets/11c641b8-311d-4838-bffb-8e1a73660474" />

## 실행 환경

- Python 3.11.9 권장
- Redis 7
- FFmpeg
- MeCab과 한국어 사전
- OpenAI API 키
- ElevenLabs API 키

음성 모델 실행을 위해 충분한 메모리와 저장 공간이 필요합니다. 관심사 입력에 사용하는 Whisper의 기본 모델은 `medium`이며 개발 환경에서는 더 작은 모델을 사용할 수 있습니다. 발표 기준 낭독 분석을 재현하려면 팀에서 학습한 Wav2Vec 2.0 체크포인트를 함께 준비해야 합니다.

## 설치 및 설정

### 1. 저장소 복제

```bash
git clone https://github.com/Team-CtrlS/Totori_AI.git
cd Totori_AI
```

### 2. 시스템 의존성 설치

macOS에서 Homebrew를 사용하는 경우:

```bash
brew install ffmpeg mecab mecab-ko-dic
```

다른 운영체제에서는 패키지 관리자에 맞게 FFmpeg, MeCab, `mecab-ko-dic`을 설치하세요.

### 3. 가상환경과 Python 패키지 설치

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Redis 실행

로컬 Redis를 사용하거나 Docker로 실행합니다.

```bash
docker run --name totori-redis \
  -p 6379:6379 -d redis:7-alpine
```

연결 확인:

```bash
redis-cli ping
# PONG
```

### 5. 환경변수 설정

`app/.env`를 생성합니다. 애플리케이션은 시작할 때 이 경로의 파일을 읽습니다.

```dotenv
OPENAI_API_KEY=your-openai-api-key

REDIS_HOST=localhost
REDIS_PORT=6379

ELEVENLABS_API_KEY=your-elevenlabs-api-key
ELEVENLABS_BASE_URL=https://api.elevenlabs.io/v1
ELEVENLABS_VOICE_ID=your-voice-id
ELEVENLABS_MODEL_ID=eleven_multilingual_v2

# 선택 사항: 기본값은 medium
WHISPER_MODEL=medium

# 설치 환경에 따라 경로를 변경하세요.
MECABRC=/opt/homebrew/etc/mecabrc
MECAB_DIC_PATH=/opt/homebrew/lib/mecab/dic/mecab-ko-dic
```

Intel Mac이나 Linux에서는 MeCab 설치 경로가 다를 수 있습니다. 다음 명령으로 실제 설정 파일과 사전 경로를 확인한 뒤 값을 수정하세요.

```bash
mecab-config --sysconfdir
mecab-config --dicdir
```

## 실행

저장소 루트에서 실행합니다.

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

정상 실행 확인:

```bash
curl http://localhost:8000/
# {"status":"ok"}
```

- API 문서: <http://localhost:8000/docs>
- OpenAPI JSON: <http://localhost:8000/openapi.json>

첫 실행 시 Whisper 모델을 내려받고 메모리에 로드하므로 시간이 오래 걸릴 수 있습니다. 빠른 개발 확인에는 아래처럼 작은 모델을 사용하세요.

```dotenv
WHISPER_MODEL=base
```

## API

| Method | 경로 | 설명 |
| --- | --- | --- |
| `GET` | `/` | 서버 상태 확인 |
| `POST` | `/ai/stt/transcribe` | 음성 파일 STT |
| `POST` | `/ai/story/make` | 관심사 음성 기반 동화 생성 |
| `POST` | `/ai/reading/analyze` | 페이지 낭독 분석 및 결과 누적 |
| `POST` | `/ai/reading/complete/{child_id}/{book_id}` | 누적 분석 결과 반환 및 삭제 |
| `POST` | `/ai/quiz/generate` | 취약점 기반 퀴즈 생성 |
| `POST` | `/ai/quiz/analyze` | 퀴즈 답변 음성 채점 |

## 🧠 AI 처리 흐름

### 맞춤형 동화 생성

```text
관심사 음성
  → STT
  → 아동 레벨·이전 오류 패턴 결합
  → LLM 동화 텍스트 및 이미지 프롬프트 생성
  → Spring Boot에서 이미지·TTS 생성
  → 동화 콘텐츠 저장
```

### 읽기 오류 패턴 분석

```text
아동 낭독 음성
  → Wav2Vec 2.0 특징 추출
  → CTC Decoder로 문맥 보정 없는 전사
  → 원문과 전사문을 Levenshtein Distance로 정렬
  → 생략·대치·삽입 오류와 WCPM 계산
  → Redis 임시 저장 후 학습 완료 시 백엔드 반환
```

언어모델이 문맥상 자연스러운 단어로 자동 교정하면 실제 발음 오류가 사라질 수 있습니다. 발표 설계에서는 언어모델 보정이 없는 CTC Decoder 결과를 사용해 아동이 발음한 형태를 최대한 보존합니다.

## 🧪 연구 데이터와 재현

### 입력·출력 데이터

| 종류 | 형식 | 용도 |
| --- | --- | --- |
| 관심사 음성 | WAV, M4A, MP3 | 동화 주제 추출 |
| 낭독 음성 | WAV, M4A, MP3 | 오류 유형과 WCPM 분석 |
| 원문 | UTF-8 문자열 | 전사 결과 비교 기준 |
| 아동 레벨 | `L1`~`L6` | 동화·퀴즈 난이도 조절 |
| 분석 결과 | JSON | 생략·대치·삽입, 조사 오류, 평균 WCPM |

실제 아동 음성은 개인정보와 연구윤리 문제로 공개 저장소에 커밋하지 않습니다. 제출 및 재현용 저장소에는 비식별 동의가 완료된 샘플 또는 직접 제작한 프로토 음성을 `samples/`에 별도로 제공해야 합니다.

권장 샘플 구조:

```text
samples/
├── audio/
│   ├── interest_sample.wav      # 관심사 입력 예시
│   └── reading_sample.wav       # 낭독 입력 예시
├── text/
│   └── reading_original.txt     # 낭독 원문
└── expected/
    └── reading_result.json      # 기대 분석 결과
```

### Wav2Vec 2.0 모델 재현 자원

Wav2Vec 2.0 기반 낭독 분석을 동일하게 재현하려면 아래 항목이 필요합니다.

- 아동 음성으로 학습한 모델 체크포인트 또는 다운로드 링크
- 전처리·학습·추론 스크립트
- 데이터셋 출처, 라이선스, 비식별화 방법
- 학습 환경과 하이퍼파라미터
- 샘플 음성과 기대 전사·오류 분석 결과

대용량 체크포인트는 Git LFS 또는 외부 스토리지에 보관하고 README에 다운로드 위치와 배치 경로를 명시해야 합니다. 공개할 수 없는 연구 데이터는 원본 대신 같은 형식의 프로토 데이터를 제공해야 합니다.

> **제출 전 확인:** 현재 공개 저장소의 실행 코드는 Whisper 기반 전사 모듈을 포함하고 있습니다. 발표자료와 동일한 Wav2Vec 2.0 결과를 재현하려면 Wav2Vec 추론 코드, 체크포인트, 샘플 입력과 기대 결과를 저장소에 추가해야 합니다. README의 설명만 변경해서는 재현 요건을 충족할 수 없습니다.

### 발표자료 기준 분석 방식

| 단계 | 방법 |
| --- | --- |
| 음성 전사 | 아동 음성 기반 Wav2Vec 2.0 + CTC Decoder |
| 문자열 정렬 | Levenshtein Distance |
| 오류 유형 | 생략·대치·삽입 |
| 읽기 유창성 | WCPM |
| 적응형 난이도 | 최근 3회 이동평균 기반 `L1`~`L6` |

## 코드 검사

Python 문법을 빠르게 확인하려면 다음 명령을 사용합니다.

```bash
python -m compileall app
```

## 🚀 문제 해결

#### 서버 시작 시 Redis 연결에 실패합니다

Redis가 실행 중인지, `REDIS_HOST`와 `REDIS_PORT`가 실제 주소와 같은지 확인하세요. 애플리케이션은 시작 단계에서 Redis에 연결합니다.

#### OpenAI API 키 오류가 발생합니다

환경변수 파일의 위치가 저장소 루트가 아니라 `app/.env`인지 확인하세요.

#### MeCab 사전을 찾지 못합니다

`MECABRC`와 `MECAB_DIC_PATH`를 현재 운영체제의 실제 설치 경로로 바꾸세요.

#### 첫 요청이나 서버 시작이 느립니다

Whisper 모델 다운로드 또는 로딩이 원인일 수 있습니다. `WHISPER_MODEL=base`나 `small`로 낮춰 확인하세요.

#### Wav2Vec 분석 결과를 재현할 수 없습니다

학습 체크포인트, 전처리 방식, CTC Decoder 설정이 같은지 확인하세요. 체크포인트가 저장소 외부에 있다면 다운로드 링크와 로컬 배치 경로가 README 또는 배포 문서에 명시되어 있어야 합니다.

#### TTS 결과가 비어 있습니다

`ELEVENLABS_API_KEY`, `ELEVENLABS_BASE_URL`, `ELEVENLABS_VOICE_ID`, `ELEVENLABS_MODEL_ID`를 모두 확인하세요.

## 📁 관련 저장소

- iOS 앱: [`Totori_FE`](https://github.com/Team-CtrlS/Totori_FE)
- 백엔드: [`Totori_BE`](https://github.com/Team-CtrlS/Totori_BE)

## 라이선스

현재 별도의 라이선스 파일이 없습니다.

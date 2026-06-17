<h1>AIproject : 비판적 재평가 텍스트를 통한 planner Model 성능 개선 듀얼 시스템 VLA 모델 개발</h1> 

> OpenVLA(Planner)와 SmolVLM 500M(Actor)을 결합한 듀얼 시스템 VLA(Vision-Language-Action) 모델.
> Planner가 생성한 action을 Actor가 **비판적으로 재평가**하여 조작 성능을 개선하는 구조를 제안하고 구현함.

**Critical text And Dual System Vision Language Action Model(CADS-VLA)**

<a href="https://pytorch.org/get-started/locally/">
  <img src="https://img.shields.io/badge/PYTORCH-Qwen%202.6.0%20cu124-brightgreen?style=flat-square&label=PYTORCH&labelColor=%23eeeeee&color=%23d63f3a" height="40"/>
</a>
&nbsp;
<a href="https://pytorch.org/get-started/locally/">
  <img src="https://img.shields.io/badge/PYTORCH-openvla%202.12.0%20%2Bcpu%20-brightgreen?style=flat-square&label=PYTORCH&labelColor=%23eeeeee&color=%23d63f3a" height="40"/>
</a>
&nbsp;
<a href="https://www.python.org/">
  <img src="https://img.shields.io/badge/Python-3.10-brightgreen?style=flat-square&label=Python&labelColor=%23eeeeee&color=%2355adf4"height="40"/>
</a>

## 소개 및 문제 정의

기존의 단일 VLA 모델은 시각 정보와 언어 명령을 입력받은 뒤 곧바로 action을 생성하는 구조를 가진다. 하지만 이러한 방식은 모델이 자신이 만든 action을 다시 확인하거나 수정하는 과정이 없다는 한계가 있다. 특히 여러 단계가 연속적으로 이어지는 long-horizon task에서는 초반의 작은 판단 오류가 뒤 단계까지 누적되어 전체 과제 실패로 이어질 가능성이 크다.

이러한 문제는 LIBERO 벤치마크 결과에서도 확인할 수 있다. Zhao et al.의 CoT-VLA 논문에서는 동일한 평가 조건인 **3 seeds × 500 episodes**에서 여러 VLA 모델의 task별 성공률을 비교하였다. 그 결과, 별도의 추론 또는 검토 단계가 없는 모델들은 Spatial, Object, Goal task에서는 비교적 높은 성능을 보였지만, Long suite에서는 성공률이 50%대까지 크게 감소하였다.

**LIBERO 벤치마크 — Task별 Success Rate**

| 모델 | 추론·검토 단계 | Spatial | Object | Goal | **Long** |
|------|:------------:|:-------:|:------:|:----:|:--------:|
| Diffusion Policy | 없음 | 78.3% | 92.5% | 68.3% | **50.5%** |
| Octo | 없음 | 78.9% | 85.7% | 84.6% | **51.1%** |
| OpenVLA-7B | 없음 | 84.7% | 88.4% | 79.2% | **53.7%** |
| **CoT-VLA-7B** | **있음** (visual CoT) | 87.5% | 91.6% | 87.6% | **69.0%** |

위 결과를 보면 Diffusion Policy, Octo, OpenVLA는 서로 다른 구조를 가진 모델임에도 불구하고, 공통적으로 action을 생성하기 전 스스로 판단을 점검하는 단계가 없다. 이 모델들은 Long task에서 모두 50%대의 성공률을 보였으며, 이는 긴 작업에서 오류가 누적되기 쉽다는 점을 보여준다. 반면 visual chain-of-thought 과정을 추가한 CoT-VLA는 Long task에서 **69.0%**의 성공률을 기록하였다. 이는 행동을 바로 실행하기보다, 중간에 판단을 검토하는 단계가 long-horizon task의 성능 저하를 완화하는 데 도움을 줄 수 있음을 의미한다.

> 출처: Zhao et al., “CoT-VLA: Visual Chain-of-Thought Reasoning for VLA Models”, arXiv:2503.22020, Table 1.  
> Diffusion Policy, Octo, OpenVLA, CoT-VLA는 모두 동일 논문에서 3 seeds × 500 episodes 조건으로 평가되었다.

---

따라서 본 프로젝트는 long-horizon task에서 발생하는 오류 누적 문제를 줄이기 위해, action 실행 이전에 한 번 더 판단을 검토하고 보정하는 구조를 설계하는 것을 목표로 한다. 기존 단일 VLA 모델이 입력으로부터 바로 action을 생성하는 방식이라면, 본 프로젝트는 그 사이에 별도의 검토 단계를 추가하여 action의 적절성을 다시 판단하도록 구성하였다.

이러한 접근은 지도학습 기반 모방학습의 한계와도 관련이 있다. 모방학습 모델은 학습 데이터의 행동 분포를 기반으로 action을 예측하지만, 실제 실행 과정에서 작은 오차가 발생하면 이후 상태가 학습 데이터의 분포에서 벗어날 수 있다. 이 경우 잘못된 action이 연속적으로 발생하며, 긴 시퀀스 전체의 성공률이 낮아질 수 있다. 따라서 중간에 action을 다시 점검하고 수정하는 과정은 오류 누적을 줄이는 데 중요한 역할을 할 수 있다.

본 프로젝트에서는 이러한 문제를 **역할 분리 기반 듀얼 시스템 구조**로 해결하고자 하였다.

- **Planner (OpenVLA)**  
  시각 정보와 언어 명령을 기반으로 1차 action을 생성한다.

- **Actor (SmolVLM 500M)**  
  Planner가 제안한 action을 그대로 실행하지 않고, 현재 상황에 적절한지 검토한 뒤 필요한 경우 수정된 action을 생성한다.

즉, 전체 구조는 사람이 행동하기 전에 “생각하고, 점검하고, 실행하는 과정”을 모델 구조 안에 반영하려는 시도이다. Planner는 빠르게 action을 제안하고, Actor는 그 action을 비판적으로 재평가하여 최종 행동을 보정하는 역할을 담당한다.

---

본 프로젝트에서는 듀얼 시스템 VLA 구조를 설계하고, Planner와 Actor가 연결되어 동작하는 전체 파이프라인을 구현하였다. 다만 컴퓨팅 자원과 학습 시간의 제약으로 인해 대규모 강화학습과 LIBERO-long 정량 평가는 향후 과제로 남겨두었다.

| 단계 | 진행 상태 |
|------|------|
| 듀얼 시스템 아키텍처 설계 | 완료 |
| Planner(OpenVLA) 추론 파이프라인 구현 | 완료 |
| Actor(SmolVLM 500M) 구성 | 완료 |
| Actor 토크나이저 확장 및 projection layer 구성 | 완료 |
| ZeroMQ 기반 Planner-Actor 통신 구조 | 완료 |
| **SFT를 통한 action token 출력 및 텍스트 생성 학습** | 일부 진행, 정성적 동작 확인 완료 |
| GRPO 강화학습 | 코드 구현 완료, 대규모 학습 미진행 |
| LIBERO-long 정량 평가 | 컴퓨팅 자원 제약으로 미진행 |

정리하면, 본 프로젝트는 단순히 새로운 구조를 제안하는 것에 그치지 않고, Planner가 action을 생성하고 Actor가 이를 다시 검토하는 듀얼 시스템 파이프라인을 실제로 구현한 데 의의가 있다. 또한 SFT 단계에서 Actor가 action token과 텍스트 출력을 함께 생성하는 방향으로 학습되는 것을 확인하였다. 최종적인 성능 향상 여부는 향후 대규모 GRPO 학습과 LIBERO-long 벤치마크 평가를 통해 추가적으로 검증할 필요가 있다.
---
**Planner** : OpenVLA 7B fine tuning + 4bit, Frozen, CPU inference 

**Actor 1** : Qwen2.5VL 3B + 4bit + LoRA + GRPO

**Actor 2** : SmolVLM 500M + 4bit + LoRA + GRPO <- **(used)**

**Simulator** : LIBERO (libero_spatial)


## Back Bone Model URL

**- OpenVLA** : [openvla](https://github.com/openvla/openvla.git)

**- Qwen2.5VL** : [Qwen2.5VL](https://github.com/huggingface/transformers/tree/main/src/transformers/models/qwen2_5_vl)

**- SmolVLM** : [SmolVLM](https://github.com/huggingface/smollm/tree/main)

**- LIBERO** : [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO.git)

## 🖼️아키텍쳐

![Figure1](https://github.com/user-attachments/assets/edbb073d-d40e-452a-873f-a4de54bc41b6)
- **1. Model Structure**

 ## 모델 전체 구조

본 프로젝트의 전체 구조는 Planner와 Actor를 분리한 듀얼 시스템 형태로 구성된다.  
Planner는 OpenVLA 모델을 사용하며, 학습하지 않고 **frozen 상태**로 유지한다. Actor의 backbone 모델로는 LVM 계열 중 비교적 가벼운 **SmolVLM 500M**을 사용하였다.

Planner는 시각 정보와 언어 명령을 바탕으로 1차 action token id를 생성한다. 이후 생성된 action token id는 Actor로 전달된다. Actor는 전달받은 token id를 OpenVLA의 action embedding table을 통해 embedding으로 변환한 뒤, projection layer를 거쳐 SmolVLM의 embedding 차원에 맞게 정렬한다.

동시에 image와 text 입력은 SmolVLM processor를 거쳐 embedding으로 변환된다. 이후 action embedding과 image-text embedding을 concat하여 Actor 모델에 입력한다. Actor는 이를 기반으로 Planner의 action을 검토하고, 필요한 경우 수정된 action과 critique text를 함께 생성한다.

최종적으로 Actor가 출력한 결과는 detokenization 과정을 거쳐 텍스트 및 action 형태로 해석되며, 로봇은 해당 action을 실행한다. 실행 결과에 따라 성공 또는 실패 여부가 reward로 계산되고, 이 reward를 기반으로 GRPO 학습 커리큘럼을 통해 Actor를 개선하는 구조로 설계하였다.

```

```

정리하면, Planner는 빠르게 1차 action을 제안하는 역할을 하고, Actor는 해당 action을 다시 검토하고 보정하는 역할을 담당한다. 이 구조를 통해 단일 VLA 모델이 바로 action을 실행하는 방식보다, 행동 이전에 한 번 더 판단을 점검하는 과정을 추가할 수 있다.
 
---
![Figure2](https://github.com/user-attachments/assets/15b2c0a8-c4bb-4968-bf58-09bde7475a99)
- **2. Model Process**

 ## 모델 전체 프로세스

모델의 전체 프로세스는 Planner와 Actor가 각자의 vision encoder와 language model을 사용하여 입력을 처리하고, Planner가 제안한 action token을 Actor가 다시 검토하는 방식으로 구성된다.

Planner는 이미지와 텍스트를 입력으로 받아, OpenVLA 내부의 ViT와 LLM을 통해 처리한 뒤 action token을 추론하고, 최종적으로 **action token id**를 출력한다.

Actor는 동일한 이미지와 텍스트를 입력으로 받아, SmolVLM 내부의 ViT와 LLM을 통해 별도로 처리한다. 이와 동시에 Planner가 생성한 action token id는 OpenVLA에서 가져온 **frozen action embedding table**을 통해 embedding으로 변환된다. 이후 이 embedding은 **projection layer**를 거쳐 SmolVLM의 embedding 차원에 맞게 변환되며, SmolVLM processor를 통해 생성된 image-text embedding과 concat된다.

또한 SmolVLM의 tokenizer에는 OpenVLA의 **256개 action token**이 미리 추가되어 있어, Actor가 action token과 자연어 텍스트를 함께 다룰 수 있도록 구성하였다.

이렇게 결합된 입력은 Actor의 transformer로 들어가 attention 연산을 수행하고, 최종 출력은 두 갈래로 해석된다. 하나는 **detokenization**을 거쳐 critique text 또는 자연어 텍스트로 표현되고, 다른 하나는 **action token**으로부터 action vector 형태로 변환되어 로봇 제어 입력으로 사용된다. 이후 로봇은 해당 action을 실행한다.

```mermaid

```

정리하면, Planner는 1차 action token을 생성하는 역할을 담당하고, Actor는 이미지·텍스트 정보와 Planner의 action 정보를 함께 받아 이를 다시 해석하고 보정하는 역할을 수행한다. 이를 통해 단일 모델이 바로 행동을 생성하는 구조와 달리, 행동 이전에 한 번 더 검토하는 단계를 추가할 수 있다.
 
---
![Figure3](https://github.com/user-attachments/assets/cf47ba2e-f4df-4c42-9f32-b77c0ebf5a60)
- **3. Projection Layer**

  Openvla Embedding(4096) 차원을 Smolvlm embedding(960) 차원으로 맞춰주는 역할. projection의 구조는 LLaVA의 projection Layer를 기반으로 가져옴.

---
![Figure4](https://github.com/user-attachments/assets/e5adbf70-5ef6-4e22-94d6-8828ed1dc6be)
- **4. FLow Chart**

 ## 모델 플로우 차트

아래 플로우 차트는 모델 내부의 전체적인 데이터 흐름을 정리한 것이다.  
Planner와 Actor의 세부 내부 구조는 앞서 설명한 내용과 동일하므로 여기서는 생략하고, **데이터가 어떤 경로로 수집·전달·학습되는지**를 중심으로 표현하였다.

Planner와 Actor에서 사용하는 이미지 및 텍스트 데이터는 모두 **LIBERO 환경**에서 수집된다.  
실행 과정에서는 Actor가 **ZeroMQ**를 통해 Planner 서버에 요청을 보내고, Planner는 OpenVLA 기반으로 action token id를 추론하여 반환한다.

또한 Actor의 tokenizer 초기화 과정에서는 OpenVLA에서 사용하는 **256개의 action token**을 가져와 SmolVLM tokenizer에 추가한다. 이후 action embedding table을 반영하고, embedding resize를 수행한 뒤 projection layer를 연결하여 Actor가 OpenVLA action token을 처리할 수 있도록 초기화한다.

아래쪽 학습 흐름은 **RLinf 프레임워크 기반의 GRPO 학습 과정**을 반영한다.  
`collect rollout` 단계에서는 LIBERO 환경에서 그룹 사이즈만큼 데이터를 수집하고, 이 과정에서 Actor는 비판적 텍스트와 action token을 함께 추론한다. 이후 action이 실제로 수행되며, 그 결과로부터 reward와 loss를 계산한다. `compute` 단계에서는 이를 바탕으로 가중치 업데이트를 수행하고, 최종적으로 **LoRA 파라미터를 학습**한다.

```mermaid

```

정리하면, 전체 파이프라인은 **LIBERO 환경에서 데이터를 수집하고**, **Actor가 Planner와 ZeroMQ로 상호작용하며**, **Planner의 action token 정보를 Actor가 다시 활용하여 비판적 텍스트와 action을 함께 생성**하는 구조로 이루어진다. 이후 RLinf 기반 GRPO 학습을 통해 reward를 반영한 가중치 업데이트를 수행하고, 최종적으로 LoRA를 학습시키는 흐름으로 구성된다.
 
---
![Figure5](https://github.com/user-attachments/assets/1c0fbf23-fe96-475a-8827-3fd545224f3e)
- **5. Actor Tokenizer Process**

 구현한 actino tokenizer의 구조를 구체화. 각 데이터들은 각각 vision encoder, llm, projection layer를 통과하여 임베딩 형태로 변환되고, input merge와 action injection hook을 통해 같은 공간으로 투영되고 concat을 진행함. 이때 이미지와 텍스트는 smolvlm의 프로세서를 거침. 이후 트랜스포머를 통과하고 출력을 냄.
 
---
![Figure6](https://github.com/user-attachments/assets/f106f294-3e26-4129-ab98-1bde2140310e)
- **6. Suprevised Fine Tuning**
  
 바로 모델 학습으로 들어가면 모델이 텍스트 포멧과 액션토큰을 어떻게 내보내는지 알지 못함. GRPO 학습에서 계속 패널티를 받고, 수렴하지 못하는 문제 발생 가능. 그래서 SFT를 통해 기본적인 베이스 능력을 학습시킨 뒤 비판적 텍스트와 그에 따른 액션 토큰 수정을 하도록 하기 위함. 총 4개 스테이지로 구성되었고, 스테이지마다 순차적으로 학습.

---
## 핵심 기술 과제 및 구현 방식

본 프로젝트에서는 OpenVLA와 SmolVLM을 하나의 듀얼 시스템으로 연결하기 위해 여러 기술적 문제를 해결해야 했다. 두 모델은 임베딩 차원, 토크나이저 구조, 실행 환경이 서로 다르기 때문에 단순히 하나의 모델처럼 연결할 수 없었다. 따라서 action token 표현 방식, 모델 간 통신 구조, 메모리 효율화, Actor 출력 형식을 중심으로 다음과 같은 구현을 진행하였다.

| 기술적 문제 | 구현 방식 | 진행 상태 |
|------|------|------|
| OpenVLA의 4096차원 action embedding과 SmolVLM의 embedding 공간이 서로 맞지 않음 | LLaVA 구조를 참고하여 **projection layer**를 구현하고, OpenVLA action embedding을 SmolVLM이 처리할 수 있는 차원으로 변환 | 완료 |
| SmolVLM이 OpenVLA의 action을 토큰 단위로 처리하지 못함 | SmolVLM tokenizer vocabulary에 OpenVLA 기반 **action token 256개**를 추가 | 완료 |
| Planner와 Actor가 서로 다른 프로세스 및 환경에서 실행됨 | **ZeroMQ**를 이용하여 Planner가 생성한 action token을 Actor로 전달하는 통신 구조 구현 | 구현 완료 |
| 제한된 GPU 자원에서 OpenVLA-7B와 SmolVLM 500M을 함께 운용해야 함 | 4bit 양자화, Planner frozen, Actor LoRA 적용을 통해 메모리 사용량 절감 | 완료 |
| Actor가 자연어 critique와 action token을 함께 출력해야 함 | SFT를 통해 텍스트 출력 형식과 action token 생성 능력을 함께 학습 | 정성적 확인 완료 |

---

## 검증 결과

본 프로젝트에서는 컴퓨팅 자원과 학습 시간의 제약으로 인해 LIBERO benchmark 기반의 정량 success rate 평가는 수행하지 못하였다. 대신 구현한 구조가 의도한 방식대로 동작하는지를 중심으로 정성적 검증을 진행하였다.

- **Action token 생성 확인**  
  SFT 이후 Actor가 새롭게 추가된 256개의 action token을 출력할 수 있음을 확인하였다.

- **자연어 출력 기능 확인**  
  Actor가 action token만 생성하는 것이 아니라, 검토 과정에 해당하는 자연어 텍스트도 함께 출력할 수 있음을 확인하였다. 다만 action token 학습이 진행되면서 기존 자연어 생성 능력은 일부 약화되는 경향이 나타났다.

- **End-to-end 파이프라인 동작 확인**  
  Planner(OpenVLA)가 1차 action을 생성하고, 해당 action 정보가 ZeroMQ를 통해 Actor(SmolVLM 500M)로 전달되는 전체 흐름을 확인하였다.

정리하면, 본 단계에서는 최종 성능 향상 여부를 수치로 입증하기보다는 Planner와 Actor가 분리된 구조에서 정상적으로 연결되는지, 그리고 Actor가 action token과 자연어 출력을 함께 수행할 수 있는지를 검증하는 데 초점을 두었다. 이를 통해 향후 GRPO 학습과 LIBERO-long 정량 평가로 확장하기 위한 기반 파이프라인을 마련하였다.

![work](https://github.com/user-attachments/assets/52948375-398b-4432-8eab-1c4049e69324)

> planner의 추론 action token을 학습 데이터로 사용하여 형식 학습. 성공적으로 7개 토큰 출력.
> 
> 정량 평가(LIBERO-long success rate, baseline 대비 비교)와 GRPO 학습은 충분한 컴퓨팅 자원 확보 후 진행할 향후 과제.

## 코드 구조

### openvla_planner

- **openvla inference code**

  OpenVLA 모델의 inference를 수행하기 위한 코드이다.  
  Planner 역할을 담당하며, ZeroMQ와 연결하여 Actor 모델로 action 정보를 전달할 수 있도록 서버 형태로 동작한다.

  본 프로젝트에서는 Planner를 학습하지 않고 추론용으로만 사용하기 때문에 CUDA를 사용하지 않고 **CPU 기반 inference**로 실행하도록 구성하였다.  
  모델은 `transformers` 기반으로 로드하며, 사용한 모델은 **openvla-7b-finetuned-libero-spatial**이다.

- **action tokenizer**

  OpenVLA에서 사용하는 기존 action tokenizer를 확인하기 위한 코드이다.  
  Planner가 생성하는 action token 구조를 확인하고, Actor 쪽 tokenizer 확장 및 action token 매핑 과정에서 참고하기 위해 유지하였다.

---

### qwen_actor

- **actor_action_tokenizer**

  Qwen 계열 Actor 모델이 OpenVLA의 action token을 처리할 수 있도록 tokenizer를 확장하는 코드이다.  
  기본 LLM tokenizer에는 OpenVLA action token이 포함되어 있지 않기 때문에, OpenVLA에서 사용하는 **256개의 action token**을 vocabulary에 추가하였다.

  또한 Planner의 action embedding table을 가져온 뒤, projection layer를 통해 Qwen 모델의 embedding 차원에 맞게 변환한다.  
  이후 Qwen processor와 연결하여, Actor가 자연어 텍스트와 action token을 함께 입력 및 출력할 수 있도록 구성하였다.

  주요 확인 부분은 `setup` 함수와 `forward` 함수이다.

- **projection_layer**

  OpenVLA와 Qwen2.5-VL은 서로 다른 embedding 공간을 사용하기 때문에, 두 모델을 직접 연결하면 차원 불일치 문제가 발생한다.  
  특히 OpenVLA의 action embedding은 **4096차원**이므로, 이를 그대로 Qwen2.5-VL에 입력할 수 없다.

  이를 해결하기 위해 LLaVA에서 사용되는 projection layer 구조를 참고하여, OpenVLA의 action embedding을 Qwen2.5-VL이 처리 가능한 차원으로 변환하는 layer를 구현하였다.
  >
  >  **LLaVA** : [LLaVA](https://github.com/haotian-liu/LLaVA/blob/main/llava/model/multimodal_projector/builder.py)

- **actor_model**

  > qwen에 4bit quantization + LoRA를 적용시키고 zeroMQ와 통합한 actor의 실행파일.

---

**SmolVLM_actor**

- **smol_action_tokenizer**

  > smolvlm LLM tokenizer에 openvla의 액션 토큰을 이식.
  > 모델 실행시 토크나이저를 초기화시킴.

- **smol_actor_model**

  > smol에 4bit quatization + LoRA + zeroMQ 적용.

- **smol_projection_layer**

  > openvla와 smolvlm의 토크나이저 임베딩 공간을 맞춰줌. 
  > LLaVA의 prijection layer를 참고함.

---

**train_file**

- **train**

  > main역할을 하는 파일임. GRPO와 LIBERO를 실행.

- **smol_train**

  > smolvlm train실행 파일.
  > GRPO를 RLinf를 통해 구현함.
  >
  > **RLinf** : [RLinf](https://github.com/RLinf/RLinf)
  >
  > colliect_rollout과 compute_grpo_loss 함수로 학습 진행.
  > 모델 실행시 해당 파일을 실행해야함.

- **smol_sft**

  > GRPO강화학습을 진행하기 전 SFT를 통해 모델이 사전 학습을 하도록 유도.
  > DeepSeek에서 사용한 방식.
  > 강화학습 모델의 성능이 더 좋아지고, 성능에도 긍정적 효과를 준다고 증명됨.
  > 여기선 모델의 비판적 텍스트 생성과 7개 액션 토큰 생성 능력을 sft를 통해 학습 시킴.

---

**SFT**

- **SFT**
  
  > GRPO 학습을 들어가기 전 사전 학습을 통해 actor에게 텍스트 포멧 형식을 학습시키기 위함.

---

**assets**

- **make_embeddings.py**

  > openvla action embedding 파라미터를 다운받는 파일.
  > 모델 실행시키기 전에 무조건 한 번 실행시켜야함.

---

**checkpoints**

- **sft**

 > SFT를 수행한 모델 체크포인트 파일.
 > train이 마음에 안 들때 해당 체크포인트로 다시 학습.
 > 해당 학습을 시킬 때 비전 인코더를 사용하지 않은 오류가 있음. 사용하지 않음

- **sft2**

  > SFT 수행한 체크포인트.
  > 비판적 텍스트와 액션 토큰을 추론하는 능력을 기본적으로 학습 시킴.
  > 해당 체크파일은 비전 인코더 사용 버전으로, 해당 파일 사용.

---

**logs**

- 학습시 나왔던 로그들을 저장해둠. 어떻게 학습이 되었는지 기록용


## 환경 구성

```
#torch version (qwen)

pip install torch torchvision torchaudio \ --index-url https://download.pytorch.org/whl/cu124

pip install requirements_qwen.txt --no-deps

git clone https://github.com/Lifelong-Robot-Learning/LIBERO

#torch version (openvla)

pip install torch==2.12.0+cpu torchvision==0.27.0+cpu torchaudio==2.11.0+cpu --index-url https://download.pytorch.org/whl/cpu

pip install requirements_openvla.txt
```

SmolVLM 모델 실행시 미리 구성한 qwen 환경에서 실행해도 무방함. 

## 실행 방법 (Getting Started)
---

- 모델을 실행하기 전 openvla_embeddings 파일이 필요함.openvla 환경에 진입해서 

```
git clone https://github.com/imgonnago/AIproject

conda activate openvla

python make_embeddings.py #embedding파일 생성.

python openvla_planner/openvla_inference_code.py

#actor 환경 진입 및 실행(SmolVLM도 qwen 환경 사용)

conda activate qwen

#파일 실행시 openvla를 먼저 실행한 뒤 zeroMQ 서버가 열리고 qwen을 실행해야함.

python train/train.py
```

## 기타 설정
---

모델 실행시 vram 사용량과 train log를 기록할 수 있는 코드.

```
#vram_log

nvidia-smi --query-gpu=timestamp,memory.used --format=csv -l 1 >> vram_log.csv &

#train_log

python train/train.py >> train_log.txt 2>&1  or  python train/smol_train.py >> train_log.txt 2>&1

"""
터미널에 입력하면 nvidia 프로세서 정보와 vram 사용량 gpu사용량을 볼 수 있는 코드.

숫자를 바꾸면 해당 초 마다 사용량을 볼 수 있음.
"""

watch -n 0.5 nvidia-smi

#GPU가 어떤 프로세스를 사용하는지 확인할 수 있는 코드.

nvidia-smi pmon -c 1
```

vscode에서 계속 SSH서버가 끊어질 때, 우분투에서

`tmux new -s planner`

`conda activate openvla`

실행하기 전 프로젝트 폴더 안으로 이동해서 실행.

`python openvla_planner/openvla_inference_code.py`

`tmux new -s train`

`conda activate qwen`

마찬가지로 프로젝트 폴더 안으로 이동해서 실행.
```
#모델 실행 전 이 환경변수를 설정하면 vram에 도움됨.
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

`nvidia-smi --query-gpu=timestamp,memory.used --format=csv -l 1 >> vram_log.csv &`

`python train/smol_train.py 2>&1 | tee train_log.txt`s

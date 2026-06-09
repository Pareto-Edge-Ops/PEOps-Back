"""Per-layer descriptions (title / Korean summary / LaTeX formula).

Port of PEOps-Front/src/features/architecture/lib/layerDescriptions.ts so the
hover card / inspector content can be served by the backend. Resolution
priority: (1) ID-pattern match, (2) generic kind fallback.
"""

from __future__ import annotations

import re

_SUBSCRIPT = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")


def _sub(n: int) -> str:
    return str(n).translate(_SUBSCRIPT)


def get_layer_description(node_id: str, kind: str, name: str, model_type: str) -> dict:
    """Returns {"title": str, "summary": str, "formula": str | None}."""
    d = _by_id(node_id) or _generic(kind, name, model_type)
    d.setdefault("formula", None)
    return d


def _by_id(node_id: str) -> dict | None:  # noqa: C901 — direct port of the front table
    # ---------- LSTM: t{n}_gate_{f|i|c|o}, t{n}_cell, t{n}_hidden ----------
    m = re.match(r"^t(\d+)_(gate_[ficod]|cell|hidden)$", node_id)
    if m:
        t = int(m.group(1)) + 1
        sub = m.group(2)
        table = {
            "gate_f": dict(
                title=f"Forget gate · t={t}",
                summary="이전 cell state c_{t-1}을 얼마나 유지할지 결정하는 σ 게이트.",
                formula=r"f_t = \sigma(W_f \cdot [h_{t-1},\, x_t] + b_f)",
            ),
            "gate_i": dict(
                title=f"Input gate · t={t}",
                summary="새로 들어온 후보 정보를 cell state에 얼마나 반영할지 결정.",
                formula=r"i_t = \sigma(W_i \cdot [h_{t-1},\, x_t] + b_i)",
            ),
            "gate_c": dict(
                title=f"Candidate · t={t}",
                summary="현재 timestep에서 cell state에 추가할 후보 콘텐츠 (tanh).",
                formula=r"\tilde{c}_t = \tanh(W_c \cdot [h_{t-1},\, x_t] + b_c)",
            ),
            "gate_o": dict(
                title=f"Output gate · t={t}",
                summary="cell state로부터 hidden 출력을 얼마나 내보낼지 결정.",
                formula=r"o_t = \sigma(W_o \cdot [h_{t-1},\, x_t] + b_o)",
            ),
            "cell": dict(
                title=f"Cell state · t={t}",
                summary=(
                    "LSTM의 장기 기억. forget·candidate 게이트로 갱신되며 다음 timestep으로 "
                    "이어지는 recurrent state."
                ),
                formula=r"c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t",
            ),
            "hidden": dict(
                title=f"Hidden · t={t}",
                summary=(
                    "현재 timestep의 출력. 다음 timestep의 모든 게이트 입력으로도 재사용되는 "
                    "recurrent 상태."
                ),
                formula=r"h_t = o_t \odot \tanh(c_t)",
            ),
        }
        return table.get(sub)

    # ---------- HAN ----------
    m = re.match(r"^type_(user|item|session)$", node_id)
    if m:
        tag = m.group(1)[0]
        return dict(
            title=f"Type-specific projection · {m.group(1)}",
            summary=(
                "노드 타입별로 정의된 Linear 변환. heterogeneous 그래프의 서로 다른 타입을 "
                "공통 임베딩 공간으로 정렬."
            ),
            formula=rf"h'_v = W_{{{tag}}} \cdot x_v",
        )
    m = re.match(r"^mp(\d+)_h(\d+)$", node_id)
    if m:
        p, h = int(m.group(1)) + 1, int(m.group(2))
        return dict(
            title=f"Meta-path Φ{_sub(p)} attention · head H{h}",
            summary=f"메타패스 Φ{_sub(p)} 이웃에 대한 multi-head node-level attention의 {h}번 head.",
            formula=(
                r"\alpha_{ij} = \mathrm{softmax}\bigl(\mathrm{LeakyReLU}"
                r"(\mathbf{a}^\top [W h_i \,\|\, W h_j])\bigr)"
            ),
        )
    m = re.match(r"^mp(\d+)_embed$", node_id)
    if m:
        p = int(m.group(1)) + 1
        return dict(
            title=f"Meta-path embedding z^Φ{_sub(p)}",
            summary=f"메타패스 Φ{_sub(p)}의 모든 head 출력을 concat한 노드 표현.",
            formula=(
                rf"z_v^{{\Phi_{p}}} = \big\Vert_{{k=1}}^{{K}} \sigma\!\left("
                rf"\sum_{{j}} \alpha_{{ij}}^{{k}}\, W^{{k}} h_j\right)"
            ),
        )
    if node_id == "sem_score_mlp":
        return dict(
            title="Semantic attention scoring MLP",
            summary=(
                "각 메타패스 embedding을 받아 하나의 스칼라 중요도 점수로 압축. 모든 노드에 "
                "대해 평균 후 softmax되어 β_p가 됨."
            ),
            formula=(
                r"w_p = \frac{1}{|V|} \sum_{v \in V} \mathbf{q}^\top \cdot "
                r"\tanh\bigl(W \cdot z_v^{\Phi_p} + b\bigr)"
            ),
        )
    if node_id == "sem_softmax":
        return dict(
            title="softmax(β_p)",
            summary="메타패스별 점수를 정규화한 attention weight. 모든 메타패스에 대한 합은 1.",
            formula=r"\beta_p = \dfrac{\exp(w_p)}{\sum_{p'} \exp(w_{p'})}",
        )
    if node_id == "sem_agg":
        return dict(
            title="Semantic aggregation Z",
            summary="메타패스 embedding들을 β로 가중합한 최종 노드 표현. 분류 head로 전달됨.",
            formula=r"Z_v = \sum_{p=1}^{P} \beta_p \cdot z_v^{\Phi_p}",
        )

    # ---------- DiT ----------
    if re.match(r"^patch_(tl|tr|bl|br)$", node_id):
        return dict(
            title="Image patch",
            summary=(
                "입력 이미지(latent)를 16×16 패치로 분할한 한 조각. 4개 패치(2×2 grid)가 "
                "시퀀스로 펼쳐져 transformer에 들어감."
            ),
        )
    if node_id == "patch_embed":
        return dict(
            title="Patch embed + Position embed",
            summary="각 패치를 latent 차원으로 Linear 임베딩한 뒤 학습된 position embedding을 더함.",
        )
    if node_id == "t_embed":
        return dict(
            title="Timestep embedding",
            summary="Diffusion timestep t를 sinusoidal + MLP로 임베딩. 모든 block에 conditioning 신호로 주입됨.",
        )
    if node_id == "c_embed":
        return dict(
            title="Class embedding",
            summary="조건부 생성에 쓰는 class label embedding. timestep과 합쳐 conditioning이 됨.",
        )
    if node_id == "cond_mlp":
        return dict(
            title="Conditioning MLP (AdaLN params)",
            summary=(
                "timestep + class embedding을 받아 모든 transformer block에 주입할 AdaLN "
                "변조 파라미터(γ, β, α)를 산출."
            ),
            formula=r"(\gamma, \beta, \alpha) = \mathrm{MLP}(\mathrm{Embed}(t) + \mathrm{Embed}(c))",
        )
    m = re.match(r"^db(\d+)_attn_h(\d+)$", node_id)
    if m:
        return dict(
            title=f"DiT block {int(m.group(1))} · attention head H{int(m.group(2))}",
            summary="AdaLN으로 변조된 입력에 대한 multi-head self-attention. 8개 head 중 하나.",
            formula=(
                r"\mathrm{Attn}(Q, K, V) = \mathrm{softmax}\!\left("
                r"\frac{Q K^\top}{\sqrt{d_k}}\right) V"
            ),
        )
    m = re.match(r"^db(\d+)_postattn$", node_id)
    if m:
        return dict(
            title=f"Block {int(m.group(1))} · residual #1 (after attn)",
            summary="attention 출력에 입력을 더하는 첫 번째 residual 합산점. AdaLN 변조 결과를 통합.",
            formula=r"y_1 = x + \alpha_1 \cdot \mathrm{Attn}\bigl(\mathrm{AdaLN}(x,\, \mathbf{c})\bigr)",
        )
    m = re.match(r"^db(\d+)_ffn$", node_id)
    if m:
        return dict(
            title=f"Block {int(m.group(1))} · FFN + residual #2",
            summary=(
                "AdaLN 변조 후 pointwise FFN을 적용하고 다시 residual로 합치는 두 번째 "
                "합산점. 블록 출력."
            ),
            formula=r"y_2 = y_1 + \alpha_2 \cdot \mathrm{FFN}\bigl(\mathrm{AdaLN}(y_1,\, \mathbf{c})\bigr)",
        )
    if node_id == "final_norm":
        return dict(
            title="Final AdaLN",
            summary="마지막 conditional layer normalization. linear projection 직전에 한 번 더 cond로 변조.",
        )
    if node_id == "final_linear":
        return dict(
            title="Linear → ε prediction",
            summary=(
                "최종 latent를 patch 단위로 다시 펼쳐 예측 noise ε(또는 x₀)를 산출. "
                "patches → image로 reshape."
            ),
            formula=r"\hat{\varepsilon} = W \cdot y_N + b",
        )

    # ---------- CNN ----------
    if node_id == "stem_conv":
        return dict(
            title="Stem 7×7 conv",
            summary="초기 spatial 다운샘플과 저수준 feature 추출. ResNet 계열의 stem 패턴.",
        )
    m = re.match(r"^s(\d+)_conv$", node_id)
    if m:
        return dict(
            title=f"Stage {int(m.group(1)) + 1} · conv (residual main)",
            summary="residual block의 main path conv. skip 경로와 stage 끝에서 합산됨.",
        )
    m = re.match(r"^s(\d+)_pool$", node_id)
    if m:
        return dict(
            title=f"Stage {int(m.group(1)) + 1} · pool (residual join)",
            summary="stage 끝의 spatial 다운샘플. residual skip이 합류하는 지점.",
        )
    if node_id == "avg_pool":
        return dict(
            title="Global average pool",
            summary="spatial 차원을 모두 평균해 feature vector 1개로 압축.",
        )

    # ---------- Tree ----------
    if node_id == "root":
        return dict(
            title="Root split",
            summary="결정 트리의 루트 분기. 가장 정보 이득이 큰 feature·threshold로 좌우로 가른다.",
        )
    if node_id.startswith("split_"):
        return dict(
            title=f"Internal split · {node_id.removeprefix('split_')}",
            summary="내부 분기 노드. 또 다른 feature·threshold로 자식 노드를 갈라냄.",
        )
    if node_id.startswith("leaf_"):
        return dict(
            title=f"Leaf · {node_id.removeprefix('leaf_')}",
            summary="리프 노드. 도달한 샘플들의 평균 예측(또는 클래스 분포)을 출력.",
        )
    if node_id == "vote":
        return dict(
            title="Ensemble vote (softmax)",
            summary="GBDT의 각 트리 출력을 가중 합산하고 softmax로 클래스 확률 분포로 변환.",
        )
    return None


_GENERIC: dict[str, dict] = {
    "embed": dict(title="Embedding", summary="이산 토큰을 dense 벡터로 변환하는 lookup 임베딩."),
    "conv": dict(title="Convolution", summary="공간 weight-sharing 필터로 local feature 추출."),
    "bn": dict(title="Batch normalization", summary="미니배치 단위로 평균·분산을 정규화. 학습 안정화에 기여."),
    "relu": dict(title="ReLU", summary="비선형 활성. 음수는 0으로 잘라냄.", formula=r"f(x) = \max(0,\, x)"),
    "pool": dict(title="Pooling", summary="spatial 다운샘플. max 또는 average."),
    "dense": dict(title="Dense (Linear)", summary="fully-connected linear projection.", formula=r"y = W x + b"),
    "attn": dict(
        title="Multi-head attention",
        summary="scaled dot-product attention. token 간 의존성을 학습.",
        formula=r"\mathrm{Attn}(Q, K, V) = \mathrm{softmax}\!\left(\frac{Q K^\top}{\sqrt{d_k}}\right) V",
    ),
    "ffn": dict(title="Feed-forward", summary="transformer block 안의 2-layer MLP (hidden은 GELU/ReLU)."),
    "norm": dict(title="LayerNorm", summary="채널 단위 정규화. 학습 안정성·수렴 속도 향상."),
    "lstm": dict(title="LSTM gate", summary="LSTM 셀 내부의 게이트 또는 cell state 컴포넌트."),
    "softmax": dict(title="Softmax", summary="logit을 합 1의 확률 분포로 변환."),
    "upsample": dict(title="Upsample", summary="spatial 해상도를 2× 키움 (nearest 또는 bilinear)."),
}


def _generic(kind: str, name: str, model_type: str) -> dict:
    if kind == "input":
        return dict(title="Input", summary=f"{model_type} 모델의 입력 텐서.")
    if kind == "output":
        return dict(title="Output", summary=f"{model_type} 모델의 최종 출력.")
    found = _GENERIC.get(kind)
    if found:
        return dict(found)
    return dict(title=name, summary=f"{model_type}의 한 layer.")

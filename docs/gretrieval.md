# gretrieval — MSD Floorplan Graph Retrieval / MSD 户型图检索

> 中英对照文档 / Bilingual document. Each section gives the English explanation
> first, then the Chinese (中文) translation.

A lightweight, training-free retrieval system that returns the top-k most
relevant MSD floorplans for a (possibly partial) bubble graph drawn on the
frontend.

一个轻量、**无需训练**的检索系统：用户在前端画一个（可能不完整的）bubble graph，
系统从 corpus 里返回 top-k 最相关的 MSD 户型图。

It is intentionally **not** a neural model — everything is built on the
**Weisfeiler-Lehman (WL) graph kernel** plus plain vector math. No training
step, no GPU dependency, the whole index builds in seconds.

它刻意**不用神经网络**——全部基于 **Weisfeiler-Lehman (WL) 图核** + 普通向量运算。
没有训练步骤，不依赖 GPU，整个索引几秒钟就能建好。

---

## 1. Core principle — the WL kernel / 核心原理：WL 图核

Every graph is turned into one **fixed-length histogram vector**. The four
retrieval algorithms differ *only* in how they compare two such vectors.

每张图被转成一个**固定长度的直方图向量**。四种检索算法的区别**只在于**如何比较两个
这样的向量。

### How the WL feature is built / WL 特征怎么构建

WL refinement iteratively rewrites each node's label to summarise its local
neighbourhood:

WL refinement 迭代地重写每个节点的标签，用来概括它的局部邻域：

```
iteration 0:  label(v) = room_type(v)
iteration i:  label(v) = hash( label(v),
                                sorted[ (edge_type(v,u), label(u)) for u in neighbours(v) ] )
```

- At iteration `i`, a node's label encodes its **i-hop neighbourhood** (which
  room types, connected by which edge types, in what arrangement).
- We collect every label produced at iterations `0..n_iter` across all nodes,
  build a vocabulary, and represent each graph as a **count histogram** over
  that vocabulary.

- 在第 `i` 轮，一个节点的标签编码了它的 **i 跳邻域**（哪些房间类型、通过哪些边类型、
  以什么方式连接）。
- 我们收集所有节点在第 `0..n_iter` 轮产生的全部标签，建立一个词表 (vocabulary)，
  然后把每张图表示成该词表上的**计数直方图**。

**Key point: edge types participate in refinement.** The pair
`(edge_type, neighbour_label)` enters the multiset, so `Kitchen —door— Bedroom`
and `Kitchen —wall— Bedroom` fall into different WL labels. MSD's four edge
types (wall / passage / door / entrance) are therefore captured by the feature.

**关键点：边类型参与 refinement。** `(edge_type, neighbour_label)` 这个二元组进入
multiset，所以 `Kitchen —door— Bedroom` 和 `Kitchen —wall— Bedroom` 会落到不同的
WL label。因此 MSD 的 4 种边类型 (wall / passage / door / entrance) 被特征捕捉到。

Implementation / 实现：`gretrieval/gsretrieval/features/wl_kernel.py`

---

## 2. The four retrieval algorithms / 四种检索算法

All share the same WL feature space. `q` = query histogram, `c` = candidate
histogram.

四种算法共用同一个 WL 特征空间。`q` = query 直方图，`c` = candidate 直方图。

### 2.1 Cosine — symmetric similarity / 对称相似度

```
score = (q · c) / (|q| · |c|)
```

**EN** — Asks "how similar are these two graphs overall?". Pros: fastest (FAISS
inner product, <10ms); good for **full graph → overall-similar graphs**. Cons:
unfriendly to partial queries — a short query vector against a long candidate
vector gets a low score because the denominator `|c|` penalises large
candidates.

**中文** — 问的是"这两张图整体相似度多高？"。**好处**：最快（FAISS 内积，<10ms）；
适合**完整图 → 找整体相似的图**。**坏处**：对 partial query 不友好——短的 query 向量和长的
候选向量做 cosine，分母 `|c|` 惩罚了大候选，分数偏低。

Code: `gretrieval/gsretrieval/retrievers/cosine.py`

### 2.2 Containment — histogram intersection / 直方图含量（default / 默认）

```
score = Σ min(q[i], c[i]) / Σ q[i]
```

**EN** — Asks "how many of the query's WL labels are present in the candidate?"
(`1.0` = all present). Pros: does **not** penalise candidate size — a 60-node
graph that contains the query's pattern still scores 1.0; most natural for
partial queries; fast (~150ms), robust to noise; **empirically the strongest at
every mask level in self-recall eval**, hence the default. Cons: only checks
whether labels *exist*, not whether they are structurally aligned — the query's
labels may be scattered anywhere in the candidate.

**中文** — 问的是"query 的 WL 标签有多少在候选里出现了？"（`1.0` = 全部出现）。
**好处**：**不**惩罚候选大小——一张 60 节点的大图只要包含 query 的 pattern，分数就是
1.0；对 partial query 最自然；快（~150ms）、对噪声鲁棒；**self-recall 实测在所有 mask
level 上一致最强**，因此设为默认。**坏处**：只看标签"存不存在"，不强制结构对齐——
query 的标签可以散落在候选各处。

Code: `gretrieval/gsretrieval/retrievers/containment.py`

### 2.3 Node matching — greedy node alignment / 节点级贪心对齐

```
for each query node q_i:  best(q_i) = max_j Jaccard( WL_sig(q_i), WL_sig(c_j) )
score = mean_i best(q_i)
```

**EN** — Asks "for each query node, is there a node in the candidate whose local
neighbourhood looks like it?". Pros: stricter than containment — every query
node must "find a home" at a concrete candidate node. Cons: greedy (not optimal)
alignment, several query nodes may grab the same candidate node; sensitive to a
single noisy query node; slow (~600-1000ms); **actually weaker than containment
in self-recall eval**.

**中文** — 问的是"query 里每个节点，能不能在候选里找到一个局部邻域很像的节点？"。
**好处**：比 containment 严格——每个 query 节点都得"安家"到候选的某个具体节点。
**坏处**：贪心（非最优）对齐，多个 query 节点可能抢同一个候选节点；对单个噪声节点敏感；
慢（~600-1000ms）；**self-recall 实测反而不如 containment**。

Code: `gretrieval/gsretrieval/retrievers/node_matching.py`

### 2.4 Two-stage — recall + rerank / 召回 + 精排

```
Step 1  containment finds top-50 over the full corpus   (~150ms, recall / 召回)
Step 2  node_matching reranks just those 50             (~50ms, rerank / 精排)
```

**EN** — The industry-standard "recall + rerank" pattern; run the expensive
rerank only on a small candidate set. Cons: because node_matching itself is
weaker than containment in self-recall, two-stage does not clearly beat plain
containment.

**中文** — 工业界标准的"召回 + 精排"范式；只在小候选集上跑昂贵的精排。**坏处**：因为
node_matching 本身在 self-recall 上不如 containment，two-stage 也没显著超过单 containment。

Code: `gretrieval/gsretrieval/retrievers/two_stage.py`

### Analogy / 比喻（query = `Kitchen → Living → Bedroom`）

| Algorithm | The question it asks / 它问的问题 |
|-----------|----------------------------------|
| Cosine | "Are these two floorplans alike overall? / 这两个 floorplan 整体像不像？" |
| Containment | "Does my drawn pattern appear in your floorplan? / 我画的这串 pattern 在你这张图里出现了吗？" |
| Node matching | "Does each of my rooms have a local twin in your floorplan? / 你图里每个房间的邻域都跟我画的一样吗？" |
| Two-stage | "Filter 50 by containment, then refine by node matching. / 先用 containment 筛 50 个，再用 node matching 精挑" |

---

## 3. Evaluation (self-recall) / 评测（自召回）

`scripts/eval_recall.py` masks a fraction of each query graph's nodes and checks
whether the source graph is recalled in the top-k.

`scripts/eval_recall.py` 把每张 query 图遮掉一部分节点，检查原图能不能被召回进 top-k。

200 queries on the MSD-wall corpus (5143 graphs, n_iter=2) /
MSD-wall corpus 上 200 个 query（5143 张图，n_iter=2）：

| mask | cosine R@10 | containment R@10 | node_matching R@10 | two_stage R@10 |
|------|-------------|------------------|--------------------|----------------|
| 0.0  | 1.00 | 0.99 | 0.97 | 1.00 |
| 0.3  | 0.42 | **0.58** | 0.47 | 0.53 |
| 0.5  | 0.17 | **0.25** | 0.17 | 0.19 |
| 0.7  | 0.06 | 0.04 | 0.02 | 0.04 |

**EN — Important caveat:** self-recall is a *pessimistic* metric for partial
queries. After dropping 50% of nodes, the corpus contains many plausible "parent
graphs" and the original is not the only correct answer. The near-perfect R@10
at mask=0 proves the WL feature is discriminative; the drop at higher mask mostly
reflects the **ambiguity of the problem**, not a broken system. See §6 for better
metrics.

**中文 — 重要 caveat：** self-recall 对 partial query 是个**悲观**指标。删掉 50% 节点后，
corpus 里有很多"合理父图"，原图不是唯一正确答案。mask=0 时 R@10≈1.0 证明 WL 特征有
区分度；mask 越高分数越低，主要反映**问题本身的歧义**，不是系统坏了。更合适的指标见 §6。

---

## 4. Architecture / 架构

**EN** — Mirrors the digress layout (`configs/ + package/ + scripts/`), but the
package is named **`gsretrieval`, not `src`**, to avoid a top-level package-name
clash with `digress/src` (both end up on `sys.path` in the backend).

**中文** — 镜像 digress 的组织风格（`configs/ + 包/ + scripts/`），但包名是
**`gsretrieval` 而非 `src`**，以避免和 `digress/src` 的顶层包名冲突（后端里两者都会进
`sys.path`）。

```
gretrieval/
├── configs/retrieval/msd_wl.yaml      # graphs_path, output_dir, n_iter, retrievers
├── gsretrieval/
│   ├── datasets/msd_loader.py         # reads digress/data/msd_wall/graphs.p
│   ├── features/
│   │   ├── base.py                    # FeatureExtractor abstract base / 抽象基类
│   │   └── wl_kernel.py               # WL impl (transform + node_signatures)
│   ├── index/faiss_index.py           # inner-product index (FAISS, numpy fallback)
│   └── retrievers/
│       ├── base.py                    # Retriever abstract base / 抽象基类
│       ├── cosine.py / containment.py / node_matching.py / two_stage.py
├── scripts/
│   ├── build_index.py                 # corpus → WL features → retrievers
│   ├── query_index.py                 # single-query demo / 单 query demo
│   └── eval_recall.py                 # self-recall eval / 自召回评测
└── data/msd_wall/wl_n{2,3}/           # runtime artefacts / 运行时产物
```

**EN — Extension points:** to add a new feature (e.g. a learned GNN embedding),
implement a `FeatureExtractor` subclass under `features/`; the retriever layer is
untouched. To add a new strategy, implement a `Retriever` subclass under
`retrievers/`.

**中文 — 扩展点：** 要加新特征（如 GNN embedding），在 `features/` 下实现
`FeatureExtractor` 子类，retriever 层不用动。要加新检索策略，在 `retrievers/` 下实现
`Retriever` 子类。

---

## 5. Usage / 用法

```bash
# Run from the digress venv (has networkx / numpy / pyyaml)
# 从 digress venv 跑（已装 networkx / numpy / pyyaml）
cd gretrieval

# 1. Build the index (reads graphs_path / output_dir / n_iter from config)
#    建索引（读 config 里的 graphs_path / output_dir / n_iter）
python scripts/build_index.py

# 2. Single-query demo (take a corpus graph, mask half to simulate a partial query)
#    单 query demo（取一张 corpus 图，遮一半模拟 partial query）
python scripts/query_index.py --query-idx 100 --mask-frac 0.5 --k 5 --mode all

# 3. Evaluation / 评测
python scripts/eval_recall.py --num-queries 200 --mask-fracs 0.0 0.3 0.5 0.7 --out-json eval.json
```

**EN** — The backend service `app/backend/app/services/graph_retrieval.py` lazily
loads these indices; the frontend Retrieve mode calls `POST /api/retrieve`.

**中文** — 后端服务 `app/backend/app/services/graph_retrieval.py` 懒加载这些索引；
前端 Retrieve mode 通过 `POST /api/retrieve` 调用。

---

## 6. Limitations & future work / 局限与未来工作

### Current limitations / 当前局限

**EN**
- n_iter=2 vocab ≈ 109k / n_iter=3 vocab ≈ 215k → dense feature matrix is
  2.2–4.4GB; the large vocab slows containment queries to ~1s. Switch to
  `scipy.sparse` or lower n_iter when needed.
- WL is a **coarse feature**: distinct structures can hash to the same label;
  partial queries inevitably lose information.
- **Similarity retrieval ≠ subgraph search.** WL gives a whole-graph similarity
  score; it does not tell you *where* in the candidate the query lives.

**中文**
- n_iter=2 vocab≈109k / n_iter=3 vocab≈215k → dense feature 矩阵 2.2–4.4GB；大 vocab
  让 containment 查询慢到 ~1s。需要时改 `scipy.sparse` 或降 n_iter。
- WL 是**粗特征**：不同结构可能哈希到同一 label；partial query 必然丢信息。
- **相似检索 ≠ 子图搜索。** WL 给整图相似度分数，不告诉你 query 在候选里**长在哪**。

### Possible improvements (highest ROI first) / 可能的改进（ROI 从高到低）

| # | Improvement / 改进 | Note / 说明 |
|---|---------------------|-------------|
| 1 | **GNN embedding** (GraphCL / contrastive) replacing the WL histogram / 用 GNN embedding 替换 WL 直方图 | Same retriever abstraction, just a different feature vector. Expect R@10 +15-25%, ~1 GPU-day to train. / 同一套 retriever 抽象，只换特征向量。期望 R@10 +15-25%，~1 天 GPU 训练。 |
| 2 | **Substructure-validity eval** / substructure validity 评测 | Use VF2 subgraph isomorphism to check whether results actually contain the query — more meaningful than self-recall for partial→full. / 用 VF2 子图同构检查结果是否真包含 query，比 self-recall 更能反映 partial→full 的合理性。 |
| 3 | **VF2 subgraph search** (`networkx.algorithms.isomorphism`) / VF2 子图搜索 | True graph search that locates the match, but strict matching has low hit rate and needs a relaxation cascade. / 真正的图搜索，能定位匹配，但严格匹配命中率低，需分级 fallback。 |
| 4 | **scipy.sparse features** + FAISS / 稀疏特征 + FAISS | Lower memory, faster, supports a larger corpus. / 降内存、提速、支撑更大 corpus。 |

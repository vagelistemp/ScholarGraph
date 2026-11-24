"""A synthetic generator of OpenAlex-format records.

It exists for three reasons: (1) unit tests and the ``demo`` command need data that is small, deterministic and *dense* in citations
(a random 600-paper sample of the real data would contain almost no citations between its own papers); (2) every phase can be
exercised end to end without a download; (3) it plants known structure (topics, concept phrases, communities, preferential
attachment, a few deliberate data errors) so tests can check that the pipeline finds it.

The records have exactly the shape of real API records (inverted-index abstracts, ``https://openalex.org/...`` ids, nested
authorships), so the whole parsing path is exercised, not a shortcut.
"""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

TOPIC_NAMES = ["Graph Neural Networks", "Language Models", "Computer Vision", "Reinforcement Learning", "Recommender Systems",
               "Speech Processing", "Optimization Methods", "Time Series Forecasting"]
# concept phrases (methods / tasks / datasets) that the abstracts mention; the concept extractor of Phase 2 should recover them
METHODS = ["transformer", "graph neural network", "contrastive learning", "diffusion model", "random forest", "attention mechanism",
           "reinforcement learning", "variational autoencoder"]
TASKS = ["link prediction", "question answering", "image classification", "machine translation", "anomaly detection",
         "node classification", "speech recognition", "recommendation"]
DATASETS = ["ImageNet", "Cora", "SQuAD", "MovieLens", "LibriSpeech", "CIFAR-10"]
FILLER = ("we propose a novel approach and evaluate it on several benchmarks. our experiments show consistent improvements over strong "
          "baselines while remaining efficient. we analyse the results and discuss limitations and directions for future work.").split()


def _inverted_index(text: str) -> Dict[str, List[int]]:
    inv: Dict[str, List[int]] = {}
    for i, w in enumerate(text.split()):
        inv.setdefault(w, []).append(i)
    return inv


def make_synthetic_works(n_papers: int = 600, seed: int = 0, year_from: int = 2015, year_to: int = 2025,
                         mean_refs: float = 9.0, n_authors: int = 500) -> List[Dict[str, Any]]:
    rng = np.random.default_rng(seed)
    k_topics = len(TOPIC_NAMES)
    topic_vocab = [[f"{TOPIC_NAMES[t].split()[0].lower()}{j}" for j in range(25)] for t in range(k_topics)]
    years = np.sort(rng.integers(year_from, year_to + 1, size=n_papers))
    topic = rng.integers(0, k_topics, size=n_papers)

    # authors live in a topic community and a fixed institution
    author_topic = rng.integers(0, k_topics, size=n_authors)
    author_inst = rng.integers(0, 40, size=n_authors)
    by_topic = [np.flatnonzero(author_topic == t) for t in range(k_topics)]

    indeg = np.zeros(n_papers)
    refs: List[List[int]] = []
    for i in range(n_papers):
        earlier = np.flatnonzero(years[:i] <= years[i]) if i else np.array([], dtype=int)
        earlier = earlier[earlier != i]
        if len(earlier) == 0:
            refs.append([])
            continue
        w = (indeg[earlier] + 1.0) * np.where(topic[earlier] == topic[i], 6.0, 1.0)
        k = min(len(earlier), max(0, int(rng.poisson(mean_refs))))
        chosen = rng.choice(earlier, size=k, replace=False, p=w / w.sum()) if k else np.array([], dtype=int)
        indeg[chosen] += 1
        refs.append([int(c) for c in chosen])

    records: List[Dict[str, Any]] = []
    for i in range(n_papers):
        t = int(topic[i])
        m, ta = rng.choice(METHODS), rng.choice(TASKS)
        ds = rng.choice(DATASETS)
        words = list(rng.choice(topic_vocab[t], size=22)) + list(rng.choice(FILLER, size=18))
        text = f"{m} for {ta} on {ds} " + " ".join(words) + f" using a {rng.choice(METHODS)} baseline"
        rng.shuffle(words)
        n_auth = int(rng.integers(1, 5))
        pool = by_topic[t] if len(by_topic[t]) >= n_auth else np.arange(n_authors)
        chosen_authors = rng.choice(pool, size=n_auth, replace=False)
        authorships = [{
            "author_position": "first" if p == 0 else "middle",
            "author": {"id": f"https://openalex.org/A{1000 + int(a)}", "display_name": f"Author {int(a)}"},
            "institutions": [{"id": f"https://openalex.org/I{100 + int(author_inst[a])}", "display_name": f"Institute {int(author_inst[a])}",
                              "country_code": ["GR", "DE", "US", "CN", "FR"][int(author_inst[a]) % 5]}],
        } for p, a in enumerate(chosen_authors)]
        referenced = [f"https://openalex.org/W{10_000 + j}" for j in refs[i]]
        referenced += [f"https://openalex.org/W{9_000_000 + int(x)}" for x in rng.integers(0, 10_000, size=int(rng.integers(0, 5)))]
        records.append({
            "id": f"https://openalex.org/W{10_000 + i}",
            "doi": f"https://doi.org/10.0000/synthetic.{i}",
            "title": f"{m.title()} for {ta.title()} ({TOPIC_NAMES[t]}) {i}",
            "publication_year": int(years[i]),
            "type": "article",
            "language": "en",
            "cited_by_count": 0,                                   # filled below from the planted graph
            "referenced_works": referenced,
            "authorships": authorships,
            "primary_topic": {"id": f"https://openalex.org/T{t + 1}", "display_name": TOPIC_NAMES[t], "score": 0.9,
                              "subfield": {"id": "https://openalex.org/subfields/1702", "display_name": "Artificial Intelligence"},
                              "field": {"id": "https://openalex.org/fields/17", "display_name": "Computer Science"}},
            "topics": [{"id": f"https://openalex.org/T{t + 1}", "display_name": TOPIC_NAMES[t], "score": 0.9,
                        "subfield": {"id": "https://openalex.org/subfields/1702", "display_name": "Artificial Intelligence"},
                        "field": {"id": "https://openalex.org/fields/17", "display_name": "Computer Science"}}],
            "keywords": [{"id": f"https://openalex.org/keywords/{m.replace(' ', '-')}", "display_name": m, "score": 0.8},
                         {"id": f"https://openalex.org/keywords/{ta.replace(' ', '-')}", "display_name": ta, "score": 0.7}],
            "primary_location": {"source": {"id": f"https://openalex.org/S{50 + (t * 3 + i % 3)}",
                                           "display_name": f"Journal of {TOPIC_NAMES[t]} {i % 3}", "type": "journal"}},
            "abstract_inverted_index": _inverted_index(text),
        })
    for r, j in zip(records, range(n_papers)):
        r["cited_by_count"] = int(indeg[j])

    # deliberate data errors that the quality rules must catch
    if n_papers >= 20:
        records.append(dict(records[3]))                                                  # exact duplicate record
        bad = dict(records[5]); bad["id"] = "https://openalex.org/W77777"; bad["abstract_inverted_index"] = None
        records.append(bad)                                                               # no abstract
        future = dict(records[2]); future["id"] = "https://openalex.org/W77778"; future["publication_year"] = year_from
        future["referenced_works"] = [f"https://openalex.org/W{10_000 + n_papers - 1}"]
        records.append(future)                                                            # cites a (much) later paper
    return records

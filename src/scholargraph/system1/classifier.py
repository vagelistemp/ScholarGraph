"""A small trained intent classifier: the "System 1" decision of which route a request should take.

Classes
-------
graph_lookup   an exact question about entities and links ("who wrote ...", "how many papers cite ...")  -> direct handler, else agent
topic_search   an open question about a topic ("what are the main approaches to ...")                      -> retrieval + RAG answer
analysis       a multi-step comparison or trend question                                                  -> agent / planner
off_topic      nothing to do with scientific papers                                                       -> polite refusal

The model is deliberately tiny and fast (character n-gram TF-IDF + logistic regression, microseconds per request, no GPU, no network).
Quoted titles and numbers are masked before featurising so the model learns the *shape* of a question, not the particular paper.

Honest limits: the training data is generated from templates filled with entities of the graph. ``evaluate_generalisation`` therefore
trains on one set of phrasings and tests on a **different** set of phrasings, which is the only meaningful test; accuracy on real user
questions has not been measured and will be lower. When the classifier is unsure (confidence below a threshold) the request falls
back to the most capable route (the agent) instead of guessing.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

LABELS = ["graph_lookup", "topic_search", "analysis", "off_topic"]

TRAIN_TEMPLATES: Dict[str, List[str]] = {
    "graph_lookup": ['Who wrote the paper titled "{t}"?', 'Who are the authors of "{t}"?', 'Who authored "{t}"?', 'How many papers cite "{t}"?',
                     'How many times has "{t}" been cited?', 'Which papers does "{t}" cite?', 'List the references of "{t}"', 'What are the references of "{t}"?',
                     'Which papers cite both "{t}" and "{t2}"?', 'Papers citing both "{t}" and "{t2}"', "Which papers published in {y} or later mention {c}?",
                     "List papers from {y} onwards about {c}", "What is the most cited paper in {y}?", "Which paper from {y} has the highest number of citations?",
                     "Who collaborates most with {n}?", "Who are the main co-authors of {n}?"],
    "topic_search": ["What are the main approaches to {c}?", "Find papers about {c}.", "Explain how {c} is used for {k}.", "Which papers discuss {c} for {k}?",
                     "Summarize research on {c}.", "What is known about {k}?", "What methods exist for {k}?", "Search for work on {k}.", "Describe how {c} works.",
                     "What do papers say about {c}?", "Show papers on {k} using {c}.", "Which techniques address {k}?"],
    "analysis": ["Compare {c} and {c2} for {k}.", "How has {c} evolved over the years?", "Which authors work on both {c} and {c2}, and what do they have in common?",
                 "Is there a trend from {c} towards {c2}?", "What are the strengths and weaknesses of {c} compared with {c2}?", "Analyse how {k} research changed over time.",
                 "What is the relationship between {c} and {c2}?", "Which is better for {k}, {c} or {c2}, and why?", "How do the communities working on {c} and {c2} differ?",
                 "Trace the history of {c}."],
    "off_topic": ["What's the weather like today?", "Write me a poem about the sea.", "Tell me a joke.", "What is the capital of France?", "Who won the football match yesterday?",
                  'Translate "good morning" to Greek.', "Give me a pasta recipe.", "What time is it in Tokyo?", "How do I fix my bicycle?", "Suggest a name for my dog.",
                  "What is the best restaurant nearby?", "Tell me a story about a dragon.", "How tall is Mount Everest?", "Play some music.", "What should I wear tomorrow?",
                  "Who is the president of Brazil?", "How do I lose weight fast?", "Give me tips for a job interview.", "Calculate 45 divided by 5.", "Will it rain this weekend?"],
}
TEST_TEMPLATES: Dict[str, List[str]] = {
    "graph_lookup": ['Name the authors of the article "{t}".', 'What is the citation count of "{t}"?', 'Show me what "{t}" references.',
                     'Which works cite "{t}" as well as "{t2}"?', "Papers since {y} that use {c}?", "Which paper got the most citations in {y}?",
                     "Who has co-authored the most with {n}?"],
    "topic_search": ["Give me an overview of {c}.", "Recent work on {k}?", "How do researchers apply {c} to {k}?", "Tell me about {k} using {c}."],
    "analysis": ["Contrast {c} with {c2} on {k}.", "How did research on {c} change since {y}?", "Which researchers bridge {c} and {c2}?", "Why did {c} replace {c2}?"],
    "off_topic": ["How hot will it be tomorrow?", "Compose a short song about summer.", "What is 17 times 23?", "Recommend a movie for tonight."],
}


def canonicalize(text: str) -> str:
    """Mask the parts that vary from question to question: quoted strings, numbers and capitalised names after 'of'/'with'."""
    t = re.sub(r"[\"“”][^\"“”]+[\"“”]", '"<T>"', text)
    t = re.sub(r"\d+", "<N>", t)
    return t.lower()


def make_intent_data(titles: Sequence[str], names: Sequence[str], concepts: Sequence[str], tasks: Sequence[str], n_per_class: int = 200,
                     seed: int = 0, split: str = "train") -> Tuple[List[str], List[str]]:
    rng = np.random.default_rng(seed)
    templates = TRAIN_TEMPLATES if split == "train" else TEST_TEMPLATES
    X, y = [], []
    pick = lambda xs: str(xs[int(rng.integers(len(xs)))])  # noqa: E731
    for label, tpls in templates.items():
        for i in range(n_per_class):
            tpl = tpls[i % len(tpls)]
            X.append(tpl.format(t=pick(titles), t2=pick(titles), n=pick(names), c=pick(concepts), c2=pick(concepts), k=pick(tasks), y=int(rng.integers(2015, 2026))))
            y.append(label)
    return X, y


@dataclass
class Intent:
    label: str
    confidence: float


class IntentClassifier:
    def __init__(self, min_confidence: float = 0.55, fallback: str = "analysis", encoder=None):
        """``encoder=None``: lexical features (TF-IDF). With a sentence encoder (e.g. ``HFEncoder``) the features are semantic embeddings,
        which generalise far better to phrasings and vocabulary never seen in training (measured in ``tests/test_system1.py``)."""
        self.min_confidence, self.fallback, self.encoder = min_confidence, fallback, encoder
        self.vec = None
        self.clf = None

    def _features(self, texts: Sequence[str], fit: bool = False):
        if self.encoder is not None:
            return self.encoder.encode([canonicalize(t) for t in texts])
        return self.vec.fit_transform(texts) if fit else self.vec.transform(texts)

    def fit(self, texts: Sequence[str], labels: Sequence[str]) -> "IntentClassifier":
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import FeatureUnion
        if self.encoder is not None:
            self.clf = LogisticRegression(C=20.0, max_iter=2000).fit(self._features(texts), list(labels))
            return self
        self.vec = FeatureUnion([("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True, preprocessor=canonicalize)),
                                 ("word", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, preprocessor=canonicalize))])
        X = self.vec.fit_transform(texts)
        self.clf = LogisticRegression(C=20.0, max_iter=2000).fit(X, list(labels))
        return self

    def predict(self, text: str) -> Intent:
        p = self.clf.predict_proba(self._features([text]))[0]
        i = int(np.argmax(p))
        label, conf = str(self.clf.classes_[i]), float(p[i])
        if conf < self.min_confidence and label != "off_topic":
            return Intent(self.fallback, conf)                  # unsure: use the most capable route rather than guessing
        return Intent(label, conf)

    def evaluate(self, texts: Sequence[str], labels: Sequence[str]) -> Dict[str, float]:
        pred = [self.predict(t).label for t in texts]
        acc = float(np.mean([p == y for p, y in zip(pred, labels)]))
        per = {c: float(np.mean([p == y for p, y in zip(pred, labels) if y == c])) for c in sorted(set(labels))}
        return {"accuracy": acc, **{f"recall_{c}": v for c, v in per.items()}}


def train_default(store, n_per_class: int = 300, seed: int = 0, encoder=None) -> IntentClassifier:
    """Train on templates filled with the graph's own titles, author names, concepts and task names."""
    from ..graph.concepts import load_lexicon
    papers = list(store.papers["title"].head(500))
    names = [n for n in store.authors["name"].head(500) if isinstance(n, str)]
    lex = load_lexicon()
    concepts = sorted(set(store._by_concept))[:60] or list(lex["method"])
    tasks = list(lex["task"])
    X, y = make_intent_data(papers, names, concepts, tasks, n_per_class, seed, "train")
    return IntentClassifier(encoder=encoder).fit(X, y)

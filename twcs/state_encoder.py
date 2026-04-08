import re
from typing import List, Dict, Tuple
import numpy as np


class StateEncoder:
    def __init__(self):
        self.intent_keywords = self._build_intent_keywords()
        self.sentiment_patterns = self._build_sentiment_patterns()
        self.urgency_patterns = self._build_urgency_patterns()
        self.resolution_indicators = self._build_resolution_indicators()
        self.frustration_indicators = self._build_frustration_indicators()

        self.intent_to_idx = {k: i for i, k in enumerate(self.intent_keywords.keys())}
        self.state_dim = len(self.intent_keywords) + 3 + 1 + 1 + 1

    def _build_intent_keywords(self) -> Dict[str, List[str]]:
        return {
            "billing_issue": [
                "bill",
                "charge",
                "payment",
                "charged",
                "pay",
                "invoice",
                "price",
                "cost",
                "fee",
                "refund",
            ],
            "service_issue": [
                "service",
                "internet",
                "wifi",
                "connection",
                "network",
                "signal",
                "data",
                "speed",
                "slow",
            ],
            "account_issue": [
                "account",
                "login",
                "password",
                "username",
                "verify",
                "authentication",
                "account",
                "serial",
            ],
            "technical_support": [
                "error",
                "not working",
                "broken",
                "fix",
                "help",
                "troubleshoot",
                "issue",
                "problem",
                "device",
            ],
            "order_issue": [
                "order",
                "shipping",
                "delivery",
                "package",
                "arrived",
                "received",
                "missing",
                "wrong",
            ],
            "general_inquiry": [
                "question",
                "how",
                "can you",
                "want to",
                "need to",
                "would like",
                "please",
            ],
            "complaint": [
                "worst",
                "terrible",
                "awful",
                "horrible",
                "rude",
                "unacceptable",
                "frustrated",
                "angry",
            ],
            "praise": [
                "thank",
                "great",
                "awesome",
                "amazing",
                "best",
                "love",
                "excellent",
                "wonderful",
            ],
        }

    def _build_sentiment_patterns(self) -> Dict[str, Dict[str, List[str]]]:
        return {
            "positive": {
                "words": [
                    "thank",
                    "thanks",
                    "great",
                    "awesome",
                    "amazing",
                    "love",
                    "excellent",
                    "wonderful",
                    "perfect",
                    "best",
                    "good",
                    "helpful",
                    "appreciate",
                    "resolved",
                    "fixed",
                ],
                "patterns": [
                    r"\bthx\b",
                    r"\bthanx\b",
                    r"\btysm\b",
                    r":\)",
                    r":D",
                    r";\)",
                    r"\bawesome\b",
                    r"\bamazing\b",
                ],
            },
            "negative": {
                "words": [
                    "worst",
                    "terrible",
                    "awful",
                    "horrible",
                    "bad",
                    "hate",
                    "angry",
                    "frustrated",
                    "annoyed",
                    "upset",
                    "furious",
                    "disappointed",
                    "several",
                    "as usual",
                    "nobody",
                    "nothing",
                    "ridiculous",
                    "unacceptable",
                    "sick",
                    "tired",
                ],
                "patterns": [
                    r":\(",
                    r":/",
                    r">\(",
                    r"😩",
                    r"😡",
                    r"🤬",
                    r"😭",
                    r"\bworst\b",
                    r"\bterrible\b",
                    r"\bawful\b",
                ],
            },
            "neutral": {
                "words": [
                    "help",
                    "need",
                    "want",
                    "would",
                    "could",
                    "please",
                    "ask",
                    "wondering",
                ],
                "patterns": [],
            },
        }

    def _build_urgency_patterns(self) -> Dict[str, List[str]]:
        return {
            "high": [
                "immediately",
                "right now",
                "asap",
                "urgent",
                "emergency",
                "now",
                "waiting",
                "can't wait",
                "need help now",
                "please help",
                "help me",
                "immediately",
                "stuck",
                "critical",
                "how do you propose",
            ],
            "medium": [
                "soon",
                "whenever",
                "today",
                "this week",
                "quickly",
                "as soon as possible",
            ],
            "low": [
                "when you can",
                "when possible",
                "sometime",
                "eventually",
                "no rush",
            ],
        }

    def _build_resolution_indicators(self) -> List[str]:
        return [
            "resolved",
            "fixed",
            "solved",
            "done",
            "complete",
            "helped",
            "thank",
            "thanks",
            "appreciate",
            "great",
            "awesome",
            "perfect",
            "works now",
            "all set",
            "good now",
        ]

    def _build_frustration_indicators(self) -> List[str]:
        return [
            "worst",
            "terrible",
            "awful",
            "horrible",
            "hate",
            "angry",
            "frustrated",
            "annoyed",
            "ridiculous",
            "unacceptable",
            "sick of",
            "tired of",
            "fed up",
            "never again",
            "wrong number",
            "no one helps",
            "no response",
            "no one is responding",
            "been waiting",
            "still waiting",
            "several",
            "as usual",
            "nobody",
        ]

    def encode(self, text: str) -> np.ndarray:
        state = np.zeros(self.state_dim)

        text_lower = text.lower()

        intent_scores = self._encode_intent(text_lower)
        for intent, score in intent_scores.items():
            idx = self.intent_to_idx.get(intent)
            if idx is not None:
                state[idx] = score

        sentiment_score = self._encode_sentiment(text_lower)
        state[len(self.intent_keywords)] = sentiment_score

        urgency_score = self._encode_urgency(text_lower)
        state[len(self.intent_keywords) + 1] = urgency_score

        is_frustrated = self._detect_frustration(text_lower)
        state[len(self.intent_keywords) + 2] = 1.0 if is_frustrated else 0.0

        is_resolved = self._detect_resolution(text_lower)
        state[len(self.intent_keywords) + 3] = 1.0 if is_resolved else 0.0

        return state

    def _encode_intent(self, text: str) -> Dict[str, float]:
        scores = {}
        for intent, keywords in self.intent_keywords.items():
            score = sum(1 for kw in keywords if kw in text) / len(keywords)
            scores[intent] = min(score * 2, 1.0)
        return scores

    def _encode_sentiment(self, text: str) -> float:
        pos_count = sum(
            1 for w in self.sentiment_patterns["positive"]["words"] if w in text
        )
        neg_count = sum(
            1 for w in self.sentiment_patterns["negative"]["words"] if w in text
        )

        for pattern in self.sentiment_patterns["positive"]["patterns"]:
            if re.search(pattern, text):
                pos_count += 1
        for pattern in self.sentiment_patterns["negative"]["patterns"]:
            if re.search(pattern, text):
                neg_count += 1

        if pos_count + neg_count == 0:
            return 0.0
        return (pos_count - neg_count) / (pos_count + neg_count + 1)

    def _encode_urgency(self, text: str) -> float:
        high_count = sum(1 for kw in self.urgency_patterns["high"] if kw in text)
        med_count = sum(1 for kw in self.urgency_patterns["medium"] if kw in text)
        low_count = sum(1 for kw in self.urgency_patterns["low"] if kw in text)

        if high_count > 0:
            return 1.0
        elif med_count > 0:
            return 0.5
        elif low_count > 0:
            return 0.25
        return 0.1

    def _detect_frustration(self, text: str) -> bool:
        return any(indicator in text for indicator in self.frustration_indicators)

    def _detect_resolution(self, text: str) -> bool:
        return any(indicator in text for indicator in self.resolution_indicators)

    def get_state_dimension(self) -> int:
        return self.state_dim

    def get_state_components(self, state: np.ndarray) -> Dict[str, float]:
        components = {}

        for intent, idx in self.intent_to_idx.items():
            components[f"intent_{intent}"] = state[idx]

        components["sentiment"] = state[len(self.intent_keywords)]
        components["urgency"] = state[len(self.intent_keywords) + 1]
        components["frustrated"] = state[len(self.intent_keywords) + 2]
        components["resolved"] = state[len(self.intent_keywords) + 3]

        return components


def test_encoder():
    encoder = StateEncoder()
    print(f"State dimension: {encoder.get_state_dimension()}")

    test_texts = [
        "@sprintcare I have sent several private messages and no one is responding as usual",
        "@VerizonSupport I finally got someone that helped me, thanks!",
        "@sprintcare is the worst customer service",
        "@VerizonSupport My picture on @Ask_Spectrum pretty much every day. Why should I pay $171 per month?",
    ]

    for text in test_texts:
        state = encoder.encode(text)
        components = encoder.get_state_components(state)
        print(f"\nText: {text[:60]}...")
        print(
            f"  Sentiment: {components['sentiment']:.2f}, Urgency: {components['urgency']:.2f}"
        )
        print(
            f"  Frustrated: {components['frustrated']:.2f}, Resolved: {components['resolved']:.2f}"
        )


if __name__ == "__main__":
    test_encoder()

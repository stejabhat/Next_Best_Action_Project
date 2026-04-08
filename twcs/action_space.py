import re
from typing import List, Dict, Tuple, Optional
import numpy as np


class ActionSpace:
    def __init__(self):
        self.action_types = [
            "request_dm",
            "apologize",
            "empathize",
            "provide_info",
            "ask_question",
            "resolve_issue",
            "escalate",
            "redirect",
            "request_info",
            "greeting",
            "closing",
            "acknowledge",
        ]

        self.action_patterns = self._build_action_patterns()
        self.action_to_idx = {a: i for i, a in enumerate(self.action_types)}
        self.action_dim = len(self.action_types)

        self.response_templates = self._build_response_templates()

    def _build_action_patterns(self) -> Dict[str, List[str]]:
        return {
            "request_dm": [
                r"\bprivate message\b",
                r"\bdm\b",
                r"\bshoot us a\b",
                r"\bmessage\b",
                r"\bdirectly\b",
                r"\bchat\b",
                r"\bsecure\b",
                r"\bfollow and\b",
            ],
            "apologize": [
                r"\bsorry\b",
                r"\bapologi[zs]e\b",
                r"\bforgive\b",
                r"\bmy mistake\b",
                r"\bweapologize\b",
                r"\bapologizes\b",
            ],
            "empathize": [
                r"\bunderstand\b",
                r"\bfrustration\b",
                r"\bsaddening\b",
                r"\bconcern\b",
                r"\bfrustrating\b",
                r"\bhow can we help\b",
                r"\bhelp has arrived\b",
            ],
            "provide_info": [
                r"\bhere\'s\b",
                r"\bhere is\b",
                r"\bwe can\b",
                r"\bwill\b",
                r"\bable to\b",
                r"\boptions\b",
                r"\bsolution\b",
                r"\bfix\b",
                r"\bresolve\b",
            ],
            "ask_question": [
                r"\bwhat\b",
                r"\bhow\b",
                r"\bwhen\b",
                r"\bwhere\b",
                r"\bwhy\b",
                r"\bcould you\b",
                r"\bwould you\b",
                r"\bcan you\b",
                r"\bdo you\b",
            ],
            "resolve_issue": [
                r"\bresolved\b",
                r"\bfixed\b",
                r"\bsolved\b",
                r"\bhelped\b",
                r"\btaken care\b",
                r"\ball set\b",
                r"\btaken care of\b",
                r"\bno longer\b",
            ],
            "escalate": [
                r"\bsupervisor\b",
                r"\bmanager\b",
                r"\bspecialist\b",
                r"\bteam\b",
                r"\bescalat\b",
                r"\btransfer\b",
                r"\breach out\b",
                r"\bfurther\b",
            ],
            "redirect": [
                r"\bwebsite\b",
                r"\bapp\b",
                r"\bcall\b",
                r"\bstore\b",
                r"\boffice\b",
                r"\bphone\b",
                r"\b1-800\b",
                r"\b\.com\b",
            ],
            "request_info": [
                r"\bacct\b",
                r"\baccount\b",
                r"\bphone\b",
                r"\bemail\b",
                r"\bname\b",
                r"\baddress\b",
                r"\bnumber\b",
                r"\bid\b",
                r"\bprovide\b",
            ],
            "greeting": [
                r"\bhello\b",
                r"\bhi\b",
                r"\bhey\b",
                r"\bgood (morning|afternoon|evening)\b",
                r"\bhow can i\b",
                r"\bhow may i\b",
                r"\bwhat can i\b",
            ],
            "closing": [
                r"\bthank you\b",
                r"\bthanks\b",
                r"\bappreciate\b",
                r"\bcontact us\b",
                r"\btweet\b",
                r"\bjust a tweet away\b",
                r"\blet us know\b",
                r"\banything else\b",
            ],
            "acknowledge": [
                r"\bgot it\b",
                r"\bunderstand\b",
                r"\bsee\b",
                r"\bknow\b",
                r"\bnoted\b",
                r"\breceived\b",
                r"\bhear\b",
            ],
        }

    def _build_response_templates(self) -> Dict[str, str]:
        return {
            "request_dm": "I'd like to help you further. Please send us a DM so we can assist you privately.",
            "apologize": "I apologize for any frustration or inconvenience this has caused.",
            "empathize": "I understand this is frustrating. Let me help you resolve this.",
            "provide_info": "Here's what I can do to help...",
            "ask_question": "Could you provide more details about...",
            "resolve_issue": "Great! I'm glad we could resolve this for you.",
            "escalate": "Let me connect you with a specialist who can help further.",
            "redirect": "You can find more information at our website or by calling...",
            "request_info": "To help you better, I'll need your account information.",
            "greeting": "Hello! How can I help you today?",
            "closing": "Is there anything else I can help you with today?",
            "acknowledge": "I understand. Let me look into that for you.",
        }

    def encode(self, text: str) -> np.ndarray:
        action_vec = np.zeros(self.action_dim)

        text_lower = text.lower()

        for action, patterns in self.action_patterns.items():
            action_idx = self.action_to_idx[action]
            score = 0
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    score += 1
            action_vec[action_idx] = min(score, 1.0)

        if action_vec.sum() == 0:
            action_vec[self.action_to_idx["provide_info"]] = 0.5

        return action_vec

    def decode(self, action_vec: np.ndarray) -> List[Tuple[str, float]]:
        results = []
        for action, idx in self.action_to_idx.items():
            if action_vec[idx] > 0.1:
                results.append((action, float(action_vec[idx])))

        results.sort(key=lambda x: x[1], reverse=True)
        return results

    def get_all_actions(self) -> List[str]:
        return self.action_types

    def get_action_count(self) -> int:
        return self.action_dim

    def get_template(self, action: str) -> Optional[str]:
        return self.response_templates.get(action)


class ActionLearner:
    def __init__(self, action_space: ActionSpace, state_encoder_dim: int):
        self.action_space = action_space
        self.state_dim = state_encoder_dim
        self.action_dim = action_space.get_action_count()

        self.transition_matrix = self._initialize_transition_matrix()
        self.action_effects = self._initialize_action_effects()
        self.resolution_stats = self._initialize_resolution_stats()

    def _initialize_transition_matrix(self) -> np.ndarray:
        return np.random.randn(self.state_dim, self.state_dim) * 0.1

    def _initialize_action_effects(self) -> Dict[str, np.ndarray]:
        effects = {}
        for action in self.action_space.get_all_actions():
            effects[action] = np.random.randn(self.state_dim) * 0.1
        return effects

    def _initialize_resolution_stats(self) -> Dict[str, Dict[str, int]]:
        stats = {}
        for action in self.action_space.get_all_actions():
            stats[action] = {
                "resolved": 0,
                "unresolved": 0,
                "escalated": 0,
                "frustrated": 0,
            }
        return stats

    def learn_from_trajectory(
        self, states: List[np.ndarray], actions: List[np.ndarray], outcomes: List[str]
    ):
        if len(states) < 2:
            return

        for i in range(len(states) - 1):
            state_curr = states[i]
            state_next = states[i + 1]
            action = actions[i]

            delta = state_next - state_curr

            action_idx = np.argmax(action)
            if action_idx < len(self.action_space.get_all_actions()):
                action_name = self.action_space.get_all_actions()[action_idx]
                self.action_effects[action_name] += 0.1 * delta

    def update_resolution_stats(self, action: str, outcome: str):
        if action in self.resolution_stats:
            if outcome in self.resolution_stats[action]:
                self.resolution_stats[action][outcome] += 1

    def get_action_effect(self, action: str) -> np.ndarray:
        return self.action_effects.get(action, np.zeros(self.state_dim))

    def get_success_rate(self, action: str) -> float:
        stats = self.resolution_stats.get(action, {})
        total = sum(stats.values())
        if total == 0:
            return 0.5
        return stats.get("resolved", 0) / total

    def get_best_action_for_state(self, state: np.ndarray) -> Tuple[str, float]:
        scores = {}

        for action in self.action_space.get_all_actions():
            effect = self.action_effects[action]

            sentiment_improvement = effect[0] if len(effect) > 0 else 0
            resolution_effect = effect[-1] if len(effect) > 0 else 0

            score = sentiment_improvement + resolution_effect * 2

            scores[action] = score

        best_action = max(scores, key=scores.get)
        return best_action, scores[best_action]


def test_action_space():
    action_space = ActionSpace()
    print(f"Action dimension: {action_space.get_action_count()}")

    test_responses = [
        "Please send us a private message so that I can assist you further.",
        "I apologize for any inconvenience. Let me help you with this.",
        "I understand your frustration. How can I help?",
        "We can help resolve this issue for you today.",
        "What information is incorrect?",
        "Great! I'm glad we could help resolve this.",
    ]

    for text in test_responses:
        action_vec = action_space.encode(text)
        decoded = action_space.decode(action_vec)
        print(f"\nText: {text[:50]}...")
        print(f"  Actions: {decoded[:3]}")


if __name__ == "__main__":
    test_action_space()

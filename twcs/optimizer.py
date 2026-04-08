import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import copy


@dataclass
class ActionResult:
    action: str
    expected_cost: float
    expected_trajectory: List[np.ndarray]
    confidence: float
    reasoning: str


class ActionOptimizer:
    def __init__(self, dynamics, action_space, cost_function):
        self.dynamics = dynamics
        self.action_space = action_space
        self.cost_function = cost_function

    def evaluate_action(
        self, state: np.ndarray, action_name: str, simulate_steps: int = 3
    ) -> ActionResult:
        action_vec = np.zeros(self.action_space.get_action_count())
        action_idx = self.action_space.action_to_idx.get(action_name, 0)
        action_vec[action_idx] = 1.0

        states = [state.copy()]
        current_state = state.copy()
        total_cost = 0.0

        for step in range(simulate_steps):
            next_state = self.dynamics.predict_next_state(current_state, action_vec)

            outcome = self._estimate_outcome(current_state, next_state, action_name)
            step_cost = self.cost_function.compute(next_state, action_name, outcome)
            total_cost += step_cost

            states.append(next_state.copy())
            current_state = next_state

        confidence = self.dynamics.get_confidence()

        resolution_likelihood = self._estimate_resolution_likelihood(state, action_name)

        reasoning = self._generate_reasoning(state, action_name, resolution_likelihood)

        return ActionResult(
            action=action_name,
            expected_cost=total_cost,
            expected_trajectory=states,
            confidence=confidence,
            reasoning=reasoning,
        )

    def evaluate_all_actions(
        self, state: np.ndarray, top_k: int = 5
    ) -> List[ActionResult]:
        results = []

        for action_name in self.action_space.get_all_actions():
            result = self.evaluate_action(state, action_name)
            results.append(result)

        results.sort(key=lambda x: x.expected_cost)

        return results[:top_k]

    def find_optimal_action(self, state: np.ndarray) -> ActionResult:
        results = self.evaluate_all_actions(state, top_k=1)
        return results[0]

    def simulate_conversation(
        self, initial_state: np.ndarray, max_turns: int = 5
    ) -> Tuple[List[str], float]:
        conversation_actions = []
        current_state = initial_state.copy()
        total_cost = 0.0

        for turn in range(max_turns):
            best_result = self.find_optimal_action(current_state)

            conversation_actions.append(best_result.action)
            total_cost += best_result.expected_cost

            action_vec = np.zeros(self.action_space.get_action_count())
            action_idx = self.action_space.action_to_idx.get(best_result.action, 0)
            action_vec[action_idx] = 1.0

            current_state = self.dynamics.predict_next_state(current_state, action_vec)

            if current_state[-1] > 0.8:
                break

        return conversation_actions, total_cost

    def _estimate_outcome(self, state: np.ndarray, next_state: np.ndarray, action: str):
        from dynamics import ConversationOutcome

        resolved_before = state[-1] if len(state) > 0 else 0
        resolved_after = next_state[-1] if len(next_state) > 0 else 0

        sentiment_before = state[-3] if len(state) > 2 else 0
        sentiment_after = next_state[-3] if len(next_state) > 2 else 0

        resolved = resolved_after > resolved_before
        escalated = action == "escalate"
        frustrated = next_state[-2] > state[-2] if len(next_state) > 1 else False
        sentiment_improved = sentiment_after > sentiment_before

        return ConversationOutcome(
            resolved=resolved,
            escalated=escalated,
            frustrated=frustrated,
            sentiment_improved=sentiment_improved,
            turns_count=1,
            success=resolved or sentiment_improved,
        )

    def _estimate_resolution_likelihood(self, state: np.ndarray, action: str) -> float:
        base_rates = {
            "resolve_issue": 0.8,
            "empathize": 0.6,
            "apologize": 0.5,
            "provide_info": 0.5,
            "request_dm": 0.4,
            "ask_question": 0.4,
            "greeting": 0.3,
            "closing": 0.3,
            "escalate": 0.3,
            "redirect": 0.2,
            "request_info": 0.3,
            "acknowledge": 0.3,
        }

        base = base_rates.get(action, 0.3)

        urgency = state[-4] if len(state) > 3 else 0.1
        frustration = state[-2] if len(state) > 1 else 0

        urgency_modifier = 1.0 - urgency * 0.3
        frustration_modifier = 1.0 - frustration * 0.5

        return base * urgency_modifier * frustration_modifier

    def _generate_reasoning(
        self, state: np.ndarray, action: str, resolution_likelihood: float
    ) -> str:
        reasons = []

        sentiment = state[-3] if len(state) > 2 else 0
        if sentiment < -0.3:
            reasons.append("negative sentiment detected")

        urgency = state[-4] if len(state) > 3 else 0
        if urgency > 0.7:
            reasons.append("high urgency")

        frustration = state[-2] if len(state) > 1 else 0
        if frustration > 0.5:
            reasons.append("customer frustrated")

        reasons.append(f"resolution likelihood: {resolution_likelihood:.2f}")

        return "; ".join(reasons)


class DecisionIntelligenceSystem:
    def __init__(self, state_dim: int = 15):
        from state_encoder import StateEncoder
        from action_space import ActionSpace
        from dynamics import DynamicsModel, CostFunction

        self.state_encoder = StateEncoder()
        self.action_space = ActionSpace()
        self.dynamics = DynamicsModel(
            state_dim=state_dim, action_dim=self.action_space.get_action_count()
        )
        self.cost_function = CostFunction()
        self.optimizer = ActionOptimizer(
            self.dynamics, self.action_space, self.cost_function
        )

        self.conversation_history = []
        self.state_history = []

    def ingest_conversation_turn(self, text: str, is_agent: bool = False):
        state = self.state_encoder.encode(text)

        if not is_agent:
            self.conversation_history.append(("customer", text))
            self.state_history.append(state)
        else:
            self.conversation_history.append(("agent", text))

    def get_current_state(self) -> np.ndarray:
        if not self.state_history:
            return np.zeros(self.state_encoder.get_state_dimension())

        return self.state_history[-1].copy()

    def recommend_action(self) -> ActionResult:
        current_state = self.get_current_state()

        result = self.optimizer.find_optimal_action(current_state)

        return result

    def recommend_alternatives(self, top_k: int = 3) -> List[ActionResult]:
        current_state = self.get_current_state()

        results = self.optimizer.evaluate_all_actions(current_state, top_k=top_k)

        return results

    def simulate_outcome(self, action: str) -> Dict:
        current_state = self.get_current_state()

        result = self.optimizer.evaluate_action(current_state, action)

        trajectory = result.expected_trajectory

        final_state = trajectory[-1] if trajectory else current_state
        components = self.state_encoder.get_state_components(final_state)

        return {
            "action": action,
            "expected_cost": result.expected_cost,
            "trajectory_length": len(trajectory),
            "components": components,
            "reasoning": result.reasoning,
            "confidence": result.confidence,
        }

    def update_from_feedback(self, actual_outcome: str):
        if len(self.state_history) < 2:
            return

        state = self.state_history[-2]
        action_text = None
        for speaker, text in reversed(self.conversation_history):
            if speaker == "agent":
                action_text = text
                break

        if action_text is None:
            return

        action_vec = self.action_space.encode(action_text)

        from dynamics import ConversationOutcome

        outcome = ConversationOutcome(
            resolved=actual_outcome == "resolved",
            escalated=actual_outcome == "escalated",
            frustrated=actual_outcome == "frustrated",
            sentiment_improved=actual_outcome == "improved",
            turns_count=1,
            success=actual_outcome in ["resolved", "improved"],
        )

        next_state = self.state_history[-1]

        self.dynamics.update_from_observation(state, action_vec, next_state, outcome)

    def get_system_status(self) -> Dict:
        return {
            "conversation_length": len(self.conversation_history),
            "state_history_length": len(self.state_history),
            "dynamics_confidence": self.dynamics.get_confidence(),
            "current_state": self.get_current_state().tolist()[:5],
        }


def test_system():
    system = DecisionIntelligenceSystem()

    system.ingest_conversation_turn(
        "@sprintcare I have sent several private messages and no one is responding as usual"
    )

    print("Current state:", system.get_current_state()[:5])

    recommendation = system.recommend_action()
    print(f"\nRecommended action: {recommendation.action}")
    print(f"Expected cost: {recommendation.expected_cost:.3f}")
    print(f"Confidence: {recommendation.confidence:.3f}")
    print(f"Reasoning: {recommendation.reasoning}")

    alternatives = system.recommend_alternatives(top_k=3)
    print("\nAlternative actions:")
    for alt in alternatives:
        print(f"  {alt.action}: cost={alt.expected_cost:.3f}")

    system.ingest_conversation_turn(
        "Please send us a private message so that I can assist you further.",
        is_agent=True,
    )

    system.ingest_conversation_turn("@sprintcare and how do you propose we do that")

    recommendation2 = system.recommend_action()
    print(f"\nNext recommended action: {recommendation2.action}")


if __name__ == "__main__":
    test_system()

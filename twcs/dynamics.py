import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class ConversationOutcome:
    resolved: bool
    escalated: bool
    frustrated: bool
    sentiment_improved: bool
    turns_count: int
    success: bool


class CostFunction:
    def __init__(self):
        self.frustration_weight = 2.0
        self.escalation_weight = 1.5
        self.unresolved_weight = 1.0
        self.resolution_reward = 3.0
        self.sentiment_improvement_reward = 1.5
        self.length_penalty = 0.1

    def compute(
        self, state: np.ndarray, action: str, outcome: ConversationOutcome
    ) -> float:
        cost = 0.0

        frustration = state[-2] if len(state) > 1 else 0
        cost += frustration * self.frustration_weight

        if outcome.escalated:
            cost += self.escalation_weight

        if not outcome.resolved and outcome.success:
            cost += self.unresolved_weight

        if outcome.resolved:
            cost -= self.resolution_reward

        if outcome.sentiment_improved:
            cost -= self.sentiment_improvement_reward

        cost += outcome.turns_count * self.length_penalty

        return cost

    def compute_trajectory_cost(
        self, states: List[np.ndarray], actions: List[str], outcome: ConversationOutcome
    ) -> float:
        total_cost = 0.0

        for state in states:
            frustration = state[-2] if len(state) > 1 else 0
            total_cost += frustration * self.frustration_weight * 0.1

        if outcome.escalated:
            total_cost += self.escalation_weight

        if not outcome.resolved:
            total_cost += self.unresolved_weight

        if outcome.resolved:
            total_cost -= self.resolution_reward

        if outcome.sentiment_improved:
            total_cost -= self.sentiment_improvement_reward

        total_cost += len(states) * self.length_penalty

        return total_cost

    def get_reward(self, outcome: ConversationOutcome) -> float:
        reward = 0.0

        if outcome.resolved:
            reward += self.resolution_reward

        if outcome.sentiment_improved:
            reward += self.sentiment_improvement_reward

        reward -= outcome.turns_count * self.length_penalty

        return reward


class DynamicsModel:
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 32):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        self.W_state = np.random.randn(state_dim, state_dim) * 0.1
        self.W_action = np.random.randn(state_dim, action_dim) * 0.1
        self.b = np.random.randn(state_dim) * 0.1

        self.learned_effects = {}
        self.confidence_scores = {}

    def predict_next_state(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        state_term = self.W_state @ state
        action_term = self.W_action @ action
        delta = state_term + action_term + self.b

        next_state = state + 0.3 * np.tanh(delta)

        next_state = np.clip(next_state, -2, 2)

        return next_state

    def simulate_trajectory(
        self,
        initial_state: np.ndarray,
        action_sequence: List[np.ndarray],
        steps: int = 5,
    ) -> List[np.ndarray]:
        states = [initial_state.copy()]
        current_state = initial_state.copy()

        for i in range(steps):
            if i < len(action_sequence):
                action = action_sequence[i]
            else:
                action = np.zeros(self.action_dim)

            current_state = self.predict_next_state(current_state, action)
            states.append(current_state.copy())

        return states

    def train(
        self,
        state_deltas: List[np.ndarray],
        state_action_pairs: List[Tuple[np.ndarray, np.ndarray]],
        learning_rate: float = 0.01,
    ):
        if not state_action_pairs:
            return

        total_loss = 0.0

        for delta, (state, action) in zip(state_deltas, state_action_pairs):
            predicted_delta = self.W_state @ state + self.W_action @ action + self.b

            loss = np.mean((delta - predicted_delta) ** 2)
            total_loss += loss

            grad_state = -2 * (delta - predicted_delta)
            grad_action = -2 * (delta - predicted_delta)

            self.W_state -= learning_rate * np.outer(grad_state, state)
            self.W_action -= learning_rate * np.outer(grad_action, action)
            self.b -= learning_rate * grad_state

        return total_loss / len(state_deltas)

    def update_from_observation(
        self,
        state: np.ndarray,
        action: np.ndarray,
        next_state: np.ndarray,
        outcome: ConversationOutcome,
    ):
        delta = next_state - state

        action_idx = np.argmax(action)

        if action_idx not in self.learned_effects:
            self.learned_effects[action_idx] = []

        self.learned_effects[action_idx].append(delta)

        if len(self.learned_effects[action_idx]) > 100:
            self.learned_effects[action_idx] = self.learned_effects[action_idx][-100:]

        effect = np.mean(self.learned_effects[action_idx], axis=0)

        confidence = min(len(self.learned_effects[action_idx]) / 50.0, 1.0)
        self.confidence_scores[action_idx] = confidence

        alpha = 0.1 * confidence
        self.W_action[:, action_idx] = (1 - alpha) * self.W_action[
            :, action_idx
        ] + alpha * effect

    def get_confidence(self) -> float:
        if not self.confidence_scores:
            return 0.0
        return np.mean(list(self.confidence_scores.values()))


class TrajectorySimulator:
    def __init__(
        self, dynamics: DynamicsModel, action_space, cost_function: CostFunction
    ):
        self.dynamics = dynamics
        self.action_space = action_space
        self.cost_function = cost_function

    def simulate_action(
        self, state: np.ndarray, action_name: str
    ) -> Tuple[np.ndarray, float]:
        action_vec = np.zeros(self.action_space.get_action_count())
        action_idx = self.action_space.action_to_idx.get(action_name, 0)
        action_vec[action_idx] = 1.0

        next_state = self.dynamics.predict_next_state(state, action_vec)

        outcome = self._estimate_outcome(state, next_state, action_name)
        cost = self.cost_function.compute(next_state, action_name, outcome)

        return next_state, cost

    def simulate_sequence(
        self, initial_state: np.ndarray, action_names: List[str]
    ) -> Tuple[List[np.ndarray], float]:
        states = [initial_state.copy()]
        current_state = initial_state.copy()
        total_cost = 0.0

        for action_name in action_names:
            next_state, cost = self.simulate_action(current_state, action_name)
            states.append(next_state)
            total_cost += cost
            current_state = next_state

        return states, total_cost

    def _estimate_outcome(
        self, state: np.ndarray, next_state: np.ndarray, action: str
    ) -> ConversationOutcome:
        resolved_before = state[-1] if len(state) > 0 else 0
        resolved_after = next_state[-1] if len(next_state) > 0 else 0

        sentiment_before = state[-3] if len(state) > 2 else 0
        sentiment_after = next_state[-3] if len(next_state) > 2 else 0

        resolved = resolved_after > resolved_before
        escalated = action == "escalate"
        frustrated = next_state[-2] > state[-2] if len(next_state) > 1 else False
        sentiment_improved = sentiment_after > sentiment_before

        success = resolved or sentiment_improved

        return ConversationOutcome(
            resolved=resolved,
            escalated=escalated,
            frustrated=frustrated,
            sentiment_improved=sentiment_improved,
            turns_count=1,
            success=success,
        )


def test_dynamics():
    state_dim = 15
    action_dim = 12

    dynamics = DynamicsModel(state_dim, action_dim)
    cost_function = CostFunction()

    print(f"State dim: {state_dim}, Action dim: {action_dim}")

    test_state = np.random.randn(state_dim)
    test_action = np.random.randn(action_dim)

    next_state = dynamics.predict_next_state(test_state, test_action)
    print(f"Next state norm: {np.linalg.norm(next_state):.3f}")

    outcome = ConversationOutcome(
        resolved=True,
        escalated=False,
        frustrated=False,
        sentiment_improved=True,
        turns_count=3,
        success=True,
    )

    cost = cost_function.compute(test_state, "resolve_issue", outcome)
    print(f"Cost: {cost:.3f}")


if __name__ == "__main__":
    test_dynamics()

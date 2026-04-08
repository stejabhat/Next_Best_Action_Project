import numpy as np
from typing import List, Dict, Tuple, Optional
import sys


class DecisionIntelligenceEngine:
    def __init__(self):
        from state_encoder import StateEncoder
        from action_space import ActionSpace, ActionLearner
        from dynamics import DynamicsModel, CostFunction, TrajectorySimulator
        from optimizer import ActionOptimizer, DecisionIntelligenceSystem

        self.state_encoder = StateEncoder()
        self.action_space = ActionSpace()
        self.action_learner = ActionLearner(
            self.action_space, self.state_encoder.get_state_dimension()
        )
        self.dynamics = DynamicsModel(
            state_dim=self.state_encoder.get_state_dimension(),
            action_dim=self.action_space.get_action_count(),
        )
        self.cost_function = CostFunction()
        self.trajectory_simulator = TrajectorySimulator(
            self.dynamics, self.action_space, self.cost_function
        )
        self.optimizer = ActionOptimizer(
            self.dynamics, self.action_space, self.cost_function
        )

        self.conversation_context = []
        self.current_state = None

    def infer_state(self, text: str) -> np.ndarray:
        state = self.state_encoder.encode(text)
        self.current_state = state
        return state

    def infer_state_from_history(self, conversation_history: List[Dict]) -> np.ndarray:
        if not conversation_history:
            return np.zeros(self.state_encoder.get_state_dimension())

        states = []
        for turn in conversation_history:
            text = turn.get("text", "")
            if text:
                state = self.state_encoder.encode(text)
                states.append(state)

        if not states:
            return np.zeros(self.state_encoder.get_state_dimension())

        current_state = states[-1]

        if len(states) > 1:
            momentum = states[-1] - states[-2]
            current_state = current_state + 0.1 * momentum

        self.current_state = current_state
        return current_state

    def compute_dynamics(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        return self.dynamics.predict_next_state(state, action)

    def simulate_trajectory(
        self, state: np.ndarray, action: str, steps: int = 3
    ) -> List[np.ndarray]:
        action_vec = np.zeros(self.action_space.get_action_count())
        action_idx = self.action_space.action_to_idx.get(action, 0)
        action_vec[action_idx] = 1.0

        return self.dynamics.simulate_trajectory(state, [action_vec] * steps, steps)

    def compute_cost(self, state: np.ndarray, action: str, outcome: Dict) -> float:
        from dynamics import ConversationOutcome

        conv_outcome = ConversationOutcome(
            resolved=outcome.get("resolved", False),
            escalated=outcome.get("escalated", False),
            frustrated=outcome.get("frustrated", False),
            sentiment_improved=outcome.get("sentiment_improved", False),
            turns_count=outcome.get("turns_count", 1),
            success=outcome.get("success", False),
        )
        return self.cost_function.compute(state, action, conv_outcome)

    def select_optimal_action(self) -> Tuple[str, float, List[np.ndarray], float]:
        if self.current_state is None:
            return ("greeting", 0.5, [], 0.0)

        result = self.optimizer.find_optimal_action(self.current_state)

        return (
            result.action,
            result.expected_cost,
            result.expected_trajectory,
            result.confidence,
        )

    def rank_alternative_actions(self, top_k: int = 5) -> List[Dict]:
        if self.current_state is None:
            return []

        results = self.optimizer.evaluate_all_actions(self.current_state, top_k=top_k)

        return [
            {
                "action": r.action,
                "expected_cost": float(r.expected_cost),
                "trajectory_length": len(r.expected_trajectory),
                "confidence": float(r.confidence),
                "reasoning": r.reasoning,
            }
            for r in results
        ]

    def train_from_data(self, trajectories: List[Dict], outcomes: List[Dict]):
        for traj, outcome in zip(trajectories, outcomes):
            states = []
            actions = []

            for turn in traj.get("customer_turns", []):
                for msg in turn:
                    state = self.state_encoder.encode(msg["text"])
                    states.append(state)

            for turn in traj.get("agent_turns", []):
                for msg in turn:
                    action = self.action_space.encode(msg["text"])
                    actions.append(action)

            if len(states) >= 2 and len(actions) >= 1:
                for i in range(len(states) - 1):
                    delta = states[i + 1] - states[i]
                    if i < len(actions):
                        self.dynamics.update_from_observation(
                            states[i],
                            actions[i],
                            states[i + 1],
                            self._outcome_to_conversation(outcome),
                        )

    def _outcome_to_conversation(self, outcome: Dict):
        from dynamics import ConversationOutcome

        return ConversationOutcome(
            resolved=outcome.get("resolved", False),
            escalated=outcome.get("escalated", False),
            frustrated=outcome.get("frustrated", False),
            sentiment_improved=outcome.get("sentiment_improved", False),
            turns_count=outcome.get("turns_count", 1),
            success=outcome.get("success", False),
        )


def initialize_system():
    engine = DecisionIntelligenceEngine()

    print("=" * 60)
    print("Continuous-Time Decision Intelligence System")
    print("=" * 60)
    print(f"State dimension: {engine.state_encoder.get_state_dimension()}")
    print(f"Action dimension: {engine.action_space.get_action_count()}")
    print(f"Action types: {engine.action_space.get_all_actions()}")
    print("=" * 60)

    return engine


def demonstrate_inference(engine: DecisionIntelligenceEngine, conversation: List[Dict]):
    print("\n--- Conversation State Inference ---")

    for turn in conversation:
        text = turn.get("text", "")
        speaker = turn.get("speaker", "customer")

        state = engine.infer_state(text)
        components = engine.state_encoder.get_state_components(state)

        print(f"\n[{speaker.upper()}] {text[:80]}...")
        print(
            f"  Sentiment: {components['sentiment']:.2f}, Urgency: {components['urgency']:.2f}"
        )
        print(
            f"  Frustrated: {components['frustrated']:.2f}, Resolved: {components['resolved']:.2f}"
        )

    print("\n--- Optimal Action Selection ---")
    optimal_action, cost, trajectory, confidence = engine.select_optimal_action()
    print(f"Optimal action: {optimal_action}")
    print(f"Expected cost: {cost:.3f}")
    print(f"Confidence: {confidence:.3f}")
    print(f"Trajectory length: {len(trajectory)}")

    print("\n--- Alternative Actions Ranked ---")
    alternatives = engine.rank_alternative_actions(top_k=5)
    for i, alt in enumerate(alternatives):
        print(
            f"  {i + 1}. {alt['action']}: cost={alt['expected_cost']:.3f}, confidence={alt['confidence']:.3f}"
        )
        print(f"     Reasoning: {alt['reasoning']}")


def demonstrate_trajectory_simulation(
    engine: DecisionIntelligenceEngine, state: np.ndarray
):
    print("\n--- Trajectory Simulation ---")

    test_actions = ["empathize", "apologize", "resolve_issue", "request_dm"]

    for action in test_actions:
        trajectory = engine.simulate_trajectory(state, action, steps=3)
        print(f"\nAction: {action}")

        for i, s in enumerate(trajectory):
            components = engine.state_encoder.get_state_components(s)
            print(
                f"  Step {i}: sentiment={components['sentiment']:.2f}, resolved={components['resolved']:.2f}"
            )


def main():
    engine = initialize_system()

    sample_conversation = [
        {
            "text": "@sprintcare I have sent several private messages and no one is responding as usual",
            "speaker": "customer",
        },
        {
            "text": "@115712 Please send us a Private Message so that we can further assist you.",
            "speaker": "agent",
        },
        {"text": "@sprintcare I did.", "speaker": "customer"},
        {
            "text": "@115712 Can you please send us a private message, so that I can gain further details?",
            "speaker": "agent",
        },
        {
            "text": "@sprintcare and how do you propose we do that",
            "speaker": "customer",
        },
    ]

    demonstrate_inference(engine, sample_conversation)

    state = engine.infer_state(sample_conversation[-1]["text"])
    demonstrate_trajectory_simulation(engine, state)

    print("\n" + "=" * 60)
    print("Decision Intelligence System Ready")
    print("=" * 60)


if __name__ == "__main__":
    main()

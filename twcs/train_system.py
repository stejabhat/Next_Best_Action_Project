import numpy as np
from typing import List, Dict, Tuple
import sys
from data_loader import DataLoader
from state_encoder import StateEncoder
from action_space import ActionSpace
from dynamics import DynamicsModel, CostFunction, ConversationOutcome
from optimizer import ActionOptimizer


class DecisionIntelligenceSystem:
    def __init__(self, state_dim: int = 14):
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
        self.trained = False

    def train(self, csv_path: str, max_convs: int = 5000):
        print(f"Loading data from {csv_path}...")
        loader = DataLoader(csv_path)
        loader.load(max_tweets=100000)
        print(f"Loaded {len(loader.tweets)} tweets")

        print("Building conversations...")
        loader.build_conversations(min_turns=3)
        print(f"Found {len(loader.conversations)} conversations")

        trajectories = loader.get_trajectories(max_convs=max_convs)
        print(f"Extracted {len(trajectories)} trajectories")

        print("Training dynamics model...")

        for i, traj in enumerate(trajectories):
            if i % 500 == 0:
                print(f"  Processing trajectory {i}/{len(trajectories)}")

            customer_turns = traj["customer_turns"]
            agent_turns = traj["agent_turns"]

            states = []
            actions = []
            outcomes = []

            for j, customer_msg in enumerate(customer_turns):
                for msg in customer_msg:
                    state = self.state_encoder.encode(msg["text"])
                    states.append(state)

            for j, agent_msg in enumerate(agent_turns):
                for msg in agent_msg:
                    action = self.action_space.encode(msg["text"])
                    actions.append(action)

            if len(states) < 2 or len(actions) < 1:
                continue

            for k in range(min(len(states) - 1, len(actions))):
                self.dynamics.update_from_observation(
                    states[k],
                    actions[k],
                    states[k + 1],
                    ConversationOutcome(
                        resolved=False,
                        escalated=False,
                        frustrated=False,
                        sentiment_improved=False,
                        turns_count=1,
                        success=False,
                    ),
                )

        self.trained = True
        confidence = self.dynamics.get_confidence()
        print(f"Training complete. Model confidence: {confidence:.3f}")

        return confidence

    def infer_state(self, text: str) -> np.ndarray:
        return self.state_encoder.encode(text)

    def select_optimal_action(self, state: np.ndarray) -> Tuple[str, float, float]:
        result = self.optimizer.evaluate_action(state, "resolve_issue")

        all_results = self.optimizer.evaluate_all_actions(state, top_k=5)

        return (
            all_results[0].action,
            all_results[0].expected_cost,
            all_results[0].confidence,
        )

    def get_alternatives(self, state: np.ndarray, top_k: int = 5) -> List[Dict]:
        results = self.optimizer.evaluate_all_actions(state, top_k=top_k)

        return [
            {
                "action": r.action,
                "expected_cost": float(r.expected_cost),
                "confidence": float(r.confidence),
                "reasoning": r.reasoning,
            }
            for r in results
        ]

    def simulate_trajectory(
        self, state: np.ndarray, action: str, steps: int = 3
    ) -> List[Dict]:
        action_vec = np.zeros(self.action_space.get_action_count())
        action_idx = self.action_space.action_to_idx.get(action, 0)
        action_vec[action_idx] = 1.0

        states = [state.copy()]
        current_state = state.copy()

        for _ in range(steps):
            next_state = self.dynamics.predict_next_state(current_state, action_vec)
            states.append(next_state.copy())
            current_state = next_state

        return [
            {
                "sentiment": self.state_encoder.get_state_components(s)["sentiment"],
                "urgency": self.state_encoder.get_state_components(s)["urgency"],
                "frustrated": self.state_encoder.get_state_components(s)["frustrated"],
                "resolved": self.state_encoder.get_state_components(s)["resolved"],
            }
            for s in states
        ]


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 train_system.py <csv_path>")
        sys.exit(1)

    csv_path = sys.argv[1]

    system = DecisionIntelligenceSystem()
    confidence = system.train(csv_path, max_convs=3000)

    print("\n" + "=" * 60)
    print("Training Complete!")
    print(f"Final model confidence: {confidence:.3f}")
    print("=" * 60)

    test_state = system.infer_state("@sprintcare is the worst customer service")
    print(f"\nTest state from frustration text:")
    components = system.state_encoder.get_state_components(test_state)
    print(f"  Sentiment: {components['sentiment']:.2f}")
    print(f"  Frustrated: {components['frustrated']:.2f}")
    print(f"  Urgency: {components['urgency']:.2f}")

    action, cost, conf = system.select_optimal_action(test_state)
    print(f"\nOptimal action: {action}")
    print(f"Expected cost: {cost:.3f}")
    print(f"Confidence: {conf:.3f}")

    alternatives = system.get_alternatives(test_state, top_k=5)
    print("\nRanked alternatives:")
    for i, alt in enumerate(alternatives):
        print(
            f"  {i + 1}. {alt['action']}: cost={alt['expected_cost']:.3f}, reasoning={alt['reasoning']}"
        )


if __name__ == "__main__":
    main()

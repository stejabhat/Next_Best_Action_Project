import numpy as np
import sys


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 demo_system.py <csv_path>")
        print("\nThis demo shows the continuous-time decision intelligence system")
        print("in action on customer support conversations.\n")
        sys.exit(1)

    csv_path = sys.argv[1]

    from train_system import DecisionIntelligenceSystem

    print("=" * 70)
    print("Continuous-Time Decision Intelligence System")
    print("Optimizing Customer Support Conversations")
    print("=" * 70)

    system = DecisionIntelligenceSystem()
    print("\nTraining on historical conversation data...")
    confidence = system.train(csv_path, max_convs=2000)
    print(f"Model trained with confidence: {confidence:.3f}")

    print("\n" + "=" * 70)
    print("DEMO: Analyzing Customer Support Conversation")
    print("=" * 70)

    conversation = [
        "@sprintcare I have sent several private messages and no one is responding as usual",
        "@sprintcare I did.",
        "@sprintcare and how do you propose we do that",
    ]

    current_state = None
    for i, text in enumerate(conversation):
        current_state = system.infer_state(text)
        components = system.state_encoder.get_state_components(current_state)

        print(f"\n[Customer Turn {i + 1}]")
        print(f"  Text: {text}")
        print(f"  State:")
        print(f"    - Sentiment: {components['sentiment']:.2f}")
        print(f"    - Urgency: {components['urgency']:.2f}")
        print(f"    - Frustrated: {components['frustrated']:.2f}")
        print(
            f"    - Intent: {max(components.items(), key=lambda x: x[1] if 'intent_' in x[0] else 0)}"
        )

    print("\n" + "-" * 70)
    print("OPTIMAL ACTION SELECTION")
    print("-" * 70)

    action, cost, conf = system.select_optimal_action(current_state)
    print(f"\nOptimal Next Action: {action}")
    print(f"  Expected Cost: {cost:.3f}")
    print(f"  Model Confidence: {conf:.3f}")

    print("\nAlternative Actions Ranked by Outcome:")
    alternatives = system.get_alternatives(current_state, top_k=5)
    for i, alt in enumerate(alternatives):
        print(f"  {i + 1}. {alt['action']}")
        print(f"     Expected Cost: {alt['expected_cost']:.3f}")
        print(f"     Reasoning: {alt['reasoning']}")

    print("\n" + "-" * 70)
    print("TRAJECTORY SIMULATION")
    print("-" * 70)

    test_actions = ["empathize", "apologize", "provide_info", "request_dm"]

    for action in test_actions:
        trajectory = system.simulate_trajectory(current_state, action, steps=3)
        print(f"\nAction: {action}")
        for j, state in enumerate(trajectory):
            print(
                f"  Step {j}: sentiment={state['sentiment']:.2f}, resolved={state['resolved']:.2f}, frustrated={state['frustrated']:.2f}"
            )

    print("\n" + "=" * 70)
    print("Expected Future Trajectory Summary:")
    print("=" * 70)
    print(f"""
Given the current state:
  - Sentiment: {components["sentiment"]:.2f} (negative)
  - Urgency: {components["urgency"]:.2f} (high)
  - Frustrated: {components["frustrated"]:.2f} (yes)

The optimal action '{action}' is selected by:
  1. Simulating future trajectories x(t+Δt) using learned dynamics
  2. Estimating cost: penalizing frustration, escalation, unresolved states
  3. Rewarding: resolution, sentiment improvement, shorter interactions
  4. Selecting action that minimizes total expected cost: argmin_a ∫ cost(x(t),a(t)) dt
    """)

    print("\n" + "=" * 70)
    print("OUTPUT: Decision Intelligence Recommendation")
    print("=" * 70)

    return {
        "optimal_action": action,
        "expected_cost": cost,
        "confidence": conf,
        "alternatives": [
            {"action": alt["action"], "expected_cost": alt["expected_cost"]}
            for alt in alternatives
        ],
        "current_state": {
            "sentiment": components["sentiment"],
            "urgency": components["urgency"],
            "frustrated": components["frustrated"],
            "resolved": components["resolved"],
        },
    }


if __name__ == "__main__":
    result = main()
    print(f"\nFinal Result: {result}")

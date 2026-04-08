# Continuous-Time Decision Intelligence System for Customer Support

A dynamic system that models customer support conversations as continuous-time processes and optimizes agent responses using learned trajectory dynamics.

## Overview

This system treats customer support conversations as a dynamic system evolving over time. Given the current conversation state x(t), the system models how the conversation will evolve under different agent actions and selects the optimal next action to minimize long-term cost.

## Problem Formulation

**State Vector x(t)** includes:
- Semantic intent (8 dimensions): billing_issue, service_issue, account_issue, technical_support, order_issue, general_inquiry, complaint, praise
- Sentiment score (1 dimension): -1 to +1
- Urgency score (1 dimension): 0 to 1
- Frustration indicator (1 dimension): 0 or 1
- Resolution indicator (1 dimension): 0 or 1

**Dynamics Function**: dx/dt = f(x(t), a(t))
- Models how the conversation state evolves given current state and agent action
- Learned from historical TWCS conversation trajectories

**Cost Function** penalizes:
- Increased frustration (+2.0)
- Escalation likelihood (+1.5)
- Unresolved states (+1.0)

**Rewards** for:
- Resolution (+3.0)
- Sentiment improvement (+1.5)
- Shorter interaction length (-0.1 per turn)

## Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  State Encoder  │────▶│ Dynamics Model   │────▶│ Action Optimizer│
│                 │     │  dx/dt = f(x,a)  │     │                 │
│ text → x(t)     │     │                  │     │ argmin_a ∫cost  │
└─────────────────┘     └──────────────────┘     └─────────────────┘
        │                       │                        │
        ▼                       ▼                        ▼
   - Intent           - Trajectory              - Optimal Action
   - Sentiment        - State Transitions       - Alternatives
   - Urgency         - Cost Estimation         - Confidence
```

## Files

| File | Description |
|------|-------------|
| `data_loader.py` | Loads TWCS CSV, builds conversation trajectories |
| `state_encoder.py` | Encodes text → state vector (intent, sentiment, urgency) |
| `action_space.py` | Defines 12 customer support action types |
| `dynamics.py` | Learns continuous-time dynamics from data |
| `optimizer.py` | Optimizes action selection via trajectory simulation |
| `inference_engine.py` | Main inference engine |
| `train_system.py` | Training script |
| `demo_system.py` | Full demonstration |

## Usage

### 1. Download the Dataset

Download the Twitter Customer Support Dataset from Kaggle:
https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter

Unzip `twcs.csv.zip` and place `twcs.csv` in the project directory.

### 2. Train the System

```bash
python3 train_system.py twcs.csv
```

### Run Demo

```bash
python3 demo_system.py twcs.csv
```

### Sample Output

A sample output file is provided for reference:

```bash
cat sample_output.txt
```

This shows:
- Training progress (loading data, building conversations, model confidence)
- Conversation state inference for 3 customer turns
- Optimal action selection with expected cost and confidence
- Alternative actions ranked by outcome
- Trajectory simulation for different actions
- Final recommendation output

### Use in Code

```python
from train_system import DecisionIntelligenceSystem

system = DecisionIntelligenceSystem()
system.train('twcs.csv', max_convs=2000)

# Infer state from customer message
state = system.infer_state("@sprintcare is the worst customer service")

# Get optimal action
action, cost, confidence = system.select_optimal_action(state)

# Get ranked alternatives
alternatives = system.get_alternatives(state, top_k=5)
```

## Dataset

The Twitter Customer Support Dataset (TWCS) is available on Kaggle:

**Download**: https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter

This dataset contains ~3M tweets from multiple companies (Sprint, Verizon, Spectrum, Chipotle, etc.) representing customer support conversations.

After downloading, unzip `twcs.csv.zip` and place `twcs.csv` in the project directory.

## Training Data

The system is trained on the Twitter Customer Support Dataset (TWCS):
- ~3M tweets
- ~300K conversations
- Multiple companies: Sprint, Verizon, Spectrum, Chipotle, etc.

## Action Types

The system considers 12 action types:
1. `request_dm` - Request direct message for private assistance
2. `apologize` - Apologize for inconvenience
3. `empathize` - Show understanding of customer frustration
4. `provide_info` - Provide information/solution
5. `ask_question` - Ask clarifying questions
6. `resolve_issue` - Directly resolve the issue
7. `escalate` - Escalate to supervisor/specialist
8. `redirect` - Redirect to website/app/call
9. `request_info` - Request account information
10. `greeting` - Greeting message
11. `closing` - Closing message
12. `acknowledge` - Acknowledge customer concern

## Output

The system returns:
- **Optimal Next Action**: The action that minimizes expected cost
- **Expected Future Trajectory**: Simulated state evolution
- **Confidence Score**: Model confidence based on training data
- **Alternative Actions**: Ranked list of alternative actions with reasoning

## Constraints

- No external APIs or pretrained LLMs
- Learned representations from TWCS data only
- Decisions are explainable via state transitions

## References

This system implements the continuous-time decision intelligence framework described in the task specification, modeling conversations as dynamical systems and optimizing actions using trajectory simulation.

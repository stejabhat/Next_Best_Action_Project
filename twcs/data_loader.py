import csv
import json
from collections import defaultdict
from datetime import datetime
from typing import List, Dict, Tuple, Optional


class DataLoader:
    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self.tweets = {}
        self.conversations = []

    def load(self, max_tweets: Optional[int] = None):
        with open(self.csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            count = 0
            for row in reader:
                tweet_id = int(row["tweet_id"])
                self.tweets[tweet_id] = {
                    "id": tweet_id,
                    "author_id": row["author_id"],
                    "inbound": row["inbound"] == "True",
                    "created_at": row["created_at"],
                    "text": row["text"],
                    "response_tweet_id": self._parse_ids(
                        row.get("response_tweet_id", "")
                    ),
                    "in_response_to_tweet_id": self._parse_id(
                        row.get("in_response_to_tweet_id", "")
                    ),
                }
                count += 1
                if max_tweets and count >= max_tweets:
                    break

    def _parse_ids(self, s: str) -> List[int]:
        if not s or s.strip() == "":
            return []
        return [int(x.strip()) for x in s.split(",") if x.strip()]

    def _parse_id(self, s: str) -> Optional[int]:
        if not s or s.strip() == "":
            return None
        try:
            return int(s.strip())
        except ValueError:
            return None

    def build_conversations(self, min_turns: int = 2):
        parent_to_children = defaultdict(list)
        root_tweets = []

        for tweet_id, tweet in self.tweets.items():
            parent_id = tweet["in_response_to_tweet_id"]
            if parent_id:
                parent_to_children[parent_id].append(tweet_id)
            else:
                root_tweets.append(tweet_id)

        def get_conversation(root_id: int) -> List[int]:
            path = [root_id]
            children = parent_to_children.get(root_id, [])
            for child_id in sorted(children):
                path.extend(get_conversation(child_id))
            return path

        seen_texts = set()
        for root in sorted(root_tweets):
            conv = get_conversation(root)
            if len(conv) >= min_turns:
                conv_tweets = [self.tweets[tid] for tid in conv if tid in self.tweets]
                conv_text = tuple(t["text"][:50] for t in conv_tweets[:3])
                if conv_text not in seen_texts:
                    seen_texts.add(conv_text)
                    self.conversations.append(conv_tweets)

    def get_trajectories(self, max_convs: Optional[int] = None) -> List[Dict]:
        trajectories = []

        for conv in self.conversations:
            if max_convs and len(trajectories) >= max_convs:
                break

            customer_turns = []
            agent_turns = []
            prev_speaker = None

            for i, tweet in enumerate(conv):
                if tweet["inbound"]:
                    if prev_speaker == "agent":
                        customer_turns.append([])
                    if customer_turns:
                        customer_turns[-1].append(tweet)
                    prev_speaker = "customer"
                else:
                    if prev_speaker == "customer":
                        agent_turns.append([])
                    if agent_turns:
                        agent_turns[-1].append(tweet)
                    prev_speaker = "agent"

            if customer_turns and agent_turns:
                trajectories.append(
                    {
                        "customer_turns": customer_turns,
                        "agent_turns": agent_turns,
                        "full_conversation": conv,
                    }
                )

        return trajectories

    def get_action_texts(self) -> List[str]:
        action_texts = []
        for conv in self.conversations:
            for tweet in conv:
                if not tweet["inbound"]:
                    action_texts.append(tweet["text"])
        return action_texts

    def get_customer_initial_texts(self) -> List[str]:
        initial_texts = []
        for conv in self.conversations:
            for tweet in conv:
                if tweet["inbound"] and tweet["in_response_to_tweet_id"] is None:
                    initial_texts.append(tweet["text"])
        return initial_texts

    def get_customer_texts(self) -> List[str]:
        texts = []
        for conv in self.conversations:
            for tweet in conv:
                if tweet["inbound"]:
                    texts.append(tweet["text"])
        return texts


if __name__ == "__main__":
    loader = DataLoader("twcs.csv")
    print("Loading tweets...")
    loader.load(max_tweets=50000)
    print(f"Loaded {len(loader.tweets)} tweets")

    print("Building conversations...")
    loader.build_conversations(min_turns=3)
    print(f"Found {len(loader.conversations)} conversations")

    trajectories = loader.get_trajectories(max_convs=10)
    print(f"Extracted {len(trajectories)} trajectories")

    if trajectories:
        print("\nSample trajectory:")
        t = trajectories[0]
        print(f"  Customer turns: {len(t['customer_turns'])}")
        print(f"  Agent turns: {len(t['agent_turns'])}")
        if t["customer_turns"]:
            print(
                f"  First customer message: {t['customer_turns'][0][0]['text'][:100]}..."
            )

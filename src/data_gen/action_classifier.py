"""Embedding-based action classification using BAAI/bge-large-en-v1.5.

This module provides semantic action classification by comparing text embeddings
against prototype embeddings for each ActionType category. This replaces the
regex-based keyword matching approach for more robust action inference.

Example:
    >>> classifier = EmbeddingActionClassifier()
    >>> action = classifier.classify("A person walking briskly down the street")
    >>> print(action)  # ActionType.WALK
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Optional

import torch
import torch.nn.functional as F

from src.data_gen.schema import ActionType

logger = logging.getLogger(__name__)

# Default embedding model - same as used for text conditioning
DEFAULT_MODEL_NAME = "BAAI/bge-large-en-v1.5"

# Representative text descriptions for each ActionType.
# Multiple descriptions per action improve matching robustness.
ACTION_PROTOTYPES: dict[ActionType, list[str]] = {
    # Basic actions
    ActionType.IDLE: [
        "A person standing still",
        "Someone remaining stationary",
        "A figure not moving, staying in place",
        "Person idle, waiting",
    ],
    ActionType.WALK: [
        "A person walking",
        "Someone strolling forward",
        "Walking at a casual pace",
        "Moving forward with steady steps",
        "Striding ahead",
    ],
    ActionType.RUN: [
        "A person running",
        "Someone jogging quickly",
        "Running fast",
        "Sprinting forward",
        "Moving at a quick pace",
    ],
    ActionType.SPRINT: [
        "A person sprinting at full speed",
        "Running as fast as possible",
        "Dashing forward rapidly",
    ],
    ActionType.JUMP: [
        "A person jumping",
        "Someone leaping into the air",
        "Hopping up",
        "Jumping high",
        "Bouncing upward",
    ],
    # Social actions
    ActionType.WAVE: [
        "A person waving their hand",
        "Someone waving hello or goodbye",
        "Waving arm in greeting",
    ],
    ActionType.TALK: [
        "A person talking",
        "Someone speaking",
        "Having a conversation",
        "Discussing with gestures",
    ],
    ActionType.SHOUT: [
        "A person shouting",
        "Someone yelling loudly",
        "Calling out with force",
    ],
    ActionType.WHISPER: [
        "A person whispering",
        "Someone speaking quietly",
        "Talking in a hushed voice",
    ],
    ActionType.SING: [
        "A person singing",
        "Someone performing a song",
        "Vocalizing melodically",
    ],
    ActionType.POINT: [
        "A person pointing",
        "Someone indicating a direction",
        "Pointing at something",
    ],
    ActionType.CLAP: [
        "A person clapping",
        "Someone applauding",
        "Clapping hands together",
    ],
    # Sports actions - Baseball
    ActionType.BATTING: [
        "A person batting",
        "Swinging a baseball bat",
        "Hitting a ball with a bat",
    ],
    ActionType.PITCHING: [
        "A person pitching",
        "Throwing a baseball pitch",
        "Winding up to throw",
    ],
    ActionType.CATCHING: [
        "A person catching a ball",
        "Someone receiving a throw",
        "Catching with hands or glove",
    ],
    ActionType.RUNNING_BASES: [
        "A person running bases",
        "Running around the baseball diamond",
        "Sprinting between bases",
    ],
    ActionType.FIELDING: [
        "A person fielding",
        "Playing defense in baseball",
        "Catching and throwing in the field",
    ],
    ActionType.THROWING: [
        "A person throwing",
        "Someone tossing an object",
        "Throwing a ball",
    ],
    # Sports actions - General
    ActionType.KICKING: [
        "A person kicking",
        "Kicking a ball",
        "Striking with the foot",
    ],
    ActionType.DRIBBLING: [
        "A person dribbling",
        "Bouncing a basketball",
        "Dribbling a soccer ball",
    ],
    ActionType.SHOOTING: [
        "A person shooting a ball",
        "Taking a shot at a goal",
        "Shooting a basketball",
    ],
    # Combat actions
    ActionType.FIGHT: [
        "A person fighting",
        "Engaging in combat",
        "Two people fighting each other",
    ],
    ActionType.PUNCH: [
        "A person punching",
        "Throwing a punch",
        "Striking with a fist",
    ],
    ActionType.KICK: [
        "A person performing a kick",
        "Martial arts kick",
        "Kicking in a fight",
    ],
    ActionType.DODGE: [
        "A person dodging",
        "Evading an attack",
        "Moving out of the way quickly",
    ],
    # Narrative actions
    ActionType.SIT: [
        "A person sitting down",
        "Someone seated",
        "Sitting on a chair",
    ],
    ActionType.STAND: [
        "A person standing up",
        "Rising to stand",
        "Getting up from seated position",
    ],
    ActionType.KNEEL: [
        "A person kneeling",
        "Getting down on one knee",
        "Kneeling on the ground",
    ],
    ActionType.LIE_DOWN: [
        "A person lying down",
        "Someone reclining",
        "Lying flat on the ground",
    ],
    ActionType.EATING: [
        "A person eating",
        "Someone consuming food",
        "Eating a meal",
    ],
    ActionType.DRINKING: [
        "A person drinking",
        "Someone consuming a beverage",
        "Drinking from a cup",
    ],
    ActionType.READING: [
        "A person reading",
        "Someone looking at a book",
        "Reading a document",
    ],
    ActionType.TYPING: [
        "A person typing",
        "Someone using a keyboard",
        "Typing on a computer",
    ],
    # Exploration actions
    ActionType.LOOKING_AROUND: [
        "A person looking around",
        "Someone surveying the area",
        "Turning head to observe surroundings",
    ],
    ActionType.CLIMBING: [
        "A person climbing",
        "Someone ascending",
        "Climbing up a ladder or wall",
    ],
    ActionType.CRAWLING: [
        "A person crawling",
        "Someone moving on hands and knees",
        "Crawling on the ground",
    ],
    ActionType.SWIMMING: [
        "A person swimming",
        "Someone moving through water",
        "Swimming strokes",
    ],
    ActionType.FLYING: [
        "A person flying",
        "Someone soaring through the air",
        "Flying like a superhero",
    ],
    # Emotional actions
    ActionType.CELEBRATE: [
        "A person celebrating",
        "Someone expressing joy",
        "Celebrating a victory",
        "Cheering with excitement",
    ],
    ActionType.DANCE: [
        "A person dancing",
        "Someone moving rhythmically",
        "Dancing to music",
        "Performing dance moves",
    ],
    ActionType.CRY: [
        "A person crying",
        "Someone weeping",
        "Expressing sadness with tears",
    ],
    ActionType.LAUGH: [
        "A person laughing",
        "Someone expressing amusement",
        "Laughing happily",
    ],
    # Interactive Actions (Multi-Actor)
    ActionType.HANDSHAKE: [
        "Two people shaking hands",
        "A handshake greeting",
        "Shaking hands with someone",
    ],
    ActionType.HUG: [
        "Two people hugging",
        "Someone giving a hug",
        "Embracing another person",
    ],
    ActionType.HIGH_FIVE: [
        "Two people high-fiving",
        "Giving a high five",
        "Slapping hands in celebration",
    ],
    ActionType.FIGHT_STANCE: [
        "A person in fighting stance",
        "Taking a combat ready position",
        "Adopting a defensive posture",
    ],
}


class EmbeddingActionClassifier:
    """Classify actions from text using embedding similarity.

    This classifier uses the BGE text encoder to embed input descriptions
    and compares them against pre-computed prototype embeddings for each
    ActionType using cosine similarity.

    Attributes:
        model_name: Name of the sentence transformer model.
        device: Device to run embeddings on (cpu/cuda).
        action_embeddings: Pre-computed mean embeddings per ActionType.
    """

    _instance: Optional["EmbeddingActionClassifier"] = None
    _initialized: bool = False

    def __new__(cls, *args, **kwargs) -> "EmbeddingActionClassifier":
        """Singleton pattern to avoid reloading the model multiple times."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        device: Optional[str] = None,
    ) -> None:
        """Initialize the classifier.

        Args:
            model_name: Hugging Face model name for embeddings.
            device: Device to use (auto-detected if None).
        """
        if self._initialized:
            return

        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model = None
        self._action_embeddings: Optional[dict[ActionType, torch.Tensor]] = None

        EmbeddingActionClassifier._initialized = True

    @property
    def model(self):
        """Lazy-load the embedding model."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer

                logger.info(f"Loading embedding model: {self.model_name}...")
                self._model = SentenceTransformer(self.model_name)
                self._model.to(self.device)
                logger.info(f"Embedding model loaded on {self.device}")
            except ImportError:
                logger.error(
                    "sentence-transformers not installed. "
                    "Run: pip install sentence-transformers"
                )
                raise
        return self._model

    @property
    def action_embeddings(self) -> dict[ActionType, torch.Tensor]:
        """Lazy-compute action prototype embeddings."""
        if self._action_embeddings is None:
            self._action_embeddings = self._compute_prototype_embeddings()
        return self._action_embeddings

    def _compute_prototype_embeddings(self) -> dict[ActionType, torch.Tensor]:
        """Compute mean embedding for each action type from prototypes."""
        logger.info("Computing action prototype embeddings...")
        embeddings: dict[ActionType, torch.Tensor] = {}

        for action, prototypes in ACTION_PROTOTYPES.items():
            # Encode all prototypes for this action
            proto_embeddings = self.model.encode(
                prototypes,
                convert_to_tensor=True,
                device=self.device,
                show_progress_bar=False,
            )
            # Mean pooling across all prototypes
            mean_embedding = proto_embeddings.mean(dim=0)
            # Normalize for cosine similarity
            embeddings[action] = F.normalize(mean_embedding, dim=0)

        logger.info(f"Computed embeddings for {len(embeddings)} action types")
        return embeddings

    def classify(
        self,
        text: str,
        return_scores: bool = False,
        top_k: int = 1,
    ) -> ActionType | tuple[ActionType, dict[ActionType, float]]:
        """Classify text into an ActionType.

        Args:
            text: Input text description to classify.
            return_scores: If True, also return similarity scores.
            top_k: Return top-k most similar actions (only if return_scores=True).

        Returns:
            Best matching ActionType, or tuple of (ActionType, scores) if
            return_scores is True.
        """
        if not text or not text.strip():
            return (ActionType.IDLE, {ActionType.IDLE: 1.0}) if return_scores else ActionType.IDLE

        # Encode input text
        text_embedding = self.model.encode(
            text,
            convert_to_tensor=True,
            device=self.device,
            show_progress_bar=False,
        )
        text_embedding = F.normalize(text_embedding, dim=0)

        # Compute cosine similarity with all action embeddings
        scores: dict[ActionType, float] = {}
        for action, action_emb in self.action_embeddings.items():
            similarity = torch.dot(text_embedding, action_emb).item()
            scores[action] = similarity

        # Find best match
        best_action = max(scores, key=scores.get)

        if return_scores:
            # Sort and return top-k
            sorted_scores = dict(
                sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
            )
            return best_action, sorted_scores

        return best_action

    def classify_batch(
        self,
        texts: list[str],
        return_scores: bool = False,
    ) -> list[ActionType] | list[tuple[ActionType, dict[ActionType, float]]]:
        """Classify multiple texts efficiently.

        Args:
            texts: List of text descriptions to classify.
            return_scores: If True, also return similarity scores.

        Returns:
            List of ActionTypes (or tuples with scores if return_scores=True).
        """
        if not texts:
            return []

        # Handle empty strings
        valid_indices = [i for i, t in enumerate(texts) if t and t.strip()]
        valid_texts = [texts[i] for i in valid_indices]

        if not valid_texts:
            if return_scores:
                return [(ActionType.IDLE, {ActionType.IDLE: 1.0})] * len(texts)
            return [ActionType.IDLE] * len(texts)

        # Batch encode
        text_embeddings = self.model.encode(
            valid_texts,
            convert_to_tensor=True,
            device=self.device,
            show_progress_bar=False,
        )
        text_embeddings = F.normalize(text_embeddings, dim=1)

        # Stack action embeddings into a matrix
        action_list = list(self.action_embeddings.keys())
        action_matrix = torch.stack(
            [self.action_embeddings[a] for a in action_list]
        )  # [num_actions, embed_dim]

        # Compute all similarities at once: [batch, num_actions]
        similarities = torch.matmul(text_embeddings, action_matrix.T)

        # Get best matches
        best_indices = similarities.argmax(dim=1)

        # Build results
        results: list = [None] * len(texts)
        for i, valid_idx in enumerate(valid_indices):
            best_action = action_list[best_indices[i].item()]
            if return_scores:
                scores = {
                    action_list[j]: similarities[i, j].item()
                    for j in range(len(action_list))
                }
                results[valid_idx] = (best_action, scores)
            else:
                results[valid_idx] = best_action

        # Fill in empty strings with IDLE
        for i in range(len(texts)):
            if results[i] is None:
                if return_scores:
                    results[i] = (ActionType.IDLE, {ActionType.IDLE: 1.0})
                else:
                    results[i] = ActionType.IDLE

        return results


# Global singleton accessor
@lru_cache(maxsize=1)
def get_action_classifier(
    model_name: str = DEFAULT_MODEL_NAME,
) -> EmbeddingActionClassifier:
    """Get or create the singleton action classifier.

    This function provides a convenient way to access the classifier
    without worrying about initialization.

    Args:
        model_name: Embedding model name.

    Returns:
        Initialized EmbeddingActionClassifier instance.
    """
    return EmbeddingActionClassifier(model_name=model_name)


def classify_action(text: str) -> ActionType:
    """Convenience function to classify a single text.

    This is a drop-in replacement for _infer_action_from_text().

    Args:
        text: Text description to classify.

    Returns:
        Best matching ActionType.
    """
    classifier = get_action_classifier()
    return classifier.classify(text)


def classify_action_from_texts(texts: list[str]) -> ActionType:
    """Classify action from multiple text descriptions.

    Combines all texts and classifies the combined description.
    This matches the signature of the original _infer_action_from_text().

    Args:
        texts: List of text descriptions.

    Returns:
        Best matching ActionType.
    """
    if not texts:
        return ActionType.IDLE

    combined = " ".join(texts)
    return classify_action(combined)


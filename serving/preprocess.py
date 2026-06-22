"""
ClearML Serving — pre/post-processing for sklearn spam classifier.

Expected model: sklearn Pipeline(TfidfVectorizer -> LogisticRegression/MultinomialNB)
  - Input:  list of strings
  - Output: list of labels ("ham" or "spam")

Request body:   {"text": "free entry in a weekly draw"}
Response body:  {"label": "spam"}
"""


class Preprocess:
    """ClearML Serving custom pre/post-processing class."""

    def __init__(self):
        pass

    def preprocess(self, body: dict, state: dict, collect_custom_statistics_fn=None):
        """Validate input and return a list of texts for model.predict()."""

        # body arrives as {"text": "..."} from the HTTP request
        if not isinstance(body, dict):
            raise ValueError(
                f"Request body must be a JSON object, got {type(body).__name__}"
            )

        text = body.get("text")
        if not text or not isinstance(text, str) or not text.strip():
            raise ValueError(
                'Missing or empty "text" field in request body. '
                "Send JSON like: {\"text\": \"your message here\"}"
            )

        # sklearn Pipeline.predict() expects an iterable of strings
        return [text.strip()]

    def postprocess(self, data, state: dict, collect_custom_statistics_fn=None):
        """Convert model prediction to JSON-friendly response."""

        # data is the raw output of model.predict([...])
        # sklearn returns a numpy array with one element per input
        if hasattr(data, "__len__") and len(data) > 0:
            label = str(data[0])
        else:
            label = str(data)

        return {"label": label}

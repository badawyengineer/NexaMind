class GenAIError(Exception):
    """Base application error."""


class ConfigurationError(GenAIError):
    pass


class ProviderError(GenAIError):
    pass


class ValidationFailure(GenAIError):
    pass


class RetrievalError(GenAIError):
    pass


class EmbeddingCompatibilityError(RetrievalError):
    """Raised when collection vectors do not match the configured embedder."""


class ToolError(GenAIError):
    pass

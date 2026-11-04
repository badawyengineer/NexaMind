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


class ToolError(GenAIError):
    pass

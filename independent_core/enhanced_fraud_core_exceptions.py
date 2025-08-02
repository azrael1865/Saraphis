class EnhancedFraudCoreException(Exception): pass
class FraudDataError(Exception):
    def __init__(self, message='', **kwargs):
        super().__init__(message)
        self.kwargs = kwargs
class FraudValidationError(Exception): pass
class FraudProcessingError(Exception): pass

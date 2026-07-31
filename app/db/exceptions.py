"""
Domain exceptions for database operations.
"""

class DatabaseError(Exception):
    """Base exception for database errors."""
    pass


class CaseNotFoundError(DatabaseError):
    """Exception raised when a case is not found."""
    def __init__(self, case_id: str):
        self.case_id = case_id
        super().__init__(f"Case with ID '{case_id}' was not found in the database.")


class CaseSaveError(DatabaseError):
    """Exception raised when saving a case fails."""
    def __init__(self, case_id: str, reason: str):
        self.case_id = case_id
        self.reason = reason
        super().__init__(f"Failed to save case '{case_id}': {reason}")

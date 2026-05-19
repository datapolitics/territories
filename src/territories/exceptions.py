class MissingTreeException(Exception):
    pass


class MissingTreeCache(MissingTreeException):
    pass


class InvalidTreeCache(MissingTreeException):
    pass


class NotOnTreeError(Exception):
    pass


class EmptyTerritoryError(Exception):
    pass

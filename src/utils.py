from uuid import UUID


def is_uuid(string: str):
    try:
        uuid_obj = UUID(string, version=4)
    except ValueError:
        return False
    return str(uuid_obj) == string

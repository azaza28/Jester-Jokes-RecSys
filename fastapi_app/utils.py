import hashlib

def assign_user_to_group(user_id: int) -> str:
    """Hash the user ID and assign to one of the 6 groups."""
    groups = ['A', 'B', 'C', 'D', 'E', 'F']
    hash_value = int(hashlib.md5(str(user_id).encode()).hexdigest(), 16)
    assigned_group = groups[hash_value % len(groups)]
    return assigned_group
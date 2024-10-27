def calculate_num_warps(n):
    """Calculate optimal number of warps based on block size"""
    if n <= 512:
        return 4
    elif n <= 1024:
        return 8
    else:
        return 16
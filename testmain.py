def fibbonacci_index(n):
    """
    Returns the n-th Fibonacci index, where 0 is the 0-th index and 1 is the 1st.

    Args:
        n (int): The index of the Fibonacci number to return.

    Returns:
        int: The value at the n-th index.
    """
    if n == 0:
        return 0
    elif n == 1:
        return 1
    else:
        a, b = 0, 1
        for _ in range(2, n + 1):
            a, b = b, a + b
        return b


print(fibbonacci_index(4))

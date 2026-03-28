```python
def fibbonacci_index(n):
    """
    Returns the n-th Fibonacci index, where 0 is the 0-th index and 1 is the 1st index.
    
    Args:
        n (int): The index of the Fibonacci number to return.
    
    Returns:
        int: The value at the n-th index of the Fibonacci sequence.
    """
    if n == 0:
        return 0
    elif n == 1:
        return 1
    else:
        return fibbonacci_index(n - 1) + fibbonacci_index(n - 2)
```
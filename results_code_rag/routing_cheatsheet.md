# When to choose which RAG (from *your* Hotpot data)

Cutting-edge production pattern (Adaptive-RAG): **route by query type**, don't crown a single stack.

- **comparison / local factoid** (`local`) → **Vector + rerank** (mean 0.30)

## Pairwise takeaways

## Research honesty: is this cutting edge?
- **Yes as an engineering bake-off**: FrontierRAG (Adaptive+CRAG escalate) + BM25/dense RRF + cross-encoder rerank + GraphRAG modes + HippoRAG 2/LightRAG is a 2025–2026-relevant *system* stack.
- **No as a SOTA paper claim**: n=12 Hotpot subset, local 3B judge=generator, no full BenchmarkQED AutoQ/AutoE LLM pairwise, LazyGraphRAG itself still not OSS.
- **Cutting-edge move**: ship the *router* that grades retrieval and escalates compute; keep the cost/latency scorecard; use stronger models for OpenIE and judging.

## Concrete choose-A-over-B examples

### Choose **Vector + rerank** over Semantic (vector)
- Q: Complete the following Python function. Return only the full function implementation.

from typing import List


def has_close_elements(numbers: List[float], threshold: float) -> bool:
    """ Check if in given list of numbers, are any two numbers closer to each other than
    given threshold.
    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)
    False
    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)
    True
    """
- Gold: `from typing import List


def has_close_elements(numbers: List[float], threshold: float) -> bool:
    """ Check if in given list of numbers, are any two numbers closer to each other than
    given threshold.
    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)
    False
    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)
    True
    """
    for idx, elem in enumerate(numbers):
        for idx2, elem2 in enumerate(numbers):
            if idx != idx2:
                distance = abs(elem - elem2)
                if distance < threshold:
                    return True

    return False` (humaneval)
- Scores: 0.56 vs 0.00 (margin 0.56)
- Why: On this question, Vector + rerank beat Semantic (vector).

### Choose **Semantic (vector)** over Vector + rerank
- Q: Complete the following Python function. Return only the full function implementation.

from typing import List


def filter_by_substring(strings: List[str], substring: str) -> List[str]:
    """ Filter an input list of strings only for ones that contain given substring
    >>> filter_by_substring([], 'a')
    []
    >>> filter_by_substring(['abc', 'bacd', 'cde', 'array'], 'a')
    ['abc', 'bacd', 'array']
    """
- Gold: `from typing import List


def filter_by_substring(strings: List[str], substring: str) -> List[str]:
    """ Filter an input list of strings only for ones that contain given substring
    >>> filter_by_substring([], 'a')
    []
    >>> filter_by_substring(['abc', 'bacd', 'cde', 'array'], 'a')
    ['abc', 'bacd', 'array']
    """
    return [x for x in strings if substring in x]` (humaneval)
- Scores: 0.49 vs 0.04 (margin 0.44)
- Why: On this question, Semantic (vector) beat Vector + rerank.

### Choose **Semantic (vector)** over Vector + rerank
- Q: Complete the following Python function. Return only the full function implementation.

from typing import List


def below_zero(operations: List[int]) -> bool:
    """ You're given a list of deposit and withdrawal operations on a bank account that starts with
    zero balance. Your task is to detect if at any point the balance of account fallls below zero, and
    at that point function should return True. Otherwise it should return False.
    >>> below_zero([1, 2, 3])
    False
    >>> below_zero([1, 2, -4, 5])
    True
    """
- Gold: `from typing import List


def below_zero(operations: List[int]) -> bool:
    """ You're given a list of deposit and withdrawal operations on a bank account that starts with
    zero balance. Your task is to detect if at any point the balance of account fallls below zero, and
    at that point function should return True. Otherwise it should return False.
    >>> below_zero([1, 2, 3])
    False
    >>> below_zero([1, 2, -4, 5])
    True
    """
    balance = 0

    for op in operations:
        balance += op
        if balance < 0:
            return True

    return False` (humaneval)
- Scores: 0.58 vs 0.26 (margin 0.32)
- Why: On this question, Semantic (vector) beat Vector + rerank.

### Choose **Vector + rerank** over Semantic (vector)
- Q: Write a Python function for the following problem. Return only code.

Write a function to count the most common words in a dictionary.
- Gold: `from collections import Counter
def count_common(words):
  word_counts = Counter(words)
  top_four = word_counts.most_common(4)
  return (top_four)` (mbpp)
- Scores: 0.51 vs 0.20 (margin 0.31)
- Why: On this question, Vector + rerank beat Semantic (vector).

### Choose **Vector + rerank** over Semantic (vector)
- Q: Write a Python function for the following problem. Return only code.

Write a function to split a string at lowercase letters.
- Gold: `import re
def split_lowerstring(text):
 return (re.findall('[a-z][^a-z]*', text))` (mbpp)
- Scores: 0.47 vs 0.28 (margin 0.20)
- Why: On this question, Vector + rerank beat Semantic (vector).

### Choose **Semantic (vector)** over Vector + rerank
- Q: Write a Python function for the following problem. Return only code.

Write a function to sort a given matrix in ascending order according to the sum of its rows.
- Gold: `def sort_matrix(M):
    result = sorted(M, key=sum)
    return result` (mbpp)
- Scores: 0.30 vs 0.12 (margin 0.17)
- Why: On this question, Semantic (vector) beat Vector + rerank.

### Choose **Vector + rerank** over Semantic (vector)
- Q: Complete the following Python function. Return only the full function implementation.

from typing import List


def mean_absolute_deviation(numbers: List[float]) -> float:
    """ For a given list of input numbers, calculate Mean Absolute Deviation
    around the mean of this dataset.
    Mean Absolute Deviation is the average absolute difference between each
    element and a centerpoint (mean in this case):
    MAD = average | x - x_mean |
    >>> mean_absolute_deviation([1.0, 2.0, 3.0, 4.0])
    1.0
    """
- Gold: `from typing import List


def mean_absolute_deviation(numbers: List[float]) -> float:
    """ For a given list of input numbers, calculate Mean Absolute Deviation
    around the mean of this dataset.
    Mean Absolute Deviation is the average absolute difference between each
    element and a centerpoint (mean in this case):
    MAD = average | x - x_mean |
    >>> mean_absolute_deviation([1.0, 2.0, 3.0, 4.0])
    1.0
    """
    mean = sum(numbers) / len(numbers)
    return sum(abs(x - mean) for x in numbers) / len(numbers)` (humaneval)
- Scores: 0.47 vs 0.31 (margin 0.16)
- Why: On this question, Vector + rerank beat Semantic (vector).


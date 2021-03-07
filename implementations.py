import numpy as np
from typing import List, Tuple


np.set_printoptions(precision=10, suppress=True)  # type: ignore

# The distinct features in our database which take the values from (0,1)
FEATURES = ("COVID", "Flu", "Cold")

# The mapping from the database index to the FEATURES space
MAPPING_DICT = {
    0: frozenset({}),
    1: frozenset({"Cold"}),
    2: frozenset({"Flu"}),
    3: frozenset({"Flu", "Cold"}),
    4: frozenset({"COVID"}),
    5: frozenset({"COVID", "Cold"}),
    6: frozenset({"COVID", "Flu"}),
    7: frozenset({"COVID", "Flu", "Cold"}),
}

# The above mapping reversed
MAPPING_DICT_REVERSED = {value: key for key, value in MAPPING_DICT.items()}

# Which indexes contain the diseases
MAPPING_DICT_INDEXES = {
    "COVID": (4, 5, 6, 7),
    "Flu": (2, 3, 6, 7),
    "Cold": (1, 3, 5, 7),
}


def generateDB(N: int, P: List[float]) -> List[int]:
    """Generates a database of size N.

    Args:
        N (int): Number of participants in the database
        P (List[float]): A 1-by-8 vector which is the probability distribution of each distinct condition. Index 0 corresponds to "000", index 1 corresponds to "001", index i corresponds to the binary representation of i

    Returns:
        List[int]: A 1-by-8 vector of database which has the counts of each distinct condition
    """
    assert round(sum(P), 10) == 1, \
        f"The probability distribution doesn't add up to 1, current value: {round(sum(P), 10)}"
    D = np.zeros(2**len(FEATURES), dtype=int)
    for _ in range(N):
        D[np.random.choice(8, p=P)] += 1
    return D


def LDP(D: List[int], R: List[float]) -> List[int]:
    """Local differential privacy mechanism. Flips the bit `i` with a probability of `R[i]`.

    Args:
        D (List[int]): Database to apply the local differential privacy mechanism on
        R (List[float]): A 1-by-3 vector which is the probability of flipping the `i`th bit (0th bit is for "COVID", 1st bit is for "Flu", and 2nd bit is for "Cold")

    Returns:
        List[int]: A new database with LDP applied
    """
    for prob in R:
        assert 0 <= prob <= 1, f"The probability of flipping must be between 0 and 1, current value: {prob}"
    D_LDP = np.zeros(len(D), dtype=int)
    for index, item in enumerate(D):
        # Decomposition of index into bits
        # bit0, bit1 and bit2 correspond to the presence of "COVID", "Flu" and "Cold", respectively
        bit0 = index//4
        bit1 = (index-bit0*4)//2
        bit2 = index-bit0*4-bit1*2
        for _ in range(item):
            # The outcome 1 means "flip", the outcome 0 means "don't flip"
            flip0 = np.random.binomial(1, R[0])
            flip1 = np.random.binomial(1, R[1])
            flip2 = np.random.binomial(1, R[2])
            # The new index calculation
            newIndex = 4*abs(flip0-bit0) + \
                2*abs(flip1-bit1) + abs(flip2-bit2)
            # Increase the count of the new entry
            D_LDP[newIndex] += 1

    return D_LDP


def LaplaceCount1(D: List[int], epsilon: float = 1) -> Tuple[float, float]:
    """Counting queries with epsilon differential privacy using the Laplace mechanism

    Args:
        D (List[int]): The database to run the counting query on
        epsilon (float): The privacy parameter in the Laplace mechanism. Defaults to 1.

    Returns:
        Tuple[float, float]: A tuple of two elements where the elements are the number of entries in D and the number of entries in column 1, i.e., people who have "COVID"
    """
    # The sensitivity in this algorithm is 2 since an additional person can change at most the first and second entry by 1, summing to 2 in total
    sensitivity = 2
    queryResult = np.zeros(2)
    queryResult[0] = sum(D)
    queryResult[1] = sum(D[i] for i in MAPPING_DICT_INDEXES["COVID"])
    laplaceNoise = np.random.laplace(0, sensitivity/epsilon, 2)
    queryResult += laplaceNoise

    return queryResult


def LaplaceCount2(D: List[int], epsilon: float = 1) -> Tuple[float, float, float, float]:
    """Counting query with epsilon differential privacy, returning each individual count

    Args:
        D (List[int]): The database to run the counting query on
        epsilon (float, optional): The privacy parameter in the Laplace mechanism. Defaults to 1.

    Returns:
        Tuple[float, float, float, float]: Number of entries in total, in column 1, in column2 and in column 3
    """
    # The sensitivity of the algorithm is 4 in that case
    sensitivity = 4
    queryResult = np.zeros(4)
    queryResult[0] = sum(D)
    queryResult[1] = sum(D[i] for i in MAPPING_DICT_INDEXES["COVID"])
    queryResult[2] = sum(D[i] for i in MAPPING_DICT_INDEXES["Flu"])
    queryResult[3] = sum(D[i] for i in MAPPING_DICT_INDEXES["Cold"])
    laplaceNoise = np.random.laplace(0, sensitivity/epsilon, 4)
    queryResult += laplaceNoise

    return queryResult


def LaplaceCountHistogram(D: List[int], epsilon: float = 1) -> Tuple[float, float, float, float, float, float, float, float, float]:
    """Counting query with epsilon differential privacy, returning each individual count

    Args:
        D (List[int]): The database to run the counting query on
        epsilon (float, optional): The privacy parameter in the Laplace mechanism. Defaults to 1.

    Returns:
        Tuple[float, float, float, float, float, float, float, float, float]: Number of entries in total and each possible combination (8)
    """
    # The sensitivity of the algorithm is 2 in that case: an addition of one person has their own bin and they affect the total count
    sensitivity = 2
    laplaceNoise = np.random.laplace(0, sensitivity/epsilon, 9)
    queryResult = [sum(D), *D] + laplaceNoise

    return queryResult


def ExponentialMechanism(D: List[int], epsilon: float = 1) -> int:
    """Choose one of the more common tuples based on the counts

    Args:
        D (List[int]): The database
        epsilon (float, optional): The privacy level in the exponential mechanism. Defaults to 1.

    Returns:
        int: The index of the common entry (0-7)
    """
    # The utility is computed based on the count in that index
    utilityScores = D
    # The sensitivity here is 1 since addition of 1 person can change the count at most 1
    sensitivity = 1
    probabilities = [np.exp(epsilon*score/(2*sensitivity))
                     for score in utilityScores]
    probabilities /= np.sum(probabilities)
    choice = np.random.choice(8, 1, p=probabilities)
    choice = choice.item()
    return choice


if __name__ == "__main__":
    D = generateDB(10, [.0, .0, .0, .0, 1.0, .0, .0, .0])
    print(D)
    D_new = LDP(D, [1.0, .5, .5])
    print(D_new)
    D_laplace1 = LaplaceCount1(D)
    print(D_laplace1)
    D_laplace2 = LaplaceCount2(D, epsilon=100)
    print(D_laplace2)
    D_laplaceHist = LaplaceCountHistogram(D, epsilon=50)
    print(D_laplaceHist)
    mostCommonIndex = ExponentialMechanism(D)
    print(
        f"The most common index is {mostCommonIndex} which denotes {MAPPING_DICT[mostCommonIndex]}.")

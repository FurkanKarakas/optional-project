import numpy as np
from typing import DefaultDict, Dict, List, Tuple, Union
from collections import defaultdict

np.set_printoptions(precision=10, suppress=True)

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


def generateDB(N: int, P: List[float]) -> np.ndarray:
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


def LDP(D: List[int], R: List[float]) -> np.ndarray:
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


def ExponentialMechanism(D: np.ndarray, epsilon: float = 1) -> int:
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
    probabilities = np.exp(epsilon*utilityScores/(2*sensitivity))
    probabilities /= np.sum(probabilities)
    choice = np.random.choice(8, 1, p=probabilities)
    choice = choice.item()
    return choice

# A Huffman Tree Node


class node:
    def __init__(self, freq, symbol, left=None, right=None):
        # frequency of symbol
        self.freq = freq

        # symbol name (character)
        self.symbol = symbol

        # node left of current node
        self.left = left

        # node right of current node
        self.right = right

        # tree direction (0/1)
        self.huff = ''

# Store the symbol-code pairs


def storeNodes(node, symbolCodeDict, val=''):
    # huffman code for current node
    newVal = val + str(node.huff)

    # if node is not an edge node
    # then traverse inside it
    if(node.left):
        storeNodes(node.left, symbolCodeDict, newVal)
    if(node.right):
        storeNodes(node.right, symbolCodeDict, newVal)

    # if node is edge node then display its huffman code
    if(not node.left and not node.right):
        symbolCodeDict[node.symbol] = newVal


def HuffmanCompression(D: Union[np.ndarray, Dict[str, int]]) -> Tuple[Dict[str, str], DefaultDict[str, int]]:
    """The Huffman compression algorithm which compresses the symbols "000" to "111" based on their frequencies.

    Args:
        D (Union[List[int], DefaultDict[str, int]]): Input database to be compressed. Can be either list or dictionary.

    Returns:
        Tuple(Dict[str, str], DefaultDict[str, int]: Huffman symbol-code mapping and compressed database in the output. It maps the Huffman code (str, e.g. "01") associated with a symbol to a number, which is the number of people in a particular condition.
    """
    compressedDB = defaultdict(int)
    chars = freq = None
    # If the database is a dictionary, process it appropriately
    if isinstance(D, dict):
        chars = list(D.keys())
        freq = list(D.values())
    # If the database is a numpy array, process it appropriately
    else:
        chars = [str(i) for i in np.arange(0, 8)]
        freq = D

    # list containing unused nodes
    nodes = list()

    # Converting characters and frequencies into huffman tree nodes
    for x in range(len(chars)):
        nodes.append(node(freq[x], chars[x]))

    # We continue processing the nodes as long as there are more than 1 node in the list.
    while len(nodes) > 1:
        # Sort all the nodes in ascending order based on their frequency
        nodes = sorted(nodes, key=lambda x: x.freq)

        # pick 2 smallest nodes
        left = nodes[0]
        right = nodes[1]

        # assign directional value to these nodes
        left.huff = 0
        right.huff = 1

        # Combine the 2 smallest nodes to create new node as their parent
        newNode = node(left.freq+right.freq, left.symbol +
                       right.symbol, left, right)

        # remove the 2 nodes and add their
        # parent as new node among others
        nodes.remove(left)
        nodes.remove(right)
        nodes.append(newNode)

    # Huffman Tree is ready!
    symbolCodeMapping = dict()
    # Note that storeNodes traverses the binary tree in DFS manner. We stop at the leaf nodes and record the values in the dictionary.
    storeNodes(nodes[0], symbolCodeDict=symbolCodeMapping)

    # Return the new database
    if isinstance(D, dict):
        for symbol, code in symbolCodeMapping.items():
            compressedDB[code] += D[symbol]
    else:
        for symbol, code in symbolCodeMapping.items():
            compressedDB[code] += D[int(symbol)]

    return symbolCodeMapping, compressedDB


# Local differential privacy algorithm on a compressed databse


def LDP_compressed(D: Dict[str, int], epsilon=.1):
    """Local differential privacy applied on a compressed database D.

    Args:
        D (dict): A dictionary which maps Huffman-compressed symbols to patients
        epsilon (float, optional): Probability to flip a bit. Defaults to .1.
    """

    assert epsilon <= 1, "The probability to flip a bit must be smaller than or equal to 1."

    D_LDP = defaultdict(int)
    for key, val in D.items():
        # Decomposition of index into bits
        bits = list()
        for bit in key:
            bits.append(int(bit))
        for _ in range(val):
            # The outcome 1 means "flip", the outcome 0 means "don't flip"
            flips = list()
            for bit in bits:
                flip = np.random.binomial(1, epsilon)
                flips.append(flip)
            # The new index calculation
            newIndex = "".join(str(abs(bits[i]-flips[i]))
                               for i in range(len(bits)))
            # Increase the count of the new entry
            D_LDP[newIndex] += 1

    return D_LDP


def ComputeEntropy(D: Dict[str, int]):
    """Compute the entropy of a given database

    Args:
        D (dict): Database in a dictionary format.
    """

    freq = list(D.values())
    freq = np.array(freq)
    # Normalize the probabilities
    freq = freq/freq.sum()
    # Remove the entries with 0 to avoid 0*log(0)
    freq = freq[freq != 0]
    entropy = freq*np.log2(freq)
    entropy = entropy.sum()*-1
    return entropy


if __name__ == "__main__":
    # The mapping from the database index to the FEATURES space
    # MAPPING_DICT = {
    #    0: frozenset({}),
    #    1: frozenset({"Cold"}),
    #    2: frozenset({"Flu"}),
    #    3: frozenset({"Flu", "Cold"}),
    #    4: frozenset({"COVID"}),
    #    5: frozenset({"COVID", "Cold"}),
    #    6: frozenset({"COVID", "Flu"}),
    #    7: frozenset({"COVID", "Flu", "Cold"}),
    # }
    D = generateDB(1000, [.26, .24, .1, .1, .2, .05, .04, .01])
    print(ComputeEntropy({"0": 10, "1": 100}))

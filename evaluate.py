from typing import List
from collections import Counter


truth = {"1-1.jpg": "沪EWM957", "1-2.jpg": "沪AF02976", "1-3.jpg": "鲁NBK268",
         "2-1.jpg": "沪EWM957", "2-2.jpg": "豫B20E68",  "2-3.jpg": "沪A93S20",
         "3-1.jpg": "沪EWM957", "3-2.jpg": "沪ADE6598", "3-3.jpg": "皖SJ6M07"}


def compute_accuracy(res: List[List[str]]) -> None:
    """Computes and prints the accuracy of the recognition.

    Args:
        res (List[List[str]]): Recognition results.
    """

    print('*'*20 + "Accuracy" + '*'*20)
    cnt = 0
    for r in res:
        if r[1] == truth[r[0]]:
            cnt += 1
            correct = "Correct"
        else:
            correct = "Wrong"
        print(f"{r[0]}: {correct}")
    
    print(f"Accuracy = {cnt}/{len(res)} = {round(cnt/len(res),3)}")


def compute_char_jac_sim(res: List[List[str]]) -> None:
    """Computes and prints the Jaccard similarity of characters.

    Args:
        res (List[List[str]]): Recognition results.
    """

    print('*'*13 + "Jaccard similarity of characters" + '*'*13)
    summation = 0
    for r in res:
        rec = Counter(r[1])
        tru = Counter(truth[r[0]])
        inter = rec & tru
        union = rec | tru
        this_sim = sum(inter.values()) / sum(union.values())
        summation += this_sim
        print(f"{r[0]}: {round(this_sim,3)}")
    
    print(f"Averaged Jaccard similarity of characters = {round(summation/len(res),3)}")


def compute_2gram_jac_sim(res: List[List[str]]) -> None:
    """Computes and prints the Jaccard similarity of 2-grams.

    Args:
        res (List[List[str]]): Recognition results.
    """

    print('*'*13 + "Jaccard similarity of 2-grams" + '*'*13)
    summation = 0
    for r in res:
        rec = Counter([r[1][i:i+2] for i in range(len(r[1])-1)])
        tru = Counter([truth[r[0]][i:i+2] for i in range(len(truth[r[0]])-1)])
        inter = rec & tru
        union = rec | tru
        this_sim = sum(inter.values()) / sum(union.values())
        summation += this_sim
        print(f"{r[0]}: {round(this_sim,3)}")
    
    print(f"Averaged Jaccard similarity of 2-grams = {round(summation/len(res),3)}")

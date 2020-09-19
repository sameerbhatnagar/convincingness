# Author: Antoine Lefebvre-Brossard
# https://gitlab.com/antoinelb/simple-elo/-/blob/master/simple_elo/elo.py

from typing import Dict, List, Optional, Tuple, TypeVar, Union, cast

I = TypeVar("I")


def compute_expected_result(
    items: Dict[I, int], pairings: Optional[List[Tuple[I, I]]] = None
) -> Dict[Tuple[I, I], float]:
    """
    Computes the expected result for the given `pairings` if given or for all
    pairings if not. This will give 2 · len(pairings) if provided or
    len(items) · (len(items) - 1) if not. The expected score is given by
    E_A = 1 / (1 + 10^((R_B - R_A) / 400)), where R_X is the rating of item X.

    Parameters
    ----------
    items : Dict[I, int]
        Each item with its current rating
    pairings : Optional[List[Tuple[I, I]]] (default : None)
        Specific wanted pairings. If None, all possible pairings are computed

    Returns
    -------
    Dict[Tuple[I, I], float]
        Expected score between each pairing
    """
    if pairings is None:
        return {
            (item_a, item_b): 1
            / (1 + 10 ** ((items[item_b] - items[item_a]) / 400))
            for i, item_a in enumerate(list(items.keys())[:-1])
            for item_b in list(items)[i + 1 :]
        }
    else:
        return {
            (item_a, item_b): 1
            / (1 + 10 ** ((items[item_b] - items[item_a]) / 400))
            for item_a, item_b in pairings
        }


def compute_updated_ratings(
    items: Dict[I, int],
    results: Dict[Tuple[I, I], float],
    expected_results: Optional[Dict[Tuple[I, I], float]] = None,
    adjustment_factor: Union[int, Dict[Tuple[I, I], int]] = 24,
) -> Dict[I, int]:
    """
    Computes the updated ratings for each item with a result. If a constant
    adjustment factor is used, this rating is given by the formula:
    R'_A = R_A + K(𝜮S_A,x - ΣE_A,x),
    where R_A is the current rating, K is the adjustment factor, S_A,x is the
    score between A and another item x and E_A,x is the expected score between
    A and another item x. If K depends on the pairing, the formula becomes:
    R'_A = R_A + 𝜮K_A,x(S_A,x - E_A,x).

    Parameters
    ----------
    items : Dict[I, int]
        Each item with its current rating
    results : Dict[Tuple[I, I], float]
        Observed results where each result should be between 0 and 1
    expected_results : Optional[Dict[Tuple[I, I], float]] (default : None)
        Expected results. If None, will be computed for each pair in `results`
    adjustment_factor : Union[int, Dict[Tuple[I, I], int]] (default : 24)
        Constant adjustment factor to use or per pairing adjustment factor

    Returns
    -------
    Dict[I, int]
        Updated rating for each item
    """
    if any(res < 0 or res > 1 for res in results.values()):
        raise ValueError("The results should be between 0 and 1.")
    if expected_results is None:
        expected_results = compute_expected_result(
            items, pairings=list(results.keys())
        )

    return {
        item: value
        + round(
            adjustment_factor
            * (
                sum(
                    cast(float, result if pair[0] == item else (1 - result))
                    for pair, result in results.items()
                    if item in pair
                )
                - sum(
                    result if pair[0] == item else (1 - result)
                    for pair, result in expected_results.items()
                    if item in pair
                )
            )
            if isinstance(adjustment_factor, int)
            else sum(
                (
                    adjustment_factor[pair]
                    if pair in adjustment_factor
                    else adjustment_factor[(pair[1], pair[0])]
                )
                * (
                    (result if pair[0] == item else (1 - result))
                    - (
                        expected_results[pair]
                        if pair in expected_results and pair[0] == item
                        else 1 - expected_results[pair]
                        if pair in expected_results
                        else expected_results[(pair[1], pair[0])]
                        if pair[1] == item
                        else 1 - expected_results[(pair[1], pair[0])]
                    )
                )
                for pair, result in results.items()
                if item in pair
            )
        )
        for item, value in items.items()
    }


def get_initial_ratings(items: List[I]) -> Dict[I, int]:
    """
    Gives the initial rating for each item.

    Parameters
    ----------
    items : List[I]
        Item for which to set the rating

    Returns
    -------
    Dict[I, int]
        Items with their initial rating
    """
    return {item: 1500 for item in items}

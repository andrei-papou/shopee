from typing import List


def float_range(start: float, stop: float, step: float, precision: int = 10, exp: bool = False) -> List[float]:
    val_list = []
    corr_factor = 10 ** precision
    curr = start
    while curr <= stop:
        val_list.append(curr)
        curr = curr * step if exp else curr + step
        curr = round(curr * corr_factor) / corr_factor
    return val_list

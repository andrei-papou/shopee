import typing as t
from copy import deepcopy


def postprocess_matches_dict(matches_dict: t.Dict[str, t.Set[str]]) -> t.Dict[str, t.Set[str]]:
    too_frequent_pid_set: t.Set[str] = {
        pid for pid in matches_dict.keys()
        if len([matches_set for matches_set in matches_dict.values() if pid in matches_set]) > 50
    }
    return {
        pid: matches_set.difference({p for p in too_frequent_pid_set if p != pid})
        for pid, matches_set in matches_dict.items()
    }


def build_transitive_relations(matches_dict: t.Dict[str, t.Set[str]], rounds: int = 1) -> t.Dict[str, t.Set[str]]:
    for i in range(rounds):
        added = 0
        new_matches_dict = deepcopy(matches_dict)
        for o_pid, matches_set in matches_dict.items():
            for i_pid in matches_set:
                if o_pid not in new_matches_dict[i_pid]:
                    new_matches_dict[i_pid].add(o_pid)
                    added += 1
        matches_dict = new_matches_dict
        print(f'Transitive round {i}, added {added} matches.')
    return matches_dict

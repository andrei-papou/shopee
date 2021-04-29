import typing as t


def postprocess_matches_dict(matches_dict: t.Dict[str, t.Set[str]]) -> t.Dict[str, t.Set[str]]:
    too_frequent_pid_set: t.Set[str] = {
        pid for pid in matches_dict.keys()
        if len([matches_set for matches_set in matches_dict.values() if pid in matches_set]) > 50
    }
    return {
        pid: matches_set.difference({p for p in too_frequent_pid_set if p != pid})
        for pid, matches_set in matches_dict.items()
    }

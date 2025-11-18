def elbow_method(scores: list[float]) -> float:
    """
    Find threshold using elbow method on list of scores.

    :param scores: List of float scores.
    :return: Threshold score determined by elbow method.
    :rtype: float
    """
    scores = sorted(scores, reverse=True)
    if len(scores) < 2:
        return scores[0] if scores else 0.0
    diffs = [scores[i] - scores[i+1] for i in range(len(scores)-1)]
    if not diffs:
        return scores[0]
    elbow_index = diffs.index(max(diffs))
    threshold = scores[elbow_index]
    return threshold
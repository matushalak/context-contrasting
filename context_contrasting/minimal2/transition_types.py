from collections.abc import Mapping


def single_state_scalar(
        NO_response: float,
        O_response: float,
        activity_threshold: float = 0.025) -> tuple[float, bool]:
    """
    Scalar FF-vs-FB state summary for a single phase.
    """
    no_active = NO_response > activity_threshold
    o_active = O_response > activity_threshold
    return (
        (NO_response * no_active) - (O_response * o_active),
        no_active or o_active,
    )


def neuron_state_metrics(
        NO_response_naive:float, O_response_naive:float,
        NO_response_expert:float, O_response_expert:float,
        activity_threshold:float = 0.025) -> dict[str, tuple[float, bool]]:
    '''
    Metrics to quantify different transition types based on changes in response 
    to full, non-occluded (NO) stimulus vs occluded (O) stimulus after
    familiarization with full non-occluded stimulus.
    
    Metrics:
    - Naive State: (NO_response_naive - O_response_naive) * 1[Naive State > threshold]
    - Expert State: (NO_response_expert - O_response_expert) * 1[Expert State > threshold]

    '''
    # If responsive, positive state indicates FF-driven, 
    # negative state indicates FB-driven. 
    # If not responsive, state is 0.
    naive_state, naive_responsive = single_state_scalar(
        NO_response_naive,
        O_response_naive,
        activity_threshold=activity_threshold,
    )
    expert_state, expert_responsive = single_state_scalar(
        NO_response_expert,
        O_response_expert,
        activity_threshold=activity_threshold,
    )

    return {
        "naive": (naive_state, naive_responsive),
        "expert": (expert_state, expert_responsive)
            }


def num_state_to_category(num_state:float, responsive:bool) -> str:
    if num_state > 0 and responsive:
        return "FF"
    elif num_state < 0 and responsive:
        return "FB"
    elif not responsive:
        return "unresponsive"


def transitions(
        NO_response_naive:float, O_response_naive:float,
        NO_response_expert:float, O_response_expert:float,
        activity_threshold:float = 0.025) -> str:
    '''
    Classify transition type based on changes in response to full, non-occluded (NO) stimulus vs occluded (O) stimulus after
    familiarization with full non-occluded stimulus.
    
    Transition types:
    - FF: responsive to full stimulus in both naive and expert states, and more responsive to full than occluded stimulus
    - FB: responsive to full stimulus in both naive and expert states, but more responsive to occluded than full stimulus
    - unresponsive: not responsive to either stimulus in either state

    Total of 9 possible transition types
    '''
    metrics = neuron_state_metrics(NO_response_naive, O_response_naive, NO_response_expert, O_response_expert, activity_threshold)
    
    naive_category = num_state_to_category(*metrics["naive"])
    expert_category = num_state_to_category(*metrics["expert"])

    return f"{naive_category} -> {expert_category}"


STATE_CATEGORY_LABELS = {
    "FF": "FF",
    "FB": "FB",
    "un": "unresponsive",
    "unresponsive": "unresponsive",
}


def canonical_state_label(state_label: str) -> str:
    try:
        return STATE_CATEGORY_LABELS[state_label]
    except KeyError as exc:
        raise ValueError(
            f"Unknown state label {state_label!r}. Expected one of {sorted(STATE_CATEGORY_LABELS)}."
        ) from exc


def split_transition_label(transition_label: str) -> tuple[str, str]:
    parts = [part.strip() for part in transition_label.split("->")]
    if len(parts) != 2:
        raise ValueError(
            f"Transition label must look like 'FF -> FB', got {transition_label!r}."
        )
    return canonical_state_label(parts[0]), canonical_state_label(parts[1])


def transition_point(
        NO_response_naive: float,
        O_response_naive: float,
        NO_response_expert: float,
        O_response_expert: float,
        activity_threshold: float = 0.025) -> tuple[float, float]:
    """
    Continuous 2D transition-space coordinate.

    The first axis is the naive-state scalar and the second axis is the
    expert-state scalar. Positive values indicate FF-dominant responses,
    negative values indicate FB-dominant responses, and values close to 0
    indicate weak or unresponsive behavior.
    """
    metrics = neuron_state_metrics(
        NO_response_naive,
        O_response_naive,
        NO_response_expert,
        O_response_expert,
        activity_threshold=activity_threshold,
    )
    naive_state, _ = metrics["naive"]
    expert_state, _ = metrics["expert"]
    return naive_state, expert_state


def transition_point_distance(
        target_point: tuple[float, float],
        observed_point: tuple[float, float]) -> float:
    """
    Euclidean distance in the `(naive_state, expert_state)` plane.
    """
    dx = observed_point[0] - target_point[0]
    dy = observed_point[1] - target_point[1]
    return (dx ** 2 + dy ** 2) ** 0.5


def state_match_score(
        target_state: str,
        NO_response: float,
        O_response: float,
        activity_threshold: float = 0.025) -> float:
    """
    Scalar score for how well a pair of responses matches a desired state.

    Positive values indicate a better match. The score is expressed in the same
    units as the responses, so margins above the activity threshold matter.
    """
    target_state = canonical_state_label(target_state)
    full_margin = NO_response - O_response
    occluded_margin = O_response - NO_response
    max_response = max(NO_response, O_response)

    if target_state == "FF":
        return min(full_margin, max_response - activity_threshold)
    if target_state == "FB":
        return min(occluded_margin, max_response - activity_threshold)
    return activity_threshold - max_response


def transition_match_score(
        target_transition: str,
        NO_response_naive: float,
        O_response_naive: float,
        NO_response_expert: float,
        O_response_expert: float,
        activity_threshold: float = 0.025) -> float:
    """
    Scalar transition score for search/inference.

    The score is the sum of naive-state and expert-state match scores for the
    requested target transition. Higher is better.
    """
    naive_target, expert_target = split_transition_label(target_transition)
    naive_score = state_match_score(
        naive_target,
        NO_response=NO_response_naive,
        O_response=O_response_naive,
        activity_threshold=activity_threshold,
    )
    expert_score = state_match_score(
        expert_target,
        NO_response=NO_response_expert,
        O_response=O_response_expert,
        activity_threshold=activity_threshold,
    )
    return naive_score + expert_score


def transition_distance(
        target_transition: str,
        NO_response_naive: float,
        O_response_naive: float,
        NO_response_expert: float,
        O_response_expert: float,
        activity_threshold: float = 0.025) -> float:
    """
    Convenience minimization objective derived from ``transition_match_score``.
    """
    return -transition_match_score(
        target_transition=target_transition,
        NO_response_naive=NO_response_naive,
        O_response_naive=O_response_naive,
        NO_response_expert=NO_response_expert,
        O_response_expert=O_response_expert,
        activity_threshold=activity_threshold,
    )


def transition_profile_from_summary(
        summary: Mapping[str, float],
        activity_threshold: float = 0.025) -> dict[str, str]:
    """
    Build familiar/novel transition labels from a compact response summary.
    """
    return {
        "familiar": transitions(
            summary["full_familiar_naive"],
            summary["occlusion_familiar_naive"],
            summary["full_familiar_expert"],
            summary["occlusion_familiar_expert"],
            activity_threshold=activity_threshold,
        ),
        "novel": transitions(
            summary["full_novel_naive"],
            summary["occlusion_novel_naive"],
            summary["full_novel_expert"],
            summary["occlusion_novel_expert"],
            activity_threshold=activity_threshold,
        ),
    }


def transition_points_from_summary(
        summary: Mapping[str, float],
        activity_threshold: float = 0.025) -> dict[str, tuple[float, float]]:
    """
    Return the familiar and novel coordinates in transition space.
    """
    return {
        "familiar": transition_point(
            summary["full_familiar_naive"],
            summary["occlusion_familiar_naive"],
            summary["full_familiar_expert"],
            summary["occlusion_familiar_expert"],
            activity_threshold=activity_threshold,
        ),
        "novel": transition_point(
            summary["full_novel_naive"],
            summary["occlusion_novel_naive"],
            summary["full_novel_expert"],
            summary["occlusion_novel_expert"],
            activity_threshold=activity_threshold,
        ),
    }


INDEPENDENT_STATE_ORDER = ("FF", "FB", "un", "FF&FB")
NAIVE_SUBCLASS_ORDER = ("broad", "narrow_familiar", "narrow_novel", "unresponsive")


def ff_excess_response(
        full_response: float,
        occlusion_response: float) -> float:
    """
    FF-specific component above the occlusion-driven baseline.

    A diagonal point with ``full ~= occlusion`` therefore has near-zero FF
    excess and should not be treated as a genuine simultaneous FF and FB
    response.
    """
    return full_response - occlusion_response


def contrast_component_state_label(
        full_response: float,
        occlusion_response: float,
        activity_threshold: float = 0.025) -> str:
    """
    Endpoint label using FF excess and raw occlusion response.

    - ``FF``: FF excess active, occlusion inactive
    - ``FB``: occlusion active, FF excess inactive
    - ``un``: neither active
    - ``FF&FB``: both active
    """
    ff_excess = ff_excess_response(full_response, occlusion_response)
    ff_active = ff_excess > activity_threshold
    fb_active = occlusion_response > activity_threshold

    if ff_active and fb_active:
        return "FF&FB"
    if ff_active:
        return "FF"
    if fb_active:
        return "FB"
    return "un"


def independent_state_label(
        FF_response: float,
        FB_response: float,
        activity_threshold: float = 0.025) -> str:
    """
    Independent endpoint label based on full-vs-occluded activity.

    Unlike the signed FF-vs-FB state scalar, this treats FF and FB activity as
    separate binary dimensions. The returned label is one of:
    - ``FF``: full active, occluded inactive
    - ``FB``: occluded active, full inactive
    - ``un``: neither active
    - ``FF&FB``: both active
    """
    ff_active = FF_response > activity_threshold
    fb_active = FB_response > activity_threshold

    if ff_active and fb_active:
        return "FF&FB"
    if ff_active:
        return "FF"
    if fb_active:
        return "FB"
    return "un"


def cross_image_subclass_label(
        familiar_response: float,
        novel_response: float,
        activity_threshold: float = 0.025) -> str:
    """
    Broad/narrow subclass label for one response component across images.

    This is used independently for the naive FF component (full responses) and
    the naive FB component (occluded responses).
    """
    familiar_active = familiar_response > activity_threshold
    novel_active = novel_response > activity_threshold

    if familiar_active and novel_active:
        return "broad"
    if familiar_active:
        return "narrow_familiar"
    if novel_active:
        return "narrow_novel"
    return "unresponsive"


def scalar_state_profile_from_summary(
        summary: Mapping[str, float],
        activity_threshold: float = 0.025) -> dict[str, float | bool | str]:
    """
    Return per-image scalar state summaries useful for search and visualization.
    """
    familiar_metrics = neuron_state_metrics(
        summary["full_familiar_naive"],
        summary["occlusion_familiar_naive"],
        summary["full_familiar_expert"],
        summary["occlusion_familiar_expert"],
        activity_threshold=activity_threshold,
    )
    novel_metrics = neuron_state_metrics(
        summary["full_novel_naive"],
        summary["occlusion_novel_naive"],
        summary["full_novel_expert"],
        summary["occlusion_novel_expert"],
        activity_threshold=activity_threshold,
    )

    familiar_naive, familiar_naive_responsive = familiar_metrics["naive"]
    familiar_expert, familiar_expert_responsive = familiar_metrics["expert"]
    novel_naive, novel_naive_responsive = novel_metrics["naive"]
    novel_expert, novel_expert_responsive = novel_metrics["expert"]

    return {
        "familiar_naive_state": familiar_naive,
        "familiar_expert_state": familiar_expert,
        "novel_naive_state": novel_naive,
        "novel_expert_state": novel_expert,
        "familiar_naive_responsive": familiar_naive_responsive,
        "familiar_expert_responsive": familiar_expert_responsive,
        "novel_naive_responsive": novel_naive_responsive,
        "novel_expert_responsive": novel_expert_responsive,
        "familiar_naive_ff_scalar": max(familiar_naive, 0.0),
        "familiar_naive_fb_scalar": max(-familiar_naive, 0.0),
        "familiar_expert_ff_scalar": max(familiar_expert, 0.0),
        "familiar_expert_fb_scalar": max(-familiar_expert, 0.0),
        "novel_naive_ff_scalar": max(novel_naive, 0.0),
        "novel_naive_fb_scalar": max(-novel_naive, 0.0),
        "novel_expert_ff_scalar": max(novel_expert, 0.0),
        "novel_expert_fb_scalar": max(-novel_expert, 0.0),
        "familiar_transition": transitions(
            summary["full_familiar_naive"],
            summary["occlusion_familiar_naive"],
            summary["full_familiar_expert"],
            summary["occlusion_familiar_expert"],
            activity_threshold=activity_threshold,
        ),
        "novel_transition": transitions(
            summary["full_novel_naive"],
            summary["occlusion_novel_naive"],
            summary["full_novel_expert"],
            summary["occlusion_novel_expert"],
            activity_threshold=activity_threshold,
        ),
    }


def independent_response_profile_from_summary(
        summary: Mapping[str, float],
        activity_threshold: float = 0.025) -> dict[str, float | bool | str]:
    """
    Return independent FF/FB activity labels and naive cross-image subclasses.

    These labels do not collapse FF and FB into a single signed scalar. They
    are designed for analyses conditioned on the final expert state.
    The FF channel is defined as ``full - occlusion`` and the FB channel as the
    raw occlusion response.
    """
    familiar_naive_ff_excess = ff_excess_response(
        summary["full_familiar_naive"],
        summary["occlusion_familiar_naive"],
    )
    familiar_naive_ff_active = familiar_naive_ff_excess > activity_threshold
    familiar_naive_fb_active = summary["occlusion_familiar_naive"] > activity_threshold
    familiar_expert_ff_excess = ff_excess_response(
        summary["full_familiar_expert"],
        summary["occlusion_familiar_expert"],
    )
    familiar_expert_ff_active = familiar_expert_ff_excess > activity_threshold
    familiar_expert_fb_active = summary["occlusion_familiar_expert"] > activity_threshold

    novel_naive_ff_excess = ff_excess_response(
        summary["full_novel_naive"],
        summary["occlusion_novel_naive"],
    )
    novel_naive_ff_active = novel_naive_ff_excess > activity_threshold
    novel_naive_fb_active = summary["occlusion_novel_naive"] > activity_threshold
    novel_expert_ff_excess = ff_excess_response(
        summary["full_novel_expert"],
        summary["occlusion_novel_expert"],
    )
    novel_expert_ff_active = novel_expert_ff_excess > activity_threshold
    novel_expert_fb_active = summary["occlusion_novel_expert"] > activity_threshold

    return {
        "familiar_naive_ff_excess": familiar_naive_ff_excess,
        "familiar_expert_ff_excess": familiar_expert_ff_excess,
        "novel_naive_ff_excess": novel_naive_ff_excess,
        "novel_expert_ff_excess": novel_expert_ff_excess,
        "familiar_naive_ff_active": familiar_naive_ff_active,
        "familiar_naive_fb_active": familiar_naive_fb_active,
        "familiar_expert_ff_active": familiar_expert_ff_active,
        "familiar_expert_fb_active": familiar_expert_fb_active,
        "novel_naive_ff_active": novel_naive_ff_active,
        "novel_naive_fb_active": novel_naive_fb_active,
        "novel_expert_ff_active": novel_expert_ff_active,
        "novel_expert_fb_active": novel_expert_fb_active,
        "familiar_naive_component_state": contrast_component_state_label(
            summary["full_familiar_naive"],
            summary["occlusion_familiar_naive"],
            activity_threshold=activity_threshold,
        ),
        "familiar_expert_component_state": contrast_component_state_label(
            summary["full_familiar_expert"],
            summary["occlusion_familiar_expert"],
            activity_threshold=activity_threshold,
        ),
        "novel_naive_component_state": contrast_component_state_label(
            summary["full_novel_naive"],
            summary["occlusion_novel_naive"],
            activity_threshold=activity_threshold,
        ),
        "novel_expert_component_state": contrast_component_state_label(
            summary["full_novel_expert"],
            summary["occlusion_novel_expert"],
            activity_threshold=activity_threshold,
        ),
        "naive_ff_subclass": cross_image_subclass_label(
            familiar_naive_ff_excess,
            novel_naive_ff_excess,
            activity_threshold=activity_threshold,
        ),
        "naive_fb_subclass": cross_image_subclass_label(
            summary["occlusion_familiar_naive"],
            summary["occlusion_novel_naive"],
            activity_threshold=activity_threshold,
        ),
    }

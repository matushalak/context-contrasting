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

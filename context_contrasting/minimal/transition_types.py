
def neuron_state_metrics(
        NO_response_naive:float, O_response_naive:float, 
        NO_response_expert:float, O_response_expert:float,
        activity_threshold:float = 0.025) -> dict[str, float]:
    '''
    Metrics to quantify different transition types based on changes in response 
    to full, non-occluded (NO) stimulus vs occluded (O) stimulus after
    familiarization with full non-occluded stimulus.
    
    Metrics:
    - Naive State: (NO_response_naive - O_response_naive) * 1[Naive State > threshold]
    - Expert State: (NO_response_expert - O_response_expert) * 1[Expert State > threshold]

    '''
    no_naive, no_expert, o_naive, o_expert = (NO_response_naive > activity_threshold,
                                              NO_response_expert > activity_threshold,
                                              O_response_naive > activity_threshold,
                                              O_response_expert > activity_threshold)
    
    naive_state = (NO_response_naive * no_naive) - (O_response_naive * o_naive)
    expert_state = (NO_response_expert * no_expert) - (O_response_expert * o_expert)

    return {
        "naive": (naive_state, no_naive or o_naive),
        "expert": (expert_state, no_expert or o_expert)
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


# scalar for transition for SBI

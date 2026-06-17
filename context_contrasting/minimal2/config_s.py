import torch.nn as nn

from context_contrasting.utils import ThresholdReLU


def _copy_init_dict(init_dict: dict) -> dict:
    return {
        "mu": init_dict["mu"],
        "sigma": init_dict["sigma"],
    }


def _normalize_minimal_config(config: dict) -> dict:
    normalized = config.copy()
    normalized.setdefault("w_pv_lat_init", _copy_init_dict(normalized["w_lat_init"]))
    return normalized

# Broadly tuned: Familiar -> FB responses, Novel -> FF & FB responses
# X not seen in experimental data
broad = {
    "n_features": 3,
    "n_pv": 1,
    "n_context": 3,
    "activation": ThresholdReLU(threshold=0.1, hard = False),
    "lr_ff": 0.032,
    "lr_fb": 0.0035,
    "lr_lat": 0.002,
    "lr_pv": 0.005,
    "w_ff_init": {'mu': [0.5, 0.5, 0.5], 'sigma': 0},
    "w_fb_init": {'mu': [0.05, 0.05, 0.05], 'sigma': 0},
    "w_lat_init": {'mu': [0.3,], 'sigma': 0},
    "W_pv_init": {'mu': [0.4, 0.4, 0.4], 'sigma': 0},
    "pyc_decay": 0.05,
    "pv_decay": 0.5,
    "apical_drive_threshold": 0.30,
    "apical_drive_hard": True,
    "apical_gain_strength": 8.0,
    "apical_gain_k": 5.0,
    "apical_gain_threshold": 0.0,
    "baseline_drive_sigma": 0.1,
    "pv_noise_sigma": 0.03,
    "alpha": 1.0,
    "weight_decay": 0.0,
    "seed": 42,
    "receives_context": (True, True, True),
    "FBrule": "dampened-anti-Hebbian"
}
# NOTE: simple model cannot capture broadly tuned cell adapting to multiple familiar

# 0) weak FF & FB
weak_FB = broad.copy()
weak_FB.update({
    "w_ff_init": {'mu': [0.2, 0.2, 0.2], 'sigma': 0},
    "w_fb_init": {'mu': [0.35, 0.35, 0.35], 'sigma': 0},
    "w_lat_init": {'mu': [0.01,], 'sigma': 0},
    "w_pv_lat_init": {'mu': [0.1,], 'sigma': 0},
})

weak_FF = broad.copy()
weak_FF.update({
    "w_ff_init": {'mu': [0.2, 0.2, 0.2], 'sigma': 0},
    "w_fb_init": {'mu': [0.35, 0.35, 0.35], 'sigma': 0},
    "w_lat_init": {'mu': [0.01,], 'sigma': 0},
    "w_pv_lat_init": {'mu': [0.1,], 'sigma': 0},
    "FF_plasticity": False,
    "apical_drive_threshold": 1.2,
})

# 1) unresponsive -> unresponsive; ✅ (subthreshold only PV get stronger because just FF inhibition) 
# nonresponder (subthreshold), only FF PV strengthening
nonresponder = broad.copy()
nonresponder.update({
    "w_ff_init": {'mu': [0.01, 0.01, 0.01], 'sigma': 0},
    "w_fb_init": {'mu': [0.01, 0.01, 0.01], 'sigma': 0},
    # "w_lat_init": {'mu': [1.5,], 'sigma': 0},
    'W_pv_init': {'mu': [0.01, 0.01, 0.01], 'sigma': 0},
    'receives_context': (False, False, False)
    })
# [NOTE] Might need spiking models to capture sub-threshold behavior
# [NOTE] Because just FF inhibition, no way to prevent FB responses

# idea: to prevent runaway strengthening of FF PV synapses, condition strengthening on
# co-activation of lateral synapses onto PV and feedforward synapses onto PV

# 3) unresponsive -> FB responsive ✅
# [NOTE] strengthened by other neurons being active, hard to capture in minimal 1-neuron model :| 
# especially because context independent of input
# unresponsive probably because sub-threshold
un_FB = nonresponder.copy()
un_FB.update({'receives_context': (True, True, True),
              "W_pv_init": {'mu': [0.3]*3, 'sigma': 0},
            })

# Unresponsive -> novel NO responsive via sub-drive apical gain.
# Latent novel FF drive is masked by PV/LAT at baseline. The cell still receives
# familiar FB, but the apical-drive threshold is high enough that learned
# familiar and off-diagonal novel FB remain gain-only.
un_novel_FF = nonresponder.copy()
un_novel_FF.update({
    "w_ff_init": {'mu': [0.01, 0.01, 0.95], 'sigma': 0},
    "w_fb_init": {'mu': [0.01, 0.01, 0.01], 'sigma': 0},
    "W_pv_init": {'mu': [0.65, 0.65, 1.0], 'sigma': 0},
    "w_lat_init": {'mu': [0.75,], 'sigma': 0},
    "receives_context": (True, True, True),
    "apical_drive_threshold": 0.2,
    "apical_gain_strength": 28.0,
})

# FF -> unresponsive; ✅ (simple) cells that don't receive context and only adapt
FF_un = broad.copy()
FF_un.update({
    "w_ff_init": {'mu': [0.5, 0.5, 0.5], 'sigma': 0},
    "w_fb_init": {'mu': [1e-7, 1e-7, 1e-7], 'sigma': 0},
    "w_lat_init": {'mu': [0.1,], 'sigma': 0},
    'receives_context': (False, False, False)
    })

# FF -> FB ✅
# broad - Familiar adapt and replaced by FB
FF_FB_broad_novel = broad.copy() # no reason why novel response should be adapted (boosted novel FF & FB responses)
# FF_FB_broad_novel.update({
#     "W_pv_init": {'mu': [0.7, 0.7, 0.4], 'sigma': 0},
# })

FF_FB_broad = broad.copy() # no reason why novel response should be adapted (boosted novel FF & FB responses)
FF_FB_broad.update({
    "w_ff_init": {'mu': [0.5, 0.5,0.01], 'sigma': 0},})

# narrow, familiar ✅
narrow_familiar = broad.copy()
narrow_familiar.update({
    "w_ff_init": {'mu': [0.9, 0.01, 0.01], 'sigma': 0},
    "w_lat_init": {'mu': [0.5,], 'sigma': 0},
    })

narrow_familiar_novel = narrow_familiar.copy()
narrow_familiar_novel.update({
    "w_ff_init": {'mu': [0.9, 0.01,0.9], 'sigma': 0},
    # "W_pv_init": {'mu': [0.7, 0.7, 0.4], 'sigma': 0},
})

# narrow novel ✅ strengthen FB to familiar context
# also (less) strengthened FB to unfamiliar context + also enhanced novel response (due to no adaptation + FB boost)
narrow_novel = broad.copy()
narrow_novel.update({
    "w_ff_init": {'mu': [0.01, 0.01, 0.9], 'sigma': 0},
    # "W_pv_init": {'mu': [0.7, 0.7, 0.4], 'sigma': 0},
    })

# Weak FF -> stronger FF via learned FB gain, not FF anti-Hebbian release.
# Feedforward weights are fixed; familiar FB remains gain-only, below direct-drive threshold.
weak_FF_gain = broad.copy()
weak_FF_gain.update({
    "w_ff_init": {'mu': [0.12, 0.12, 0.01], 'sigma': 0},
    "w_fb_init": {'mu': [0.01, 0.01, 0.01], 'sigma': 0},
    "W_pv_init": {'mu': [0.05, 0.05, 0.4], 'sigma': 0},
    "w_lat_init": {'mu': [0.01,], 'sigma': 0},
    "w_pv_lat_init": {'mu': [0.1,], 'sigma': 0},
    "FF_plasticity": False,
    "apical_drive_threshold": 1.2,
})

# Overview
# unresponsive -> unresponsive (subthreshold only PV get stronger because just FF inhibition)
# unresponsive -> FF (different mechanism, X minimal circuit)
# unresponsive -> FB (based on strengthened FB without own firing and release from inhibition)

# Don't discuss - rare, different mechanism; not focus
# FB -> unresponsive ??
# FB -> FF ??

# FB -> more FB, ✅
# already FB responsive, becomes even more FB responsive
FB_FB = broad.copy()
FB_FB.update({
    "w_ff_init": {'mu': [1e-7, 1e-7, 1e-7], 'sigma': 0},
    "w_fb_init": {'mu': [0.5, 0.5, 0.5], 'sigma': 0},
    "w_lat_init": {'mu': [0.99,], 'sigma': 0},
    })

minimal_configs = {
    'weak_FB': weak_FB,
    'weak_FF': weak_FF,
    # "un_un": nonresponder,
    # "un_FB": un_FB,
    # "un_novel_FF": un_novel_FF,
    # "weak_FF_gain": weak_FF_gain,
    
    # "FF_un": FF_un,
    
    # "FF_FB_broad":FF_FB_broad,
    # "FF_FB_broad_novel": FF_FB_broad_novel,
    # "FF_FB_narrow_familiar": narrow_familiar,
    # "FF_FB_narrow_familiar_novel": narrow_familiar_novel,
    # "FF_FB_narrow_novel": narrow_novel,
    
    # "FB_FB": FB_FB
}

minimal_configs3 = {
    name: _normalize_minimal_config(config)
    for name, config in minimal_configs.items()
}

# TODO: adjust plotting around this for experiment_s
# and define new broad vs narrow on the familiar images, with novel responsiveness as a separate dimension

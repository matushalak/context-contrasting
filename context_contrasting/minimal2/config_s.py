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

CANONICAL_BASELINE_DRIVE_SIGMA = 0.08

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
    "baseline_drive_sigma": CANONICAL_BASELINE_DRIVE_SIGMA,
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
    "w_ff_init": {'mu': [0.05, 0.05, 0.05], 'sigma': 0},
    "w_fb_init": {'mu': [0.22, 0.22, 0.22], 'sigma': 0},
    "W_pv_init": {'mu': [0.12, 0.12, 0.12], 'sigma': 0},
    "w_lat_init": {'mu': [0.12,], 'sigma': 0},
    "w_pv_lat_init": {'mu': [0.05,], 'sigma': 0},
    "apical_drive_threshold": 0.22,
    "apical_gain_strength": 8.0,
    "baseline_drive_sigma": CANONICAL_BASELINE_DRIVE_SIGMA,
})

weak_FF = broad.copy()
weak_FF.update({
    "w_ff_init": {'mu': [0.08, 0.08, 0.08], 'sigma': 0},
    "w_fb_init": {'mu': [0.02, 0.02, 0.02], 'sigma': 0},
    "W_pv_init": {'mu': [0.03, 0.03, 0.03], 'sigma': 0},
    "w_lat_init": {'mu': [0.02,], 'sigma': 0},
    "w_pv_lat_init": {'mu': [0.02,], 'sigma': 0},
    "ff_plasticity_scale": 0.003,
    "apical_drive_threshold": 1.2,
    "apical_gain_strength": 18.0,
    "baseline_drive_sigma": CANONICAL_BASELINE_DRIVE_SIGMA,
})

# 1) unresponsive -> unresponsive; ✅ (subthreshold only PV get stronger because just FF inhibition) 
# nonresponder (subthreshold), only FF PV strengthening
nonresponder = broad.copy()
nonresponder.update({
    "w_ff_init": {'mu': [0.01, 0.01, 0.01], 'sigma': 0},
    "w_fb_init": {'mu': [0.01, 0.01, 0.01], 'sigma': 0},
    # "w_lat_init": {'mu': [1.5,], 'sigma': 0},
    'W_pv_init': {'mu': [0.01, 0.01, 0.01], 'sigma': 0},
    'baseline_drive_sigma': 0.1,
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
              "w_fb_init": {'mu': [0.005, 0.005, 0.005], 'sigma': 0},
              "W_pv_init": {'mu': [0.18]*3, 'sigma': 0},
              "w_lat_init": {'mu': [0.4,], 'sigma': 0},
              "w_pv_lat_init": {'mu': [0.1,], 'sigma': 0},
              "apical_drive_threshold": 0.3,
              "apical_gain_strength": 12.0,
              "baseline_drive_sigma": CANONICAL_BASELINE_DRIVE_SIGMA,
            })

# Unresponsive -> novel NO responsive via sub-drive apical gain.
# Latent novel FF drive is masked by PV/LAT at baseline. The cell still receives
# familiar FB, but the apical-drive threshold is high enough that learned
# familiar and off-diagonal novel FB remain gain-only.
un_novel_FF = nonresponder.copy()
un_novel_FF.update({
    "w_ff_init": {'mu': [0.005, 0.005, 0.16], 'sigma': 0},
    "w_fb_init": {'mu': [0.02, 0.02, 0.02], 'sigma': 0},
    "W_pv_init": {'mu': [0.03, 0.03, 0.03], 'sigma': 0},
    "w_lat_init": {'mu': [0.03,], 'sigma': 0},
    "w_pv_lat_init": {'mu': [0.03,], 'sigma': 0},
    "receives_context": (True, True, True),
    "ff_plasticity_scale": 0.003,
    "apical_drive_threshold": 1.2,
    "apical_gain_strength": 22.0,
    "baseline_drive_sigma": CANONICAL_BASELINE_DRIVE_SIGMA,
})

# FF -> unresponsive; ✅ (simple) cells that don't receive context and only adapt
FF_un = broad.copy()
FF_un.update({
    "w_ff_init": {'mu': [0.5, 0.5, 0.3], 'sigma': 0},
    "w_fb_init": {'mu': [1e-7, 1e-7, 1e-7], 'sigma': 0},
    "w_lat_init": {'mu': [0.1,], 'sigma': 0},
    'receives_context': (False, False, False)
    })

# FF -> FB ✅
# broad - Familiar adapt and replaced by FB
FF_FB_broad_novel = broad.copy() # no reason why novel response should be adapted (boosted novel FF & FB responses)
FF_FB_broad_novel.update({
    "w_fb_init": {'mu': [0.03, 0.03, 0.03], 'sigma': 0},
    "W_pv_init": {'mu': [0.1, 0.1, 0.05], 'sigma': 0},
    "w_lat_init": {'mu': [0.1,], 'sigma': 0},
    "w_pv_lat_init": {'mu': [0.4,], 'sigma': 0},
    "apical_drive_threshold": 0.3,
    "apical_gain_strength": 8.0,
    "baseline_drive_sigma": CANONICAL_BASELINE_DRIVE_SIGMA,
})

FF_FB_broad = broad.copy() # no reason why novel response should be adapted (boosted novel FF & FB responses)
FF_FB_broad.update({
    "w_ff_init": {'mu': [0.5, 0.5,0.01], 'sigma': 0},
    "w_fb_init": {'mu': [0.03, 0.03, 0.03], 'sigma': 0},
    "W_pv_init": {'mu': [0.1, 0.1, 0.05], 'sigma': 0},
    "w_lat_init": {'mu': [0.1,], 'sigma': 0},
    "w_pv_lat_init": {'mu': [0.4,], 'sigma': 0},
    "apical_drive_threshold": 0.3,
    "apical_gain_strength": 8.0,
    "baseline_drive_sigma": CANONICAL_BASELINE_DRIVE_SIGMA,
})

# narrow, familiar ✅
narrow_familiar = broad.copy()
narrow_familiar.update({
    "w_ff_init": {'mu': [0.45, 0.01, 0.01], 'sigma': 0},
    "w_fb_init": {'mu': [0.02, 0.02, 0.02], 'sigma': 0},
    "W_pv_init": {'mu': [0.03, 0.03, 0.03], 'sigma': 0},
    "w_lat_init": {'mu': [0.03,], 'sigma': 0},
    "w_pv_lat_init": {'mu': [0.03,], 'sigma': 0},
    "ff_plasticity_scale": 0.003,
    "apical_drive_threshold": 1.15,
    "apical_gain_strength": 18.0,
    "baseline_drive_sigma": CANONICAL_BASELINE_DRIVE_SIGMA,
    })

narrow_familiar_novel = narrow_familiar.copy()
narrow_familiar_novel.update({
    "w_ff_init": {'mu': [0.45, 0.01, 0.45], 'sigma': 0},
    "W_pv_init": {'mu': [0.03, 0.03, 0.03], 'sigma': 0},
    "w_lat_init": {'mu': [0.03,], 'sigma': 0},
    "apical_drive_threshold": 1.15,
    "apical_gain_strength": 18.0,
    "baseline_drive_sigma": CANONICAL_BASELINE_DRIVE_SIGMA,
})

narrow_familiar_2 = narrow_familiar.copy()
narrow_familiar_2.update({
    "w_ff_init": {'mu': [0.01, 0.45, 0.01], 'sigma': 0},
})

narrow_familiar_2_novel = narrow_familiar_novel.copy()
narrow_familiar_2_novel.update({
    "w_ff_init": {'mu': [0.01, 0.45, 0.45], 'sigma': 0},
    "W_pv_init": {'mu': [0.03, 0.08, 0.005], 'sigma': 0},
    "w_lat_init": {'mu': [0.06,], 'sigma': 0},
})

# narrow novel ✅ strengthen FB to familiar context
# also (less) strengthened FB to unfamiliar context + also enhanced novel response (due to no adaptation + FB boost)
narrow_novel = broad.copy()
narrow_novel.update({
    "w_ff_init": {'mu': [0.01, 0.01, 0.35], 'sigma': 0},
    "w_fb_init": {'mu': [0.02, 0.02, 0.02], 'sigma': 0},
    "W_pv_init": {'mu': [0.03, 0.03, 0.03], 'sigma': 0},
    "w_lat_init": {'mu': [0.03,], 'sigma': 0},
    "w_pv_lat_init": {'mu': [0.03,], 'sigma': 0},
    "ff_plasticity_scale": 0.003,
    "apical_drive_threshold": 1.2,
    "apical_gain_strength": 20.0,
    "baseline_drive_sigma": CANONICAL_BASELINE_DRIVE_SIGMA,
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
    "w_fb_init": {'mu': [0.6, 0.6, 0.6], 'sigma': 0},
    "w_lat_init": {'mu': [0.7,], 'sigma': 0},
    "w_pv_lat_init": {'mu': [0.3,], 'sigma': 0},
    "baseline_drive_sigma": CANONICAL_BASELINE_DRIVE_SIGMA,
    })

fb_fb_weak = FB_FB.copy()
fb_fb_weak.update({
    "w_lat_init": {'mu': [0.3,], 'sigma': 0},
    "w_pv_lat_init": {'mu': [0.7,], 'sigma': 0},
})

# O -> unresponsive (-O): a strong naive occluded (feedback-driven) responder
# whose surround inhibition outruns the (near-saturated) feedback during training,
# so the cell ends up an unresponsive expert. Feedback starts near saturation so
# the (1 - w_fb) damping prevents further growth; surround starts weak (naive O
# visible) but PV is strongly driven and ramps w_lat. Moderate FF feeds PV so the
# full image (NO) is already surround-suppressed below the occluded (O) response
# at naive -> the collapse is biased toward -O rather than -NO.
O_un = broad.copy()
O_un.update({
    "w_ff_init": {'mu': [0.12, 0.12, 0.02], 'sigma': 0},
    "w_fb_init": {'mu': [0.55, 0.55, 0.20], 'sigma': 0},
    "W_pv_init": {'mu': [0.45, 0.45, 0.25], 'sigma': 0},
    "w_lat_init": {'mu': [0.08,], 'sigma': 0},
    "w_pv_lat_init": {'mu': [0.6,], 'sigma': 0},
    "apical_drive_threshold": 0.13,
    "apical_gain_strength": 4.0,
    "baseline_drive_sigma": CANONICAL_BASELINE_DRIVE_SIGMA,
})

# FF -> weak FB: broadly tuned cell whose feedforward drive adapts away while a
# moderate feedback response survives, landing as a weak expert O responder
# (~ NO 0, O 0.5) that rises contiguously out of the nonresponders. The full
# image (NO) is surround-suppressed at expert via the FF->PV drive, while the
# occluded (O) response keeps the moderate feedback drive.
FF_FB_broad_weak = FF_FB_broad.copy()
FF_FB_broad_weak.update({
    "w_ff_init": {'mu': [0.40, 0.40, 0.01], 'sigma': 0},
    "w_fb_init": {'mu': [0.10, 0.10, 0.02], 'sigma': 0},
    "W_pv_init": {'mu': [0.35, 0.35, 0.10], 'sigma': 0},
    "w_lat_init": {'mu': [0.15,], 'sigma': 0},
    "w_pv_lat_init": {'mu': [0.30,], 'sigma': 0},
    "apical_drive_threshold": 0.12,
    "apical_gain_strength": 4.0,
    "baseline_drive_sigma": CANONICAL_BASELINE_DRIVE_SIGMA,
})

minimal_configs = {
    'weak_FB': weak_FB,
    'weak_FF': weak_FF,
    "un_un": nonresponder,
    "un_FB": un_FB,
    "un_novel_FF": un_novel_FF,
    
    "FF_un": FF_un,
    
    "FF_FB_broad":FF_FB_broad,
    "FF_FB_broad_weak": FF_FB_broad_weak,
    "FF_FB_broad_novel": FF_FB_broad_novel,
    "FF_FB_narrow_familiar": narrow_familiar,
    "FF_FB_narrow_familiar_2": narrow_familiar_2,
    "FF_FB_narrow_familiar_novel": narrow_familiar_novel,
    "FF_FB_narrow_familiar_2_novel": narrow_familiar_2_novel,
    "FF_FB_narrow_novel": narrow_novel,
    
    "FB_FB": FB_FB,
    "fb_fb_weak": fb_fb_weak,
    "O_un": O_un,
}

minimal_configs3 = {
    name: _normalize_minimal_config(config)
    for name, config in minimal_configs.items()
}

# TODO: adjust plotting around this for experiment_s
# and define new broad vs narrow on the familiar images, with novel responsiveness as a separate dimension

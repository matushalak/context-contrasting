# Real-data plots matched to the mean-field layout

Expert uses the native Pre to Post transition. Familiar images are 1, 2, 4, 5; novel images are 3, 6. The act reference uses the four images in the Pre to Task dataset. Its image numbering is native to that dataset and has not been assumed to match the Pre to Post image identities.

Sectors are assigned from each neuron's image-averaged transition with a 0.3 minimum displacement, independently for expert familiar, expert novel, and act. Only +NO, +O and -NO rows are plotted. These fixed memberships are then reused for every trace, image, and state within a dataset. Pooled traces first average the images for each neuron, then average the neurons. The plotted traces are baseline-subtracted source responses, with black for full (NO) and red for occluded (O); the stimulus occupies 0 to 1 s and y limits are shared within each row. Task traces are displayed from -1 to 3.05 s to match the Pre display.

Transition vectors are means of the same member neurons' image-averaged dNO and dO from the response CSVs. Those responses use 0.2 < t < 1 s relative to each stage's own t < 0 baseline. The vector CSV records exact means and cell counts. Expert and act use different recorded populations and are not paired by neuron identity.

### 2. `--gradient_accumulation_steps 32` (was 8)

**What it does:** Accumulates gradients over 32 forward passes before doing one optimizer step, simulating a larger batch.

**Why it matters:** The original paper used 4 GPUs × batch size 4 per GPU = 16 samples per step, then accumulated over 2 steps → **effective global batch = 32**. This matches the `GLOBAL_BATCH_SIZE = 32` hardcoded in ds_utils.py.

With a single RTX 4090 at batch size 1, to reach the same effective batch size:
$$1 \text{ GPU} \times 1 \text{ batch} \times 32 \text{ acc steps} = 32$$

The previous value of 8 gave an effective batch of only 8 — **4× smaller**. This causes:
- Noisier gradient estimates per update
- Different learning rate dynamics (the cosine schedule sees fewer "effective steps")
- The optimizer behaves as if learning rate is effectively higher relative to gradient noise

### Results
           	C-STANCE	FOMC 	MeetingBank	Py150	ScienceQA	NumGLUE-cm	NumGLUE-ds	20Minuten
--------------------------------------------------------------------------------------------------------------
C-STANCE   	   53.00	53.00	      39.00	35.00	    47.00	     41.00	     40.00	    30.00
FOMC       	    0.00	19.00	      25.00	 9.00	    25.00	     54.00	     32.00	    29.00
MeetingBank	    0.00	 0.00	      30.66	29.02	    29.23	     25.86	     25.52	    23.99
Py150      	    0.00	 0.00	       0.00	53.09	    51.12	     53.09	     52.83	    44.42
ScienceQA  	    0.00	 0.00	       0.00	 0.00	    67.00	     55.00	     53.00	    52.00
NumGLUE-cm 	    0.00	 0.00	       0.00	 0.00	     0.00	     25.93	     23.46	    19.75
NumGLUE-ds 	    0.00	 0.00	       0.00	 0.00	     0.00	      0.00	     42.00	    44.00
20Minuten  	    0.00	 0.00	       0.00	 0.00	     0.00	      0.00	      0.00	    38.02
All Average: 38.0551
Last Average: 35.1473
BWT: -7.4386

`remove lora depth 64 -> Performance stable`
           	C-STANCE	FOMC 	MeetingBank	Py150	ScienceQA	NumGLUE-cm	NumGLUE-ds	20Minuten
--------------------------------------------------------------------------------------------------------------
C-STANCE   	   53.00	53.00	      39.00	35.00	    47.00	     41.00	     40.00	    29.00
FOMC       	    0.00	19.00	      25.00	 9.00	    25.00	     55.00	     32.00	    26.00
MeetingBank	    0.00	 0.00	      30.65	29.27	    28.70	     26.01	     24.99	    23.08
Py150      	    0.00	 0.00	       0.00	51.46	    50.68	     51.44	     52.41	    45.41
ScienceQA  	    0.00	 0.00	       0.00	 0.00	    61.00	     56.00	     53.00	    51.00
NumGLUE-cm 	    0.00	 0.00	       0.00	 0.00	     0.00	     27.16	     22.22	    23.46
NumGLUE-ds 	    0.00	 0.00	       0.00	 0.00	     0.00	      0.00	     41.00	    44.00
20Minuten  	    0.00	 0.00	       0.00	 0.00	     0.00	      0.00	      0.00	    38.41
All Average: 37.7601
Last Average: 35.0446
BWT: -6.4160

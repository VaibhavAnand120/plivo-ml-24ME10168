
## Run 1
Score: Baseline
Changes: Ran the provided baseline model to establish a reference.
Reason: Used as the initial benchmark.

## Run 2
Score: Improved over baseline
Changes: Added energy-based features including final energy, energy decay, and relative energy.
Reason: End-of-turn speech generally shows a gradual drop in energy.

## Run 3
Score: Improved
Changes: Added pitch features including F0 slope, final pitch, and speaker-normalized pitch.
Reason: Declarative utterances often exhibit falling pitch before turn completion.

## Run 4
Score: Improved
Changes: Added prosodic features such as voiced fraction, final voiced duration, and duration ratio.
Reason: Final syllable lengthening is a useful cue for end-of-turn detection.

## Run 5
Score: Improved
Changes: Added spectral ratio and zero-crossing rate features.
Reason: Voice quality changes near utterance completion provide additional discriminative information.

## Run 6
Score: Best
Changes: Replaced Logistic Regression with HistGradientBoostingClassifier and tuned model parameters.
Reason: Nonlinear feature interactions improved overall prediction quality and reduced response delay.

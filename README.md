# gpt-1

An implementation of a GPT-1 like language model, based off of [Karpathy's video](https://youtu.be/kCc8FmEb1nY?si=GPR4WRRzxID5NbB6).

A number of changes has been made:
- Weights initialized normally, not 0
- Scaled down residual projections (combat NaNs, GPT-2 style)
- Added inference abilities, model checkpoint saving

The training data used is in `input.txt`; it is a collection of Shakespeare.

## Sample Output

> But in a merit, who indeed was to be
> piercing guilty of some sore gentleman that was something in
> him: he had lived to make his answer to do sit out of
> an unspeak to the state, of such a very pin
> here in discoverity of lead: and he hath made you a prayer life
> under the absent duke more in my sight.

> Clown:

> He soon all his heavens unknown to your pleasure.

> AUTOLYCUS:

> Agreed when I came betime I his man, or have been
> the business in the senate-house, which you are subjects a
> noisome of the old fe

## Setup

Clone the repository and create a virtual environment. Install the packages listed in `requirements.txt`. If ROCm is not required, ignore the `pytorch-triton-rocm` package and install `torch==2.12.0`.

## Usage

### Training

Run gpt.py, this will output a 'model.pt' to be used for inference.

### Inference

Run inference.py.
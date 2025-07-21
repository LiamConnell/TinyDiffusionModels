# TextDiffusion

Exploring diffusion models by building image and text implementations side by side.

## Motivation

Diffusion models have become one of the most compelling areas in generative AI, but understanding their mechanics can be elusive when working with just one modality. This project explores diffusion by implementing both image and text models in parallel, creating a clear conceptual foundation through comparison.

By seeing how the same mathematical principles apply to pixels and embeddings, the core ideas of forward/reverse diffusion, noise scheduling, and denoising become much clearer. The image implementation serves as an intuitive baseline, while the text experiments push into less explored territory.

This project has also become an unexpected exercise in how AI can enhance the research process itself. I started using claude code extensively after the first `hello-world` implementations. I use it for much more than just coding - it helps me with ML Ops, tracking experiments, writing up findings, conducting literature reviews and exploring new ideas. I hope to write up these meta-observations about AI-assisted research workflows. I think the fact that I'm not a "real" researcher at a lab might give me an interesting perspective here. I dont need to communicate with humans at all so my workflows are highly optimized for AI. 

## What I'm Exploring

- **Image Diffusion** (`src/mnist.py`): Standard DDPM on MNIST digits - the "hello world" that makes diffusion mechanics tangible
- **Text Diffusion** (`src/shakespeare.py`): Experimental embedding-space diffusion with hybrid generation modes - where things get interesting

The text experiments are particularly fascinating because they operate in embedding space rather than discrete tokens, opening questions about:
- How does diffusion work when your target isn't continuous?
- Can you blend autoregressive and diffusion generation?


## Research Journal

The real discoveries happen in the [Experiment Journal](experiments.md), where I document what works, what doesn't, and what surprises emerge from these parallel explorations.

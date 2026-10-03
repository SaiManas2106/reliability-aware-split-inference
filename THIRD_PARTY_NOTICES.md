# Third-Party Resources

This research artefact uses third-party software, datasets, and pretrained model weights. Those resources remain subject to the terms set by their respective authors and distributors.

## Software dependencies

The Python dependencies are listed in `requirements.txt`. They are not copied into this repository.

## Datasets

- CIFAR-10 is obtained by the torchvision data loader.
- UCI Covertype is obtained through the scikit-learn data loader.

Downloaded dataset files are intentionally excluded from this repository. Users are responsible for reviewing and following the applicable dataset terms.

The retained MLP checkpoint was trained using the Covertype dataset. The dataset citation is:

Blackard, J. (1998). *Covertype* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C50K5N

The UCI repository lists Covertype under the Creative Commons Attribution 4.0 International license.

## Pretrained model weights

The ResNet-18 and MobileNetV2 implementations request the default pretrained weights supplied through torchvision. Downloaded weight files are intentionally excluded from this repository and remain subject to their original terms.

## Original material in this artefact

The experiment code, preserved logs, result tables, documentation, and trained Covertype MLP checkpoint are included as research material created or retained for the thesis. `LICENSE` explains which license applies to each category of original material. Those licenses apply only where the author holds the necessary rights. The Covertype attribution and license above also apply to use of the retained MLP checkpoint.

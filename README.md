# F-ML Transfer Function

This repository contains the code for the F-ML method described in the paper titled "A Spectral Machine Learning Approach to Derive Central Aortic Pressure Waveforms from a Brachial Cuff" by Alessio Tamborini, Arian Aghilinejad and Morteza Gharib.

The F-ML method is a spectral machine learning approach designed to reconstruct a target pressure waveform from a input pressure waveform measurement where these waveforms are contained within the same system. The referenced paper demonstrates this methodology by transferring a brachial pulse waveform to an aortic waveform. However, the versatility of this method allows for transferring waveforms between any two locations.

## Installation
For installation, just clone this repository and install locally:
```bash
git clone https://github.com/aletambo/FML-TransferFunction.git
cd FML-TransferFunction
pip install -e .
pip install -r requirements.txt
```

## Quickstart
After you have installed the library, you can start training the F-ML Transfer Function method as shown below:
```python
from FML import FML

# Initialize the F-ML Transfer Function
fml = FML(
    input_modes=20,
    target_modes=20,
    name='F-ML Transfer Function', 
    samp_freq=1000
)

# Train the model with input and target data
fml.train(inputs, targets)
```

## Contributing
We welcome contributions to the project! Please follow these guidelines when contributing:

1. **Reporting Issues**: If you encounter any issues, please report them using our [issue tracker](https://github.com/aletambo/FML-TransferFunction/issues).

2. **Submitting Pull Requests**: If you would like to contribute code, please fork the repository and create a new branch for your feature or bug fix. Once your changes are ready, submit a pull request for review.

Thank you for your contributions!

## Citing
If you use this code in your research, please cite the following paper:

```
@article{tamborini2023spectral,
    title={A Spectral Machine Learning Approach to Derive Central Aortic Pressure Waveforms from a Brachial Cuff},
    author={Alessio Tamborini and Arian Aghilinejad and Morteza Gharib},
    journal={TBD},
    year={TBD},
}
```

## Contact
For any questions or inquiries, please contact:
- Alessio Tamborini: [atambori@caltech.edu](mailto:atambori@caltech.edu)
# SpikingNN Core

Computational engine for spiking neural network simulation based on the Izhikevich model.

## Features

- **Izhikevich Neuron Model**: Implementation of regular spiking (RS), intrinsically bursting (IB), chattering (CH), fast spiking (FS), thalamo-cortical (TC), resonator (RZ), and low-threshold spiking (LTS) neurons
- **Network Topology**: Flexible connectivity with configurable synaptic weights and time constants
- **Neuromechanics**: Integrated muscle models, afferent feedback (Ia, Ib, II), and limb dynamics
- **Var_Limb System**: Complete neuromechanical system combining neural networks with biomechanics

## Installation

### Development Mode

```bash
cd packages/core
pip install -e .

#Citation
```bibtex
@software{spikingnn_core,
  author = {Kovalev, Nickolai and Guba, Artem},
  title = {SpikingNN Core: Computational Engine for Spiking Neural Networks},
  year = {2024},
  url = {https://github.com/brodayagabro/SpikingNN}
}
```
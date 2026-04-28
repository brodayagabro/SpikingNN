# SpikingNN
This project provides a comprehensive computational framework for modeling and simulating Spiking Neural Networks (SNNs) based on the Izhikevich neuron model. The core library (Izh_net.py) implements biologically plausible neuron dynamics with support for multiple neuron types (RS, FS, IB, CH, etc.), configurable synaptic connectivity, and synaptic current relaxation dynamics. Beyond pure neural simulation, the framework includes neuromechanical components such as muscle models, afferent feedback pathways, and limb dynamics, enabling research into closed-loop locomotor control and Central Pattern Generators (CPGs).
## Reveiw
There will be the review of the project

## Installation
Clone repository to your local computer:
```bash
git clone https://github.com/brodayagabro/SpikingNN
```

Cd to dir with package:
```bash
cd SpikingNN
```

Install package using pip:

```
pip install .
```

With devmode:
```
pip install -e .
```

# GUI interface
Streamlit GUI Features
The interactive web interface allows users to design, simulate, and analyze neural networks without writing code. Key functionalities include:
Network Design: Create networks with customizable neuron counts and types, and edit connectivity matrices interactively.
Parameter Tuning: Adjust synaptic weights (excitatory/inhibitory), relaxation constants (τ), and individual input current vectors for each neuron.
Real-time Visualization: View membrane potentials, spike rasters, network graph structures with directed edges, and weight heatmaps.
Simulation Control: Start, stop, and reset simulations with configurable time steps and duration.
Data Management: Save and load network configurations via JSON, and export simulation results (NPZ/CSV).
CLI Support: Launch the application directly from the command line using the `spikingnn`  or `spknn` command.



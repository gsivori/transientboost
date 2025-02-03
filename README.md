# transientboost
This repository contains the code for models and figures in "Transient Boosting of Action Potential Backpropagation for Few-shot Temporal Pattern Learning".

It requires mostly Julia >1.7. Python >3.7 but was only used for certain data analysis. Standard packages are necessary for both. On Julia, these can be included from the REPL. For Python, the conda package contains all libraries used in the jupyter notebook. Runtimes differ. For a typical computer: running single models takes less than 1 min, place field simulation may take several minutes, stdp protocols may take up more 12 hours if not run in parallel, network models take up to 4 hours.

Models files are:

funs.jl: functions file -- not all functions are utilized.

lifneur.jl : two-compartmental model of LIF neuron using spike trace-based synaptic plasticity rule.

bpHVAneur.jl: same as above but for calcium-based synaptic plasticity rule and transient boosting of somato-dendritic coupling.

bp_full.jl: same as above but including NMDA receptor dynamics.

lif_burst_input.jl : two-compartmental model to explore tuning time at different input bursting units.

stdp.jl: model adapted for STDP protocol.

net.jl: recurrent network model.

Data-generating and figure-generating files:


datafigs1-3.jl: generates data for Figures 1 to 3.
suppfig*.jl, Fig*.jl, fig*.ipynb are used to generate figures. 
Jupyter notebook files (.ipynb) are zipped in the file jupyternotebooks.zip
Copyright 2023 Gaston Sivori

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

# VPINN: Variational physics-informed neural net 
### Daniel Boutros, Thomas Gaskin, Oscar de Wit

A machine learning module for solving weak non-linear PDEs, developed for the CMI
core course project at the University of Cambridge. It is based on [code](https://github.com/ehsankharazmi/hp-VPINNs) 
by Ehsan Kharazmi et al., who initially designed a machine learning tool to solve 
non-linear PDEs with weak solutions. We have extended the code independently to
also cover 2-dimensional domains, and have added several additional differential 
equations that can be solved. Furthermore, the code has been substantially reworked and 
extended. Among other things, it can now be controlled from a config file. We use
[TensorFlow 2](https://www.tensorflow.org/guide), allowing among other things 
the use of [`eager execution`](https://www.tensorflow.org/guide/function).


### Required packages

The following packages are required to run the code. We recommend installing these
into a virtal environment to avoid interfering with system-wide package installations.

| Package        | Version | Comments                        |
|----------------|---------| ------------------------------- |
| Python         | \>= 3.7 |                                 |
| Tensorflow     | \>= 2.0 | Machine learning package        |
| PyYAML         | \>= 6.0 | Handles configuration           |


### How to run

Most settings can be modified from the `config.yaml` file. To modify the external
forcing, go to the `Utils/functions.py` file and modify `f`. Test functions can also 
be modified in that file. 

### About the code

**Custom data types:** We have implemented several custom data types to facilitate handling
multidimensional data, as well as data on grids. These types are implemented in the `Utils/data_types.py`
file.

```python
class Grid(self, *, x: Sequence, y: Sequence)
```
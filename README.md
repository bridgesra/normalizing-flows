# Normalizing flows
Bobby's fork of VishakhG's NF repo. 

Two goals here: 
- [x] get it running for the 4 potentials from Rezendez et al. (originally in this repo)
- [ ] git it running fro two seemingly simple potentials. (FAILS to train :/ )

## Setup 
We'll use pyenv for managing different versions of python and venv for our python virtual envrionment. 

Quick background and useful commands appear at the bottom of this readme. 

The setup follows reference: https://www.freecodecamp.org/news/manage-multiple-python-versions-and-virtual-environments-venv-pyenv-pyvenv-a29fb00c296f/ 

### Steps for using this repo: 
1. Set pyenv envrionment to python 3.9.14 (assuming it is installed using. if not, install it using `pyenv install 3.9.14`)

        pyenv local 3.9.14

2.  Initialize virtual envrionment in the .venv folder: 
   
        python3 -m venv .venv

    in .venv/bin should be a copy of python3.9 

3. Start virtual envrionment:    

        source .venv/bin/activate 

    At this point, you should see you should see (venv) before your terminal, running `which pip` and `which python` should produce a path to the `pip` and `python` instances in `.venv/bin/`. 

    Running `python --version` should produce 3.9.14. 

    VS Code and Jupyter users may need to point them to the right interpreter. (.vinv/bin/python)

4. Install required packages from `./requirements.txt` file:

        pip install -r requirements.txt

    NOTE: if a new packages is needed then use the `pip install <package>` (which calls  `.venv/bin/pip`) to install it, and 

5. Before pushing, if new required packages were installed, these need to be added to the repo and pushed. Run 

        pip freeze > requirements.txt

    and push the new `requirements.txt`. 

    Note that the `.venv` folder is gitignored and should not ship with the repo. 


## From VishakhG's Readme:
Attempting to implement the potential function experiments from:

```
Danilo Jimenez Rezende and Shakir Mohamed. Variational inference with normalizing
flows. In Proceedings of the 32nd International Conference on Machine Learning, pages
1530–1538, 2015.
```
Other reference:

```
Papamakarios, George, et al. Normalizing Flows for Probabilistic Modeling and Inference. Dec. 2019. arxiv.org, https://arxiv.org/abs/1912.02762v1.

```

To reproduce plots run `exp/run_2d_potential_exp.sh` or take a look at `src/fit_flows.py`.

Target densities, corresponding to the 4 potentials from the paper:

![target densities](https://github.com/VishakhG/normalizing-flows/blob/master/assets/all_potentials.png)


Samples from a 2-D diagonal gaussian passed through 32 learned Planar flows:

![potential 1](https://github.com/VishakhG/normalizing-flows/blob/master/assets/pot_1_32.png)
![potential 2](https://github.com/VishakhG/normalizing-flows/blob/master/assets/pot_2_32.png)
![potential 3](https://github.com/VishakhG/normalizing-flows/blob/master/assets/pot_3_32.png)
![potential 4](https://github.com/VishakhG/normalizing-flows/blob/master/assets/pot_4_32.png)


# VQE on the Kagome Lattice

A submission for IBM Quantum's
[Open Science Prize 2022](https://research.ibm.com/blog/ibm-quantum-open-science-prize-2022).

***The code in this repo must be run with python3.10.***

Clone the repo and run the following from the cloned directory.

```sh
python3.10 -m venv venv
source venv/bin/activate
pip install -U pip
pip install -U -r requirements.txt
python -m kagomevqe -h
```

That will create the appropriate virtual environment and then print command line usage.

```
usage: python -m kagomevqe [-h] [-b {local,simulator,guadalupe,ionq}]
                           [-l {kagome-unit,asymmetric}] [-a {josephson,rotsym,combo}]
                           [-o {rotoselect,rotosolve,bfgs}] [-m MAXITER] [--no-noise]

options:
  -h, --help            show this help message and exit
  -b {local,simulator,guadalupe,ionq}, --backend {local,simulator,guadalupe,ionq}
                        Where to run. Default: local.
  -l {kagome-unit,asymmetric}, --lattice {kagome-unit,asymmetric}
                        The Hamiltonian observable. Default: kagome-unit.
  -a {josephson,rotsym,combo}, --ansatz {josephson,rotsym,combo}
                        The parameterized circuit. Default: josephson.
  -o {rotoselect,rotosolve,bfgs}, --optimizer {rotoselect,rotosolve,bfgs}
                        Variational optimization strategy. Default: rotoselect.
  -m MAXITER, --maxiter MAXITER
                        Maximum optimizer iterations. Default: 100.
  --no-noise            Don't add noise to simulations.
```

IBM and IonQ credentials are pulled from the environment, so be sure you've saved your IBM Quantum credentials in `$HOME/.qiskit` or exported your IonQ token to the environment before proceeding.

To run Rotoselect on the Kagome unit cell using a highly expressible Josephson sampler ansatz in a local simulator with the Guadalupe noise model:

```sh
python -m kagomevqe --backend local --lattice kagome-unit --ansatz josephson --optimizer rotoselect
```

Or equivalently, since those are the defaults:

```sh
python -m kagomevqe
```

The beginning of the output will look something like this:

```
Running locally
Using highly expressible Josephson sampler ansatz
Applying guadalupe noise model
04/13 11:39:21-0700 Starting
Running RotoselectVQE for 26 iterations
04/13 11:41:58-0700 Iteration 0 gate 0: (False, 'ry', 'ry')	energy:  05.09156653
04/13 11:44:38-0700 Iteration 0 gate 1: (True, 'rz', 'rx')	energy:  05.28275509
04/13 11:47:25-0700 Iteration 0 gate 2: (True, 'ry', 'rz')	energy:  01.65555962
04/13 11:50:10-0700 Iteration 0 gate 3: (True, 'rz', 'rx')	energy:  02.04865199
04/13 11:52:54-0700 Iteration 0 gate 4: (False, 'ry', 'ry')	energy:  01.73488377
04/13 11:55:33-0700 Iteration 0 gate 5: (True, 'rz', 'rx')	energy:  01.50876424
```

To perform the same algorithm against the real IBM Guadalupe backend:

```sh
python -m kagomevqe -b guadalupe
```

After some iterations through all 96 parameterized gates, the optimization will switch from Rotoselect to Rotosolve, no longer modifying gates. Iterations become much faster at that point.

To learn more, check out:

- `kagomevqe/__main__.py` for program entry, options, branching
- `kagomevqe/rotoselect_vqe.py` for the main Rotoselect implementation
- `kagomevqe/rotoselect_translator.py` for the custom transpiler pass that modifies gates during rotoselect
- `kagomevqe/ansatze.py` for classes encapsulating the different ansätze I used

Even if there is an unrecoverable error or you interrupt the program with Ctrl-C, some final output is logged and charts and data are saved to the data directory.

# pinn

## About
This repository was created to keep track of my experiments while participating in Embry-Riddle Aeronautical University's 2023 Summer Undergraduate Research Fellowship. At the moment, it is intended only for the use of those directly involved in the project.

## Branching
Branches are created to solve sub-problems (e.g. `1dof`, `2dof`, ...). `main` contains the current, most complex experiment.

## Dependencies
- `matplotlib`
- `numpy`
- `deepxde`
- `pytorch` (or other suitable machine learning framework, backend for `deepxde`)
- `scipy`

## Additional Notes
`<N>dof.py` is an executable version of `analysis.ipynb`. It exists solely to be run on the GPU, whereas the `analysis.ipynb` file is provided for better readability ease of debugging.

## Genesis version
commit id:`7db43e4caef2b185bf691d29fc545d6480cd224d`

## Genesis Modification
Genesis will raise error when encounter nan during simulation.
We do not want to nan interupt our training and want to handle them explicitly during environment.
To do so, we have modify the genesis code.

In file `genesis/engine/solvers/rigid/rigid_solver.py`,
change the line 1066&1068 from
`gs.raise_exception("$msg")`
to
`gs.warn("$msg")`

In the same file, line 7110 and 7113 comment out the nan check as
`# is_valid &= not ti.math.isnan(e)` (NOTE, there existing other nan check, e.g. grad, which are not handling currently)

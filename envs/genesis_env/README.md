## Genesis version
current commit id:`fae1db41e1271150a78940592fda6bf0c930f255`
commid id before nan error handling `d74f02c08508f67fe47df17765f46ba1f0bff33c`

## Modification
Genesis will raise error when encounter nan during simulation.
We do not want to nan interupt our training and want to handle them explicitly during environment.
To do so, we have modify the genesis code.

In file `genesis/engine/solvers/rigid/rigid_solver_decomp.py`,
change the line 961&963 from
`gs.raise_exception("$msg")`
to
`gs.warn("$msg")`

In the same file, line 6949 and 6952 comment out the nan check as
`# is_valid &= not ti.math.isnan(e)` (NOTE, there existing other nan check, e.g. grad, which are not handling currently)

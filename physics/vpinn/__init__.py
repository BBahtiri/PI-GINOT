"""Phase 10: weak-form (VPINN) machinery -- test functions, quadrature, element
maps, precomputed tensors and the finite-strain weak-form losses.

The quadrilateral path is a PyTorch port of the FastVPINNs tensor form
(Anandh et al., SIAM J. Sci. Comput.; library fastvpinns 1.0.2), verified
against the library itself (verification/phase10/g0_port.py).  The triangle
path (Lagrange test functions, Berrone, Canuto & Pintore 2022) is ours.
"""

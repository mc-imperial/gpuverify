=======================
Trouble Shooting
=======================

If you think you have found a bug in GPUVerify, please report it via
our issue tracker on GitHub: https://github.com/mc-imperial/gpuverify/issues

This page will include a list of common problems people have reported
when getting started with GPUVerify.

Configuration Error: psutil required
------------------------------------

When running GPUVerify you see an error of the form::

     __main__.ConfigurationError: GPUVerify: CONFIGURATION_ERROR error (9): psutil required. `pip install psutil` to get it.

GPUVerify requires the python module `psutil <https://github.com/giampaolo/psutil>`_.
We recommend installing this using pip::

     $ pip install psutil

.. todo:: is z3 available?

.. todo:: on Linux are all dynamic libraries installed that z3 and cvc4 were built against? Invoke each solver directly from the command line, or pass `--cruncher-opt=/proverOpt:VERBOSITY=100` to GPUVerify.

.. todo:: correct python version?

.. todo:: correct mono version if on Linux?

.. todo:: do you have write permission for the current directory?

.. todo:: Short circuit evaluation in invariants and pre-postcondition

.. todo:: Byte-level reads and writes

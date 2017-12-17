================
JSON File Format
================

The JSON file format accepted by GPUVerify has the following form::

  [ kernel_instantiation, ..., kernel_instantiation ]

where each ``kernel_instantiation`` is an object::

  {
    "language"         : "OpenCL",
    "endianness"       : string,
    "kernel_file"      : string,
    "local_size"       : [ number, ..., number ],
    "global_size"      : [ number, ..., number ],
    "global_offset"    : [ number, ..., number ],
    "compiler_flags"   : string,
    "entry_point"      : string,
    "kernel_arguments" : [ kernel_argument, ..., kernel_argument ],
    "host_api_calls"   : [ host_api_call, ..., host_api_call ]
  }

The value of ``language`` should always be the string ``OpenCL``.

The ``endianness`` can be set to ``big`` or ``little``.  This
specifies the endianness of the device on which the kernel was
captured.

The value of ``kernel_file`` is the name of an OpenCL source file
whose location is relative to the JSON file. The intended usage of
this latter value is to specify the file name of the file to which the
kernel is dumped upon interception.

If ``kernel_file`` has the value ``/some/path/foo.cl``, then if
preprocessing was applied after kernel interception the preprocessed
version of the file will be at ``/some/path/foo.pre.cl`` (where
``/some/path`` is the same as in ``kernel_file``).

The values of ``local_size``, ``global_size``, and ``global_offset``
are arrays of numbers satisfying the usual restrictions on the
``local_work_size``, ``global_work_size``, and ``global_offset``
arguments of ``clEnqueueNDRangeKernel``, respectively.

The ``compiler_flags`` value specifies the string passed to the kernel
compiler upon compilation of the kernel.

The value of ``entry_point`` specifies which kernel in the kernel
file needs to be verified.

The ``kernel_arguments`` field specifies properties of the concrete
argument values pass to the kernel during launch, where the nth
element of the array corresponds to the nth argument of the kernel.

The ``host_api_calls`` array specifies the source file locations of
relevant OpenCL host-side functions that were invoked leading up to
the original kernel launch.

The ``language``, ``kernel_file``, ``global_size``, and ``entry_point`` fields
are required.

The ``endianness``, ``local_size``, , ``global_offset``, ``kernel_arguments``
and ``host_api_calls`` are optional. Not specifiying ``local_size`` corresponds
to a ``NULL`` value being passed to ``clEnqueueNDRangeKernel`` (observe that
omitting ``local_size`` can significantly slow down verification with
GPUVerify). When ``global_offset`` is not specified, the offset is assumed to
be 0 in each dimension. When ``kernel_arguments`` is not specified the argument
values used during verification are assumed to be unconstrained.

Each ``kernel_argument`` is an object of one of the following four forms::

  {
    "type"  : "scalar",
    "value" : string
  }

::

  {
    "type" : "array",
    "size" : number,
    "address_space" : string,
    "flags" : [ string, ..., string ],
    "data" : string
  }

::

  {
    "type" : "image"
  }

::

  {
    "type" : "sampler"
  }

where the ``type`` field is the only required field.

In the case of scalars, the ``value`` field gives the argument value
passed to the kernel during launch. The value should be string
representing a hexadecimal value, e.g., ``"0xA208"``; the string
should always start with ``0x``.

In the case of arrays, the ``size`` field specifies the size of the
argument array passed to the kernel during launch. The value is a
number specifying the size of the array in bytes.  The
``address_space`` field must be one of ``global``, ``local`` and
``constant``, and specifies which memory space the array data resides
in.  The ``flags`` field provides the cl_mem_flags that were used to
create the buffer.  The ``data`` field names a binary file containing,
in bytes, the data the array was populated with on kernel launch.

If ``value`` or ``size`` is omitted their values are assumed
to be unconstrained during verification, otherwise the specific values are use.

Each ``host_api_call`` is of the following form::

  {
    "function_name"    : string,
    "compilation_unit" : string,
    "line_number"      : number
  }

All fields are required, with ``function_name`` and ``compilation_unit``
respectively specifying the name of the called function and the compilation
unit in which the call occurs. The ``line_number`` specifies the line number
within the compilation unit.

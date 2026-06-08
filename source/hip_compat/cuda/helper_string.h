#pragma once
// HIP build shim for the project's vendored <cuda/helper_string.h>. None of its
// string helpers are referenced on any code path the project compiles, so the
// HIP build needs nothing here.

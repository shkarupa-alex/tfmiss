#pragma once

namespace tensorflow {
namespace miss {

// General definition of the FoPool op, which will be specialised in:
//   - fo_pool_cpu.h for CPUs
//   - fo_pool_gpu.cuh for CUDA devices
// Concrete template instantions of this class are provided in:
//   - fo_pool_cpu.cc for CPUs
//   - fo_pool_gpu.cu for CUDA devices
template <typename Device, typename FT, bool time_major>
class FoPool
{
};

template <typename Device, typename FT, bool time_major>
class BwdFoPool
{
};

} // end namespace miss
} // namespace tensorflow

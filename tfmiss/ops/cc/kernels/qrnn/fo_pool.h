#pragma once

namespace tensorflow {
namespace miss {

// General definition of the FoPool op, which will be specialised for CPU and GPU
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

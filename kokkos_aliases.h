#ifndef kokkos_aliases
#define kokkos_aliases

// Kokkos aliases
template <typename _TYPE_>
using View = Kokkos::View<_TYPE_ *>;

using Device = Kokkos::DefaultExecutionSpace;
using Host = Kokkos::DefaultHostExecutionSpace;
using HostPinned = Kokkos::SharedHostPinnedSpace;

template <typename _TYPE_>
using HostView = Kokkos::View<_TYPE_ *, Host>;

template <typename _TYPE_>
using HostPinnedView = Kokkos::View<_TYPE_ *, HostPinned>;

using policy_t = Kokkos::RangePolicy<Device>;
using team_policy_t = Kokkos::TeamPolicy<Device>;
using member_type = team_policy_t::member_type;
using ScratchSpace = Device::scratch_memory_space;
using Unmanaged = Kokkos::MemoryTraits<Kokkos::Unmanaged>;

//Scratch views must be unmanaged
template <typename _TYPE_>
using ScratchView = Kokkos::View<_TYPE_ *, ScratchSpace, Unmanaged>;

//Subviews
template <typename _TYPE_>
using Subview = Kokkos::Subview<View<_TYPE_>>;
template <typename _TYPE_>
using HostSubview = Kokkos::Subview<HostView<_TYPE_>>;
template <typename _TYPE_>
using HostPinnedViewSubview = Kokkos::Subview<HostPinnedView<_TYPE_>>;

#endif
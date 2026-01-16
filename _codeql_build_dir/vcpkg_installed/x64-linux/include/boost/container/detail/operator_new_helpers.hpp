//////////////////////////////////////////////////////////////////////////////
//
// (C) Copyright Ion Gaztanaga 2025-2025. Distributed under the Boost
// Software License, Version 1.0. (See accompanying file
// LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//
// See http://www.boost.org/libs/container for documentation.
//
//////////////////////////////////////////////////////////////////////////////
#ifndef BOOST_CONTAINER_DETAIL_OPERATOR_NEW_HELPERS_HPP
#define BOOST_CONTAINER_DETAIL_OPERATOR_NEW_HELPERS_HPP

#ifndef BOOST_CONFIG_HPP
#  include <boost/config.hpp>
#endif

#if defined(BOOST_HAS_PRAGMA_ONCE)
#  pragma once
#endif

#include <boost/container/detail/std_fwd.hpp>
#include <boost/container/throw_exception.hpp>

namespace boost {
namespace container {
namespace dtl {

template <class T>
T* operator_new_allocate(std::size_t count)
{
   const std::size_t max_count = std::size_t(-1)/(2*sizeof(T));
   if(BOOST_UNLIKELY(count > max_count))
      throw_bad_alloc();
   #if defined(__cpp_aligned_new)
   BOOST_IF_CONSTEXPR(__STDCPP_DEFAULT_NEW_ALIGNMENT__ < alignof(T)) {
      return static_cast<T*>(::operator new(count*sizeof(T), std::align_val_t(alignof(T))));
   }
   #endif
   return static_cast<T*>(::operator new(count*sizeof(T)));
}

template <class T>
void operator_delete_deallocate(T* ptr, std::size_t n) BOOST_NOEXCEPT_OR_NOTHROW
{
   (void)n;
   #ifdef __cpp_aligned_new
   BOOST_IF_CONSTEXPR(__STDCPP_DEFAULT_NEW_ALIGNMENT__ < alignof(T)) {
      # if defined(__cpp_sized_deallocation)
      ::operator delete((void*)ptr, n * sizeof(T), std::align_val_t(alignof(T)));
      #else
      ::operator delete((void*)ptr, std::align_val_t(alignof(T)));
      # endif
      return;
   }
   #endif

   # if defined(__cpp_sized_deallocation)
   ::operator delete((void*)ptr, n * sizeof(T));
   #else
   ::operator delete((void*)ptr);
   # endif
}

}  //namespace dtl {
}  //namespace container {
}  //namespace boost {

#endif   //#ifndef BOOST_CONTAINER_DETAIL_OPERATOR_NEW_HELPERS_HPP

//////////////////////////////////////////////////////////////////////////////
//
// (C) Copyright Ion Gaztanaga 2025-2025. Distributed under the Boost
// Software License, Version 1.0. (See accompanying file
// LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//
// See http://www.boost.org/libs/container for documentation.
//
//////////////////////////////////////////////////////////////////////////////
#ifndef BOOST_CONTAINER_DEQUE_HPP
#define BOOST_CONTAINER_DEQUE_HPP

#ifndef BOOST_CONFIG_HPP
#  include <boost/config.hpp>
#endif

#if defined(BOOST_HAS_PRAGMA_ONCE)
#  pragma once
#endif

#include <boost/container/detail/config_begin.hpp>
#include <boost/container/detail/workaround.hpp>
// container
#include <boost/container/allocator_traits.hpp>
#include <boost/container/container_fwd.hpp>
#include <boost/container/new_allocator.hpp> //new_allocator
#include <boost/container/throw_exception.hpp>
#include <boost/container/options.hpp>
// container/detail
#include <boost/container/detail/advanced_insert_int.hpp>
#include <boost/container/detail/algorithm.hpp> //algo_equal(), algo_lexicographical_compare
#include <boost/container/detail/alloc_helpers.hpp>
#include <boost/container/detail/copy_move_algo.hpp>
#include <boost/container/detail/iterator.hpp>
#include <boost/move/detail/iterator_to_raw_pointer.hpp>
#include <boost/container/detail/iterators.hpp>
#include <boost/container/detail/min_max.hpp>
#include <boost/container/detail/mpl.hpp>
#include <boost/move/detail/to_raw_pointer.hpp>
#include <boost/container/detail/type_traits.hpp>
#include <boost/container/detail/math_functions.hpp>
// move
#include <boost/move/adl_move_swap.hpp>
#include <boost/move/algo/detail/merge.hpp>
#include <boost/move/iterator.hpp>
#include <boost/move/traits.hpp>
#include <boost/move/utility_core.hpp>
// move/detail
#if defined(BOOST_NO_CXX11_VARIADIC_TEMPLATES)
#include <boost/move/detail/fwd_macros.hpp>
#endif
#include <boost/move/detail/move_helpers.hpp>
// other
#include <boost/assert.hpp>
// std
#include <cstddef>

#if !defined(BOOST_NO_CXX11_HDR_INITIALIZER_LIST)
#include <initializer_list>
#endif

namespace boost {
namespace container {

#ifndef BOOST_CONTAINER_DOXYGEN_INVOKED

template <class T>
struct deque_value_traits
{
   typedef T value_type;
   BOOST_STATIC_CONSTEXPR bool trivial_dctr = dtl::is_trivially_destructible<value_type>::value;
   BOOST_STATIC_CONSTEXPR bool trivial_dctr_after_move = ::boost::has_trivial_destructor_after_move<value_type>::value;
};

template<class T, std::size_t BlockBytes, std::size_t BlockSize, class StoredSizeType>
struct deque_block_traits
{
   BOOST_CONTAINER_STATIC_ASSERT_MSG(!(BlockBytes && BlockSize), "BlockBytes and BlockSize can't be specified at the same time");
   //Calculate default block size
   BOOST_STATIC_CONSTEXPR std::size_t default_block_bytes = sizeof(void*)*128u;
   BOOST_STATIC_CONSTEXPR std::size_t default_block_start = (default_block_bytes - 1u)/sizeof(T) + 1u;
   //Round to the next power of two
   BOOST_STATIC_CONSTEXPR std::size_t default_block_size_upp_pow2  = dtl::upper_power_of_2_ct<std::size_t, default_block_start>::value;

   //Check minimal size
   BOOST_STATIC_CONSTEXPR std::size_t default_min_block_size_pow = 3u;
   BOOST_STATIC_CONSTEXPR std::size_t default_min_block_size = 8u;
   BOOST_STATIC_CONSTEXPR std::size_t default_block_size_initial  = default_block_size_upp_pow2 < default_min_block_size
                                                                  ? default_min_block_size
                                                                  : default_block_size_upp_pow2;

   //Limit by stored size max value
   BOOST_STATIC_CONSTEXPR std::size_t max_stored_size_block_size  = std::size_t(1u) << (sizeof(StoredSizeType)*CHAR_BIT - default_min_block_size_pow);
   BOOST_STATIC_CONSTEXPR std::size_t default_block_size  = default_block_size_initial > max_stored_size_block_size
                                                                                       ? max_stored_size_block_size
                                                                                       : default_block_size_initial;
   //Now select between the default or the specified by the user
   BOOST_STATIC_CONSTEXPR std::size_t value  = BlockSize  ? BlockSize
                                             : BlockBytes ? (BlockBytes-1u)/sizeof(T) + 1u : default_block_size
                                                                                   ;
   BOOST_CONTAINER_STATIC_ASSERT_MSG(value <= max_stored_size_block_size, "BlockSize or BlockBytes is too big for the stored_size_type");

};

// Class invariants:
//  For any nonsingular iterator i:
//    i.node is the address of an element in the map array.  The
//      contents of i.node is a pointer to the beginning of a node.
//    i.cur is a pointer in the range [i.first, i.last).  NOTE:
//      the implication of this is that i.cur is always a dereferenceable
//      pointer, even if i is a past-the-end iterator.
//  For every node other than start.node and finish.node, every element
//    in the node is an initialized object.  If start.node == finish.node,
//    then [start.cur, finish.cur) are initialized objects, and
//    the elements outside that range are uninitialized storage.  Otherwise,
//    [start.cur, start.last) and [finish.first, finish.cur) are initialized
//    objects, and [start.first, start.cur) and [finish.cur, finish.last)
//    are uninitialized storage.
//  [map, map + map_size) is a valid range.
//  [start.node, finish.node] is a valid range contained within
//    [map, map + map_size).
//  A pointer in the range [map, map + map_size) points to an allocated node
//    if and only if the pointer is in the range [start.node, finish.node].
template<class Pointer, bool IsConst, unsigned BlockBytes, unsigned BlockSize, class StoredSizeType>
class deque_iterator
{
   public:
   typedef std::random_access_iterator_tag                                          iterator_category;
   typedef typename boost::intrusive::pointer_traits<Pointer>::element_type         value_type;
   typedef typename boost::intrusive::pointer_traits<Pointer>::difference_type      difference_type;
   typedef typename boost::intrusive::pointer_traits<Pointer>::size_type            size_type;
   typedef typename dtl::if_c
      < IsConst
      , typename boost::intrusive::pointer_traits<Pointer>::template
                                 rebind_pointer<const value_type>::type
      , Pointer
      >::type                                                                       pointer;
   typedef typename dtl::if_c
      < IsConst
      , const value_type&
      , value_type&
      >::type                                                                       reference;

   BOOST_CONSTEXPR inline static size_type get_block_size() BOOST_NOEXCEPT_OR_NOTHROW
   {
      return deque_block_traits<value_type, BlockBytes, BlockSize, StoredSizeType>::value;
   }

   BOOST_CONSTEXPR inline static difference_type get_block_ssize() BOOST_NOEXCEPT_OR_NOTHROW
      { return difference_type((get_block_size())); }

   typedef Pointer                                                                  val_alloc_ptr;
   typedef typename boost::intrusive::pointer_traits<Pointer>::
      template rebind_pointer<Pointer>::type                                        index_pointer;

   class nat
   {
      public:
      inline Pointer get_cur()          const  {  return Pointer();  }
      inline index_pointer get_node()   const  {  return index_pointer();  }
   };

   typedef typename dtl::if_c< IsConst
                             , deque_iterator<Pointer, false, BlockBytes, BlockSize, StoredSizeType>
                             , nat>::type                                           nonconst_iterator_arg;

   typedef deque_iterator<Pointer, false, BlockBytes, BlockSize, StoredSizeType> nonconst_iterator;

   Pointer m_cur;
   index_pointer  m_node;

   public:

   BOOST_CONTAINER_ATTRIBUTE_NODISCARD inline Pointer get_cur()          const  {  return m_cur;  }
   BOOST_CONTAINER_ATTRIBUTE_NODISCARD inline index_pointer get_node()   const  {  return m_node;  }
   BOOST_CONTAINER_ATTRIBUTE_NODISCARD inline Pointer get_first()        const  {  return *m_node;  }
   BOOST_CONTAINER_ATTRIBUTE_NODISCARD inline Pointer get_last()         const  {  return *m_node + get_block_ssize(); }

   inline deque_iterator(val_alloc_ptr x, index_pointer y) BOOST_NOEXCEPT_OR_NOTHROW
      : m_cur(x), m_node(y)
   {}

   inline deque_iterator() BOOST_NOEXCEPT_OR_NOTHROW
      : m_cur(), m_node()  //Value initialization to achieve "null iterators" (N3644)
   {}

   inline deque_iterator(const deque_iterator& x) BOOST_NOEXCEPT_OR_NOTHROW
      : m_cur(x.get_cur()), m_node(x.get_node())
   {}

   inline deque_iterator(const nonconst_iterator_arg& x) BOOST_NOEXCEPT_OR_NOTHROW
      : m_cur(x.get_cur()), m_node(x.get_node())
   {}

   inline deque_iterator& operator=(const deque_iterator& x) BOOST_NOEXCEPT_OR_NOTHROW
   {  m_cur = x.get_cur(); m_node = x.get_node(); return *this; }

   inline deque_iterator& operator=(const nonconst_iterator_arg& x) BOOST_NOEXCEPT_OR_NOTHROW
   {  m_cur = x.get_cur(); m_node = x.get_node(); return *this; }

   inline nonconst_iterator unconst() const BOOST_NOEXCEPT_OR_NOTHROW
   {
      return nonconst_iterator(this->get_cur(), this->get_node());
   }

   inline reference operator*() const BOOST_NOEXCEPT_OR_NOTHROW
      { return *this->m_cur; }

   inline pointer operator->() const BOOST_NOEXCEPT_OR_NOTHROW
      { return this->m_cur; }

   BOOST_CONTAINER_ATTRIBUTE_NODISCARD difference_type operator-(const deque_iterator& x) const BOOST_NOEXCEPT_OR_NOTHROW
   {
      if(this->m_cur == x.m_cur){ //Includes when both are null
         return 0;
      }
      BOOST_CONSTEXPR_OR_CONST difference_type block_size = get_block_ssize();
      return block_size * (this->m_node - x.m_node) +
         (this->m_cur - this->get_first()) - (x.m_cur - x.get_first());
   }

   deque_iterator& operator++() BOOST_NOEXCEPT_OR_NOTHROW
   {
      BOOST_ASSERT(!!m_cur);
      ++this->m_cur;
      const Pointer last = *m_node + get_block_ssize();
      if (BOOST_UNLIKELY(this->m_cur == last)) {
         ++this->m_node;
         this->m_cur = *this->m_node;
      }
      return *this;
   }

   inline deque_iterator operator++(int) BOOST_NOEXCEPT_OR_NOTHROW
   {
      deque_iterator tmp(*this);
      ++*this;
      return tmp;
   }

//Some GCC versions issue bogus warnings about array bounds here
#if defined(BOOST_GCC) && (BOOST_GCC >= 40600)
#  pragma GCC diagnostic ignored "-Warray-bounds"
#endif

   deque_iterator& operator--() BOOST_NOEXCEPT_OR_NOTHROW
   {
      BOOST_ASSERT(!!m_cur);
      const Pointer first = *m_node;
      if (BOOST_UNLIKELY(this->m_cur == first)) {
         --this->m_node;
         this->m_cur = *this->m_node;
         this->m_cur += get_block_ssize() - 1;
      }
      else {
         --this->m_cur;
      }
      return *this;
   }

#if defined(BOOST_GCC) && (BOOST_GCC >= 40600)
#pragma GCC diagnostic pop
#endif

   inline deque_iterator operator--(int) BOOST_NOEXCEPT_OR_NOTHROW
   {
      deque_iterator tmp(*this);
      --*this;
      return tmp;
   }

   deque_iterator& operator+=(difference_type n) BOOST_NOEXCEPT_OR_NOTHROW
   {
      if (n){
         BOOST_CONSTEXPR_OR_CONST difference_type block_size = get_block_ssize();

         const difference_type offset = n + (this->m_cur - *this->m_node);

         if (offset >= 0 && offset < block_size)
            this->m_cur += difference_type(n);
         else {
            const difference_type node_offset =
               offset > 0 ? (offset / block_size)
                          : (-difference_type((-offset - 1) / block_size) - 1);
            this->m_node += node_offset;
            this->m_cur = *this->m_node + difference_type(offset - node_offset * block_size);
         }
      }
      return *this;
   }

   BOOST_CONTAINER_ATTRIBUTE_NODISCARD inline
      deque_iterator operator+(difference_type n) const BOOST_NOEXCEPT_OR_NOTHROW
      {  deque_iterator tmp(*this); return tmp += n;  }

   inline
      deque_iterator& operator-=(difference_type n) BOOST_NOEXCEPT_OR_NOTHROW
      { return *this += -n; }

   BOOST_CONTAINER_ATTRIBUTE_NODISCARD inline
      deque_iterator operator-(difference_type n) const BOOST_NOEXCEPT_OR_NOTHROW
      {  deque_iterator tmp(*this); return tmp -= n;  }

   BOOST_CONTAINER_ATTRIBUTE_NODISCARD inline
      reference operator[](difference_type n) const BOOST_NOEXCEPT_OR_NOTHROW
   {
      BOOST_ASSERT(!!m_cur);
      BOOST_CONSTEXPR_OR_CONST difference_type block_size = get_block_ssize();

      const difference_type offset = n + (this->m_cur - *this->m_node);

      if (offset >= 0 && offset < block_size)
         return this->m_cur[difference_type(n)];
      else {
         const difference_type node_offset = offset > 0
            ? (offset / block_size)
            : (-difference_type((-offset - 1) / block_size) - 1);
         return (this->m_node[node_offset]) [offset - node_offset * block_size];
      }
   }

   //Comparisons
   BOOST_CONTAINER_ATTRIBUTE_NODISCARD inline
      friend bool operator==(const deque_iterator& l, const deque_iterator& r) BOOST_NOEXCEPT_OR_NOTHROW
      { return l.m_cur == r.m_cur; }

   BOOST_CONTAINER_ATTRIBUTE_NODISCARD inline
      friend bool operator!=(const deque_iterator& l, const deque_iterator& r) BOOST_NOEXCEPT_OR_NOTHROW
      { return l.m_cur != r.m_cur; }

   BOOST_CONTAINER_ATTRIBUTE_NODISCARD inline
      friend bool operator<(const deque_iterator& l, const deque_iterator& r) BOOST_NOEXCEPT_OR_NOTHROW
      {  return (l.m_node == r.m_node) ? (l.m_cur < r.m_cur) : (l.m_node < r.m_node);  }

   BOOST_CONTAINER_ATTRIBUTE_NODISCARD inline
      friend bool operator>(const deque_iterator& l, const deque_iterator& r) BOOST_NOEXCEPT_OR_NOTHROW
      { return r < l; }

   BOOST_CONTAINER_ATTRIBUTE_NODISCARD inline
      friend bool operator<=(const deque_iterator& l, const deque_iterator& r) BOOST_NOEXCEPT_OR_NOTHROW
      { return !(r < l); }

   BOOST_CONTAINER_ATTRIBUTE_NODISCARD inline
      friend bool operator>=(const deque_iterator& l, const deque_iterator& r) BOOST_NOEXCEPT_OR_NOTHROW
      { return !(l < r); }

   BOOST_CONTAINER_ATTRIBUTE_NODISCARD inline
      friend deque_iterator operator+(difference_type n, deque_iterator x) BOOST_NOEXCEPT_OR_NOTHROW
      {  return x += n;  }

   inline void priv_set_node(index_pointer new_node) BOOST_NOEXCEPT_OR_NOTHROW
   {  this->m_node = new_node;  }
};


template<class Options, class AllocatorSizeType>
struct get_deque_opt
{
   typedef deque_opt< Options::block_bytes, Options::block_size
                    , typename default_if_void<typename Options::stored_size_type, AllocatorSizeType>::type
                    , Options::reservable> type;
};

template<class AllocatorSizeType>
struct get_deque_opt<void, AllocatorSizeType>
{
   typedef deque_opt<deque_null_opt::block_bytes, deque_null_opt::block_size, AllocatorSizeType, deque_null_opt::reservable> type;
};


// Deque base class.  It has two purposes.  First, its constructor
//  and destructor allocate (but don't initialize) storage.  This makes
//  exception safety easier.
template <class Allocator, class Options>
class deque_base
{
   BOOST_COPYABLE_AND_MOVABLE(deque_base)
   public:
   typedef allocator_traits<Allocator>                            val_alloc_traits_type;
   typedef typename val_alloc_traits_type::value_type             val_alloc_val;
   typedef typename val_alloc_traits_type::pointer                val_alloc_ptr;
   typedef typename val_alloc_traits_type::const_pointer          val_alloc_cptr;
   typedef typename val_alloc_traits_type::reference              val_alloc_ref;
   typedef typename val_alloc_traits_type::const_reference        val_alloc_cref;
   typedef typename val_alloc_traits_type::difference_type        val_alloc_diff;
   typedef typename val_alloc_traits_type::size_type              val_alloc_size;
   typedef typename val_alloc_traits_type::template
      portable_rebind_alloc<val_alloc_ptr>::type                  ptr_alloc_t;
   typedef allocator_traits<ptr_alloc_t>                          ptr_alloc_traits_type;
   typedef typename ptr_alloc_traits_type::value_type             ptr_alloc_val;
   typedef typename ptr_alloc_traits_type::pointer                ptr_alloc_ptr;
   typedef typename ptr_alloc_traits_type::const_pointer          ptr_alloc_cptr;
   typedef typename ptr_alloc_traits_type::reference              ptr_alloc_ref;
   typedef typename ptr_alloc_traits_type::const_reference        ptr_alloc_cref;
   typedef Allocator                                              allocator_type;
   typedef allocator_type                                         stored_allocator_type;
   typedef val_alloc_size                                         size_type;
   typedef val_alloc_diff                                         difference_type;

   private:
   typedef typename get_deque_opt<Options, val_alloc_size>::type  options_type;

   protected:
   typedef typename options_type::stored_size_type                stored_size_type;
   typedef deque_iterator< val_alloc_ptr, false
                         , options_type::block_bytes
                         , options_type::block_size
                         , stored_size_type>                      iterator;
   typedef deque_iterator< val_alloc_ptr, true
                         , options_type::block_bytes
                         , options_type::block_size
                         , stored_size_type>                      const_iterator;

   static const bool is_reservable = options_type::reservable;

   BOOST_CONSTEXPR inline static val_alloc_diff get_block_ssize() BOOST_NOEXCEPT_OR_NOTHROW
      { return val_alloc_diff((get_block_size())); }

   BOOST_CONSTEXPR inline static size_type get_block_size() BOOST_NOEXCEPT_OR_NOTHROW
      { return deque_block_traits<val_alloc_val, options_type::block_bytes, options_type::block_size, stored_size_type>::value; }

   typedef deque_value_traits<val_alloc_val>             traits_t;
   typedef ptr_alloc_t                                   map_allocator_type;

   inline val_alloc_ptr prot_allocate_node()
   {
      return this->alloc().allocate(get_block_size());
   }

   inline void prot_deallocate_node(val_alloc_ptr p) BOOST_NOEXCEPT_OR_NOTHROW
   {
      this->alloc().deallocate(p, get_block_size());
   }

   inline ptr_alloc_ptr prot_allocate_map(size_type n)
   {
      return this->ptr_alloc().allocate(n);
   }

   inline void prot_deallocate_map(ptr_alloc_ptr p, size_type n) BOOST_NOEXCEPT_OR_NOTHROW
   {
        this->ptr_alloc().deallocate(p, n);
   }

   inline deque_base(size_type num_elements, const allocator_type& a)
      :  members_(a)
   { this->prot_initialize_map_and_nodes(num_elements); }

   inline explicit deque_base(const allocator_type& a)
      :  members_(a)
   {}

   inline deque_base()
      :  members_()
   {}

   inline explicit deque_base(BOOST_RV_REF(deque_base) x)
      :  members_( boost::move(x.ptr_alloc())
                 , boost::move(x.alloc()) )
   {}

   ~deque_base()
   {
      if (this->members_.m_map) {
         this->prot_deallocate_all_nodes();
         this->prot_deallocate_map(this->members_.m_map, this->members_.m_map_size);
      }
   }

   private:
   deque_base(const deque_base&);

   protected:

   void swap_members(deque_base &x) BOOST_NOEXCEPT_OR_NOTHROW
   {
      this->members_.swap(x.members_);
   }

   void test_size_against_max(size_type n)
   {
      const size_type max_alloc = val_alloc_traits_type::max_size(this->alloc());
      const size_type max = max_alloc <= stored_size_type(-1) ? max_alloc : stored_size_type(-1);
      if (BOOST_UNLIKELY(max < n) )
         boost::container::throw_length_error("get_next_capacity, allocator's max size reached");
   }

   void test_size_against_n_nodes(size_type n_nodes)
   {
      test_size_against_max(size_type(size_type(n_nodes * get_block_size()) - 1u));
   }

   void prot_initialize_map_and_nodes(size_type num_elements)
   {
      BOOST_CONSTEXPR_OR_CONST size_type block_size = get_block_size();
      //Even a zero element initialized map+nodes needs at least 1 node (for sentinel finish position)
      size_type num_nodes = size_type(num_elements / block_size + 1u);

      //Allocate at least one extra slot on each end to avoid inmediate map reallocation on push/front insertions
      const size_type map_size = dtl::max_value(size_type(InitialMapSize), size_type(num_nodes + 2u));
      const size_type start_map_pos = size_type(map_size - num_nodes)/2u;

      //The end position must be representable in stored_size_type
      this->test_size_against_n_nodes(map_size);

      this->members_.m_map = this->prot_allocate_map(map_size);
      this->members_.m_map_size = static_cast<stored_size_type>(map_size);


      BOOST_CONTAINER_TRY {
         BOOST_IF_CONSTEXPR(is_reservable){   //reservable implies all slots are allocated with nodes
            this->prot_allocate_nodes(this->members_.m_map, map_size);
         }
         else{  //Otherwise only create the nodes that will hold values
            this->prot_allocate_nodes(this->members_.m_map + difference_type(start_map_pos), num_nodes);
         }
      }
      BOOST_CONTAINER_CATCH(...){
         this->prot_deallocate_map(this->members_.m_map, this->members_.m_map_size);
         this->members_.m_map = ptr_alloc_ptr();
         this->members_.m_map_size = 0;
         BOOST_CONTAINER_RETHROW
      }
      BOOST_CONTAINER_CATCH_END

      this->prot_set_start_finish_from_node(start_map_pos, num_elements);
   }

   static size_type priv_new_offset(bool add_at_front, size_type map_size, size_type active_nodes, size_type additional_nodes)
   {
      return size_type(size_type(map_size - active_nodes) / 2 + (add_at_front ? +additional_nodes : +0));
   }

   void prot_reallocate_map_and_nodes  //is_reservable == true
      (dtl::bool_<true>, const size_type new_elems, const bool add_at_front)
   {
      const ptr_alloc_ptr start_node = this->prot_start_node();
      const size_type old_active_nodes = size_type(this->prot_finish_node() + 1u - start_node);

      const size_type additional_nodes = size_type((new_elems - 1u)/get_block_size() + 1u);
      const size_type new_active_nodes = size_type(old_active_nodes + additional_nodes);

      ptr_alloc_ptr new_nstart;
      const size_type old_map_size = this->members_.m_map_size;
      const ptr_alloc_ptr old_map  = this->members_.m_map;

      //Check for 1.5 factor
      if (old_map_size/3u >= new_active_nodes/2u) {
         new_nstart = old_map + difference_type(priv_new_offset(add_at_front, old_map_size, new_active_nodes, additional_nodes));
         const ptr_alloc_ptr end_node = start_node + difference_type(old_active_nodes);
         if (new_nstart < start_node)
            boost::movelib::rotate_gcd(new_nstart, start_node, end_node);
         else
            boost::movelib::rotate_gcd(start_node, end_node, new_nstart + old_active_nodes);
      }
      else {
         //1.5 increase, but at least one spare slot on each end
         const size_type new_map_size = dtl::max_value(size_type(old_map_size + old_map_size/2u), size_type(new_active_nodes + 2u));

         //The end position must be representable in stored_size_type
         this->test_size_against_n_nodes(new_map_size);

         const ptr_alloc_ptr new_map = this->prot_allocate_map(new_map_size);
         const size_type new_active_off = priv_new_offset(add_at_front, new_map_size, new_active_nodes, additional_nodes);
         new_nstart = new_map + difference_type(new_active_off);

         const size_type new_nodes = size_type(new_map_size - old_map_size);

         BOOST_CONTAINER_TRY {
            this->prot_allocate_nodes(new_map, new_nodes);
         }
         BOOST_CONTAINER_CATCH(...){
            this->prot_deallocate_map(new_map, new_map_size);
            BOOST_CONTAINER_RETHROW
         }
         BOOST_CONTAINER_CATCH_END

         const ptr_alloc_ptr new_old_node_limit = new_map + difference_type(new_nodes);

         //Move old nodes from the old map, rotate to the right position and deallocate the old map
         boost::container::move_n(old_map, old_map_size, new_old_node_limit);

         const size_type old_active_off = size_type(start_node - old_map);
         const ptr_alloc_ptr active_start = new_old_node_limit + difference_type(old_active_off);
         const ptr_alloc_ptr active_end   = active_start       + difference_type(old_active_nodes);
         if (new_nstart < active_start)
            boost::movelib::rotate_gcd(new_nstart, active_start, active_end);
         else
            boost::movelib::rotate_gcd(active_start, active_end, new_nstart + difference_type(old_active_nodes));

         this->prot_deallocate_map(old_map, old_map_size);

         this->members_.m_map = new_map;
         this->members_.m_map_size = static_cast<stored_size_type>(new_map_size);
      }

      this->prot_start_update_node(new_nstart);
      this->prot_finish_update_node(new_nstart + difference_type(old_active_nodes - 1u));
   }

   void prot_reallocate_map_and_nodes  //is_reservable == false
      (dtl::bool_<false>, const size_type new_elems, const bool add_at_front)
   {
      const size_type additional_nodes = size_type((new_elems - 1u)/get_block_size() + 1u);

      const ptr_alloc_ptr start_node = this->prot_start_node();
      const ptr_alloc_ptr finish_node = this->prot_finish_node();

      size_type unused_slots = add_at_front ? size_type(start_node - this->members_.m_map)
                                            : size_type(this->members_.m_map_size - 1u - size_type(finish_node - this->members_.m_map));
      if (additional_nodes > unused_slots){
         const ptr_alloc_ptr next_finish_node = finish_node + 1u;
         const size_type old_active_nodes = size_type(next_finish_node - start_node);
         const size_type new_active_nodes = size_type(old_active_nodes + additional_nodes);

         ptr_alloc_ptr new_nstart;
         const size_type old_map_size = this->members_.m_map_size;
         const ptr_alloc_ptr old_map  = this->members_.m_map;

         if (old_map_size/2u >= new_active_nodes) {
            new_nstart = old_map + difference_type(priv_new_offset(add_at_front, old_map_size, new_active_nodes, additional_nodes));
            if (new_nstart < start_node)
               boost::container::move_n(start_node, old_active_nodes, new_nstart);
            else
               boost::container::move_backward_n(next_finish_node, old_active_nodes, new_nstart + difference_type(old_active_nodes));
         }
         else {
            //Doubling size, but at least one spare slot on each end
            const size_type new_map_size = dtl::max_value(size_type(old_map_size*2), size_type(new_active_nodes + 2u));

            //The end position must be representable in stored_size_type
            this->test_size_against_n_nodes(new_map_size);

            const ptr_alloc_ptr new_map = this->prot_allocate_map(new_map_size);

            new_nstart = new_map + difference_type(priv_new_offset(add_at_front, new_map_size, new_active_nodes, additional_nodes));
            boost::container::move_n(start_node, old_active_nodes, new_nstart);

            this->prot_deallocate_map(old_map, old_map_size);

            this->members_.m_map = new_map;
            this->members_.m_map_size = static_cast<stored_size_type>(new_map_size);
         }

         this->prot_start_update_node(new_nstart);
         this->prot_finish_update_node(new_nstart + difference_type(old_active_nodes - 1u));
      }
   
      if(add_at_front)
         this->prot_allocate_nodes(this->prot_start_node() - difference_type(additional_nodes), additional_nodes);
      else
         this->prot_allocate_nodes(this->prot_finish_node() + 1, additional_nodes);
   }

   void prot_allocate_nodes(ptr_alloc_ptr start, size_type n)
   {
      size_type i = 0;
      BOOST_CONTAINER_TRY {
         for (; i < n; ++i)
            start[difference_type(i)] = this->prot_allocate_node();
      }
      BOOST_CONTAINER_CATCH(...) {
         for (size_type j = 0; j < i; ++j)
            this->prot_deallocate_node(start[difference_type(j)]);
         BOOST_CONTAINER_RETHROW
      }
      BOOST_CONTAINER_CATCH_END
   }


   void prot_deallocate_nodes(ptr_alloc_ptr nstart, ptr_alloc_ptr nfinish) BOOST_NOEXCEPT_OR_NOTHROW
   {
      for (ptr_alloc_ptr n = nstart; n < nfinish; ++n){
         this->prot_deallocate_node(*n);
      }
   }

   void prot_deallocate_nodes_if_not_reservable(ptr_alloc_ptr nstart, ptr_alloc_ptr nfinish) BOOST_NOEXCEPT_OR_NOTHROW
   {
      (void)nstart; (void)nfinish;
      BOOST_IF_CONSTEXPR(!is_reservable)
         this->prot_deallocate_nodes(nstart, nfinish);
   }

   void prot_deallocate_node_if_not_reservable(val_alloc_ptr p) BOOST_NOEXCEPT_OR_NOTHROW
   {
      (void)p;
      BOOST_IF_CONSTEXPR(!is_reservable)
         this->prot_deallocate_node(p);
   }

   void prot_deallocate_all_nodes()
   {
      BOOST_IF_CONSTEXPR(is_reservable)
         this->prot_deallocate_nodes(this->members_.m_map, this->members_.m_map + difference_type(this->members_.m_map_size));
      else
         this->prot_deallocate_nodes(this->prot_start_node(), this->prot_finish_node() + 1);
   }


   void prot_clear_map() BOOST_NOEXCEPT_OR_NOTHROW
   {
      if (this->members_.m_map) {
         this->prot_deallocate_all_nodes();
         this->prot_deallocate_map(this->members_.m_map, this->members_.m_map_size);
         this->members_.m_map = ptr_alloc_ptr();
         this->members_.m_map_size = 0u;
         this->members_.m_start_off = 0u;
         this->members_.m_finish_off = 0u;
      }
   }

   enum { InitialMapSize = 4 };

   protected:
   struct members_holder
      :  public ptr_alloc_t
      ,  public allocator_type
   {
      friend class deque_base;
      members_holder()
         :  map_allocator_type(), allocator_type()
         ,  m_map(), m_map_size()
         ,  m_start_off(), m_finish_off()
      {}

      explicit members_holder(const allocator_type &a)
         :  map_allocator_type(a), allocator_type(a)
         ,  m_map(), m_map_size()
         ,  m_start_off(), m_finish_off()
      {}

      template<class ValAllocConvertible, class PtrAllocConvertible>
      members_holder(BOOST_FWD_REF(PtrAllocConvertible) pa, BOOST_FWD_REF(ValAllocConvertible) va)
         : map_allocator_type(boost::forward<PtrAllocConvertible>(pa))
         , allocator_type    (boost::forward<ValAllocConvertible>(va))
         , m_map(), m_map_size()
         , m_start_off(), m_finish_off()
      {}

      void swap(members_holder &x) BOOST_NOEXCEPT_OR_NOTHROW
      {
         ::boost::adl_move_swap(this->m_map, x.m_map);
         ::boost::adl_move_swap(this->m_map_size, x.m_map_size);
         ::boost::adl_move_swap(this->m_start_off, x.m_start_off);
         ::boost::adl_move_swap(this->m_finish_off, x.m_finish_off);
      }

      ptr_alloc_ptr     m_map;
      stored_size_type  m_map_size;
      private:
      stored_size_type  m_start_off;
      stored_size_type  m_finish_off;
   } members_;

   inline ptr_alloc_t &ptr_alloc() BOOST_NOEXCEPT_OR_NOTHROW
   {  return members_;  }

   inline const ptr_alloc_t &ptr_alloc() const BOOST_NOEXCEPT_OR_NOTHROW
   {  return members_;  }

   inline allocator_type &alloc() BOOST_NOEXCEPT_OR_NOTHROW
   {  return members_;  }

   inline const allocator_type &alloc() const BOOST_NOEXCEPT_OR_NOTHROW
   {  return members_;  }

   static BOOST_CONTAINER_FORCEINLINE val_alloc_ptr prot_node_last(ptr_alloc_ptr idx)
   {
      BOOST_CONSTEXPR_OR_CONST std::size_t block_size = deque_base::get_block_size();
      return *idx + difference_type(block_size - 1u);
   }

   BOOST_CONTAINER_FORCEINLINE size_type prot_front_free_capacity() const
   {
      BOOST_IF_CONSTEXPR(is_reservable){
         return static_cast<size_type>(this->members_.m_start_off);
      }
      else{
         BOOST_CONSTEXPR_OR_CONST std::size_t block_size = deque_base::get_block_size();
         return static_cast<size_type>(this->members_.m_start_off % block_size);
      }
   }

   BOOST_CONTAINER_FORCEINLINE size_type prot_back_free_capacity() const
   {
      BOOST_CONSTEXPR_OR_CONST std::size_t block_size = deque_base::get_block_size();

      BOOST_IF_CONSTEXPR(is_reservable){
         return static_cast<size_type>
            (size_type(this->members_.m_map_size*block_size) - size_type(this->members_.m_map != ptr_alloc_ptr()) - this->members_.m_finish_off);
      }
      else {
         //m_finish_off points to positions [0....block_size-1], and one position is always needed as the sentinel node resulting [block_size-1....0] capacity
         return static_cast<size_type>(this->members_.m_map ? (block_size - 1u) - (this->members_.m_finish_off % block_size) : 0u);
      }
   }

   //////////////////////////
   // it_to_off / off_to_it
   //////////////////////////

   BOOST_CONTAINER_FORCEINLINE stored_size_type prot_it_to_off(const_iterator it) const
   {
      const ptr_alloc_ptr n = it.get_node();
      BOOST_ASSERT(!this->members_.m_map == !n);  //Both should be null or both non-null
      if (n) {
         BOOST_CONSTEXPR_OR_CONST std::size_t block_size = deque_base::get_block_size();
         return static_cast<stored_size_type>(std::size_t(n - this->members_.m_map)*block_size + std::size_t(it.get_cur() - *n));
      }
      else{
         return 0;
      }
   }

   BOOST_CONTAINER_FORCEINLINE iterator prot_off_to_it(std::size_t off) const
   {
      BOOST_CONSTEXPR_OR_CONST std::size_t block_size  = deque_base::get_block_size();
      const ptr_alloc_ptr node = this->members_.m_map + difference_type(off/block_size);
      return iterator(node ? *node + difference_type(off%block_size) : val_alloc_ptr(), node);
   }

   stored_size_type prot_it_to_start_off(const_iterator it) const
   {
      const size_type off = this->prot_it_to_off(it);
      BOOST_ASSERT(off >= this->members_.m_start_off);
      return static_cast<stored_size_type>(off - this->members_.m_start_off);
   }

   /////////////
   // xxx_to_node
   /////////////

   BOOST_CONTAINER_FORCEINLINE ptr_alloc_ptr prot_off_to_node(std::size_t off) const
   {
      BOOST_CONSTEXPR_OR_CONST std::size_t block_size  = deque_base::get_block_size();
      return this->members_.m_map + difference_type(off/block_size);
   }

   BOOST_CONTAINER_FORCEINLINE ptr_alloc_ptr prot_start_node() const
   {
      return this->prot_off_to_node(this->members_.m_start_off);
   }

   BOOST_CONTAINER_FORCEINLINE ptr_alloc_ptr prot_finish_node() const
   {
      return this->prot_off_to_node(this->members_.m_finish_off);
   }

   //
   //    xxx_to_cur_unchecked versions, faster but need non-default constructed deque
   //
   BOOST_CONTAINER_FORCEINLINE val_alloc_ptr prot_off_to_cur_unchecked(std::size_t off) const
   {
      BOOST_ASSERT(!!this->members_.m_map);
      BOOST_CONSTEXPR_OR_CONST std::size_t block_size  = deque_base::get_block_size();
      const ptr_alloc_ptr node = this->members_.m_map + difference_type(off/block_size);
      return *node + difference_type(off%block_size);
   }

   BOOST_CONTAINER_FORCEINLINE val_alloc_ptr prot_start_cur_unchecked() const
   {
      return this->prot_off_to_cur_unchecked(this->members_.m_start_off);
   }

   BOOST_CONTAINER_FORCEINLINE val_alloc_ptr prot_finish_cur_unchecked() const
   {
      return this->prot_off_to_cur_unchecked(this->members_.m_finish_off);
   }

   BOOST_CONTAINER_FORCEINLINE val_alloc_ptr prot_last_cur_unchecked() const
   {
      BOOST_ASSERT(members_.m_start_off != members_.m_finish_off);
      return this->prot_off_to_cur_unchecked(this->members_.m_finish_off-1u);
   }

   //
   //    functions returning iterators to different positions
   //
   BOOST_CONTAINER_FORCEINLINE const_iterator prot_start() const
   {  return this->prot_off_to_it(members_.m_start_off);  }

   BOOST_CONTAINER_FORCEINLINE iterator prot_start()
   {  return this->prot_off_to_it(members_.m_start_off);  }

   BOOST_CONTAINER_FORCEINLINE const_iterator prot_finish() const
   {  return this->prot_off_to_it(members_.m_finish_off);  }

   BOOST_CONTAINER_FORCEINLINE iterator prot_finish()
   {  return this->prot_off_to_it(members_.m_finish_off);  }

   BOOST_CONTAINER_FORCEINLINE const_iterator prot_nth(size_type n) const
   {  return this->prot_off_to_it(size_type(members_.m_start_off + n));  }

   BOOST_CONTAINER_FORCEINLINE iterator prot_nth(size_type n)
   {  return this->prot_off_to_it(size_type(members_.m_start_off + n));  }

   BOOST_CONTAINER_FORCEINLINE iterator prot_back_it()
   {
      BOOST_ASSERT(members_.m_start_off != members_.m_finish_off);
      return this->prot_off_to_it(size_type(members_.m_finish_off - 1u));
   }

   //
   //  size/empty
   //
   BOOST_CONTAINER_FORCEINLINE size_type prot_size() const
   {  return size_type(this->members_.m_finish_off - this->members_.m_start_off);  }

   BOOST_CONTAINER_FORCEINLINE bool prot_empty() const
   {  return this->members_.m_finish_off == this->members_.m_start_off;  }

   //
   //  Functions to move start/finish indexes
   //
   BOOST_CONTAINER_FORCEINLINE void prot_inc_start()
   {  ++this->members_.m_start_off;  }

   BOOST_CONTAINER_FORCEINLINE void prot_dec_start()
   {  --this->members_.m_start_off;  }

   BOOST_CONTAINER_FORCEINLINE void prot_inc_finish()
   {  ++this->members_.m_finish_off;  }

   BOOST_CONTAINER_FORCEINLINE void prot_dec_finish()
   {  --this->members_.m_finish_off;  }

   BOOST_CONTAINER_FORCEINLINE void prot_dec_finish(std::size_t n)
   {  this->members_.m_finish_off = static_cast<stored_size_type>(this->members_.m_finish_off - n);  }

   BOOST_CONTAINER_FORCEINLINE void prot_inc_finish(std::size_t n)
   {  this->members_.m_finish_off = static_cast<stored_size_type>(this->members_.m_finish_off + n);  }

   BOOST_CONTAINER_FORCEINLINE void prot_dec_start(std::size_t n)
   {  this->members_.m_start_off = static_cast<stored_size_type>(this->members_.m_start_off - n);  }

   BOOST_CONTAINER_FORCEINLINE void prot_inc_start(std::size_t n)
   {  this->members_.m_start_off = static_cast<stored_size_type>(this->members_.m_start_off + n);  }

   //
   //  Functions to obtain indexes from nodes
   //
   BOOST_CONTAINER_FORCEINLINE stored_size_type prot_non_null_node_to_off(ptr_alloc_ptr n) const
   {
      BOOST_CONSTEXPR_OR_CONST std::size_t block_size  = deque_base::get_block_size();
      return static_cast<stored_size_type>(std::size_t(n - this->members_.m_map)*block_size);
   }

   void prot_set_start_finish_from_node(size_type node_idx, size_type n_elements)
   {
      BOOST_CONSTEXPR_OR_CONST std::size_t block_size  = deque_base::get_block_size();
      this->members_.m_start_off  = static_cast<stored_size_type>(node_idx*block_size);
      this->members_.m_finish_off = static_cast<stored_size_type>(this->members_.m_start_off + n_elements);
   }

   inline void prot_start_update_node(ptr_alloc_ptr new_start)
   {
      //iG: to-do: optimizable avoiding some division/remainder
      std::size_t new_block_off = prot_non_null_node_to_off(new_start);
      this->members_.m_start_off = static_cast<stored_size_type>(new_block_off + (this->members_.m_start_off % get_block_size()));
   }

   inline void prot_finish_update_node(ptr_alloc_ptr new_finish)
   {
      //iG: to-do: optimizable avoiding some division/remainder
      std::size_t new_block_off = prot_non_null_node_to_off(new_finish);
      this->members_.m_finish_off = static_cast<stored_size_type>(new_block_off + (this->members_.m_finish_off % get_block_size()));
   }

   inline void prot_reset_finish_to_start()
   {  this->members_.m_finish_off = this->members_.m_start_off;  }

   inline void prot_reset_start_to_finish()
   {  this->members_.m_start_off = this->members_.m_finish_off;  }

   inline val_alloc_val *prot_push_back_simple_pos() const
   {
      BOOST_CONSTEXPR_OR_CONST std::size_t block_size  = deque_base::get_block_size();
      const std::size_t last_in_block = block_size - 1u;
      const ptr_alloc_val *const map = boost::movelib::to_raw_pointer(this->members_.m_map);
      if(BOOST_LIKELY(map != 0)) {
         const std::size_t off = this->members_.m_finish_off;
         const std::size_t rem = off % block_size;
         if(BOOST_LIKELY(rem != last_in_block)){
            return boost::movelib::to_raw_pointer(map[difference_type(off/block_size)]) + difference_type(rem);
         }
      }
      return 0;
   }

   inline val_alloc_val *prot_push_front_simple_pos() const
   {
      BOOST_CONSTEXPR_OR_CONST std::size_t block_size  = deque_base::get_block_size();
      //No need to check !m_map, as m_start_off is zero in that case
      const std::size_t off = this->members_.m_start_off;
      const std::size_t rem = off % block_size;
      if(BOOST_LIKELY(rem != 0u)){
         return boost::movelib::to_raw_pointer(this->members_.m_map[difference_type(off/block_size)]) + difference_type(rem-1u);
      }
      return 0;
   }

   BOOST_CONTAINER_FORCEINLINE bool prot_pop_back_simple_available() const
   {
      return (this->members_.m_finish_off % get_block_size()) != 0u;
   }

   BOOST_CONTAINER_FORCEINLINE bool prot_pop_front_simple_available() const
   {
      return size_type(this->members_.m_start_off % get_block_size()) != size_type(get_block_size() - 1u);
   }

};
#endif   //#ifndef BOOST_CONTAINER_DOXYGEN_INVOKED

#ifdef BOOST_CONTAINER_DOXYGEN_INVOKED
//! A double-ended queue is a sequence that supports random access to elements, constant time insertion
//! and removal of elements at the end of the sequence, and linear time insertion and removal of elements in the middle.
//!
//! \tparam T The type of object that is stored in the deque
//! \tparam A The allocator used for all internal memory management, use void
//!   for the default allocator
//! \tparam Options A type produced from \c boost::container::deque_options.
template <class T, class Allocator = void, class Options = void>
#else
template <class T, class Allocator, class Options>
#endif
class deque : protected deque_base<typename real_allocator<T, Allocator>::type, Options>
{
   #ifndef BOOST_CONTAINER_DOXYGEN_INVOKED
   private:
   typedef deque_base<typename real_allocator<T, Allocator>::type, Options> Base;
   #endif   //#ifndef BOOST_CONTAINER_DOXYGEN_INVOKED
   typedef typename real_allocator<T, Allocator>::type ValAllocator;
   typedef constant_iterator<T> c_it;

   public:

   //////////////////////////////////////////////
   //
   //                    types
   //
   //////////////////////////////////////////////

   typedef T                                                                           value_type;
   typedef ValAllocator                                                                allocator_type;
   typedef typename ::boost::container::allocator_traits<ValAllocator>::pointer           pointer;
   typedef typename ::boost::container::allocator_traits<ValAllocator>::const_pointer     const_pointer;
   typedef typename ::boost::container::allocator_traits<ValAllocator>::reference         reference;
   typedef typename ::boost::container::allocator_traits<ValAllocator>::const_reference   const_reference;
   typedef typename ::boost::container::allocator_traits<ValAllocator>::size_type         size_type;
   typedef typename ::boost::container::allocator_traits<ValAllocator>::difference_type   difference_type;
   typedef BOOST_CONTAINER_IMPDEF(allocator_type)                                      stored_allocator_type;
   typedef BOOST_CONTAINER_IMPDEF(typename Base::iterator)                             iterator;
   typedef BOOST_CONTAINER_IMPDEF(typename Base::const_iterator)                       const_iterator;
   typedef BOOST_CONTAINER_IMPDEF(boost::container::reverse_iterator<iterator>)        reverse_iterator;
   typedef BOOST_CONTAINER_IMPDEF(boost::container::reverse_iterator<const_iterator>)  const_reverse_iterator;

   #ifndef BOOST_CONTAINER_DOXYGEN_INVOKED

   private:                      // Internal typedefs

   //`allocator_type::value_type` must match container's `value type`. If this
   //assertion fails, please review your allocator definition. 
   BOOST_CONTAINER_STATIC_ASSERT((dtl::is_same<value_type, typename allocator_traits<ValAllocator>::value_type>::value));

   BOOST_COPYABLE_AND_MOVABLE(deque)
   typedef typename Base::ptr_alloc_ptr                     index_pointer;
   typedef allocator_traits<ValAllocator>                   allocator_traits_type;
   typedef typename Base::stored_size_type                  stored_size_type;

   #endif   //#ifndef BOOST_CONTAINER_DOXYGEN_INVOKED

   using Base::get_block_ssize;

   public:

   using Base::get_block_size;

   static const std::size_t is_reservable = Base::is_reservable;

   //////////////////////////////////////////////
   //
   //          construct/copy/destroy
   //
   //////////////////////////////////////////////

   //! <b>Effects</b>: Default constructors a deque.
   //!
   //! <b>Throws</b>: If allocator_type's default constructor throws.
   //!
   //! <b>Complexity</b>: Constant.
   inline deque()
      BOOST_NOEXCEPT_IF(dtl::is_nothrow_default_constructible<ValAllocator>::value)
      : Base()
   {}

   //! <b>Effects</b>: Constructs a deque taking the allocator as parameter.
   //!
   //! <b>Throws</b>: Nothing
   //!
   //! <b>Complexity</b>: Constant.
   inline explicit deque(const allocator_type& a) BOOST_NOEXCEPT_OR_NOTHROW
      : Base(a)
   {}

   //! <b>Effects</b>: Constructs a deque
   //!   and inserts n value initialized values.
   //!
   //! <b>Throws</b>: If allocator_type's default constructor
   //!   throws or T's value initialization throws.
   //!
   //! <b>Complexity</b>: Linear to n.
   inline explicit deque(size_type n)
      : Base(n, allocator_type())
   {
      dtl::insert_value_initialized_n_proxy<ValAllocator> proxy;
      this->priv_segmented_proxy_uninitialized_copy_n_and_update(this->begin(), n, proxy);
      //deque_base will deallocate in case of exception...
   }

   //! <b>Effects</b>: Constructs a deque
   //!   and inserts n default initialized values.
   //!
   //! <b>Throws</b>: If allocator_type's default constructor
   //!   throws or T's default initialization or copy constructor throws.
   //!
   //! <b>Complexity</b>: Linear to n.
   //!
   //! <b>Note</b>: Non-standard extension
   inline deque(size_type n, default_init_t)
      : Base(n, allocator_type())
   {
      dtl::insert_default_initialized_n_proxy<ValAllocator> proxy;
      this->priv_segmented_proxy_uninitialized_copy_n_and_update(this->begin(), n, proxy);
      //deque_base will deallocate in case of exception...
   }

   //! <b>Effects</b>: Constructs a deque that will use a copy of allocator a
   //!   and inserts n value initialized values.
   //!
   //! <b>Throws</b>: If allocator_type's default constructor
   //!   throws or T's value initialization throws.
   //!
   //! <b>Complexity</b>: Linear to n.
   inline explicit deque(size_type n, const allocator_type &a)
      : Base(n, a)
   {
      dtl::insert_value_initialized_n_proxy<ValAllocator> proxy;
      this->priv_segmented_proxy_uninitialized_copy_n_and_update(this->begin(), n, proxy);
      //deque_base will deallocate in case of exception...
   }

   //! <b>Effects</b>: Constructs a deque that will use a copy of allocator a
   //!   and inserts n default initialized values.
   //!
   //! <b>Throws</b>: If allocator_type's default constructor
   //!   throws or T's default initialization or copy constructor throws.
   //!
   //! <b>Complexity</b>: Linear to n.
   //!
   //! <b>Note</b>: Non-standard extension
   inline deque(size_type n, default_init_t, const allocator_type &a)
      : Base(n, a)
   {
      dtl::insert_default_initialized_n_proxy<ValAllocator> proxy;
      this->priv_segmented_proxy_uninitialized_copy_n_and_update(this->begin(), n, proxy);
      //deque_base will deallocate in case of exception...
   }

   //! <b>Effects</b>: Constructs a deque that will use a copy of allocator a
   //!   and inserts n copies of value.
   //!
   //! <b>Throws</b>: If allocator_type's default constructor
   //!   throws or T's copy constructor throws.
   //!
   //! <b>Complexity</b>: Linear to n.
   inline deque(size_type n, const value_type& value)
      : Base(n, allocator_type())
   { this->priv_fill_initialize(n, value); }

   //! <b>Effects</b>: Constructs a deque that will use a copy of allocator a
   //!   and inserts n copies of value.
   //!
   //! <b>Throws</b>: If allocator_type's default constructor
   //!   throws or T's copy constructor throws.
   //!
   //! <b>Complexity</b>: Linear to n.
   inline deque(size_type n, const value_type& value, const allocator_type& a)
      : Base(n, a)
   { this->priv_fill_initialize(n, value); }

   //! <b>Effects</b>: Constructs a deque that will use a copy of allocator a
   //!   and inserts a copy of the range [first, last) in the deque.
   //!
   //! <b>Throws</b>: If allocator_type's default constructor
   //!   throws or T's constructor taking a dereferenced InIt throws.
   //!
   //! <b>Complexity</b>: Linear to the range [first, last).
   template <class InIt>
   inline deque(InIt first, InIt last
      #if !defined(BOOST_CONTAINER_DOXYGEN_INVOKED)
      , typename dtl::disable_if_convertible
         <InIt, size_type>::type * = 0
      #endif
      )
      : Base(allocator_type())
   {
      this->priv_range_initialize(first, last);
   }

   //! <b>Effects</b>: Constructs a deque that will use a copy of allocator a
   //!   and inserts a copy of the range [first, last) in the deque.
   //!
   //! <b>Throws</b>: If allocator_type's default constructor
   //!   throws or T's constructor taking a dereferenced InIt throws.
   //!
   //! <b>Complexity</b>: Linear to the range [first, last).
   template <class InIt>
   inline deque(InIt first, InIt last, const allocator_type& a
      #if !defined(BOOST_CONTAINER_DOXYGEN_INVOKED)
      , typename dtl::disable_if_convertible
         <InIt, size_type>::type * = 0
      #endif
      )
      : Base(a)
   {
      this->priv_range_initialize(first, last);
   }

#if !defined(BOOST_NO_CXX11_HDR_INITIALIZER_LIST)
   //! <b>Effects</b>: Constructs a deque that will use a copy of allocator a
   //!   and inserts a copy of the range [il.begin(), il.end()) in the deque.
   //!
   //! <b>Throws</b>: If allocator_type's default constructor
   //!   throws or T's constructor taking a dereferenced std::initializer_list iterator throws.
   //!
   //! <b>Complexity</b>: Linear to the range [il.begin(), il.end()).
   inline deque(std::initializer_list<value_type> il, const allocator_type& a = allocator_type())
      : Base(a)
   {
      this->priv_range_initialize(il.begin(), il.end());
   }
#endif

   //! <b>Effects</b>: Copy constructs a deque.
   //!
   //! <b>Postcondition</b>: x == *this.
   //!
   //! <b>Complexity</b>: Linear to the elements x contains.
   inline deque(const deque& x)
      :  Base(allocator_traits_type::select_on_container_copy_construction(x.alloc()))
   {
      if(x.size()){
         this->prot_initialize_map_and_nodes(x.size());
         this->priv_segmented_uninitialized_copy_alloc_n(x.begin(), x.size(), this->begin());
      }
   }

   //! <b>Effects</b>: Move constructor. Moves x's resources to *this.
   //!
   //! <b>Throws</b>: If allocator_type's copy constructor throws.
   //!
   //! <b>Complexity</b>: Constant.
   inline deque(BOOST_RV_REF(deque) x) BOOST_NOEXCEPT_OR_NOTHROW
      :  Base(BOOST_MOVE_BASE(Base, x))
   {  this->swap_members(x);   }

   //! <b>Effects</b>: Copy constructs a vector using the specified allocator.
   //!
   //! <b>Postcondition</b>: x == *this.
   //!
   //! <b>Throws</b>: If allocation
   //!   throws or T's copy constructor throws.
   //!
   //! <b>Complexity</b>: Linear to the elements x contains.
   deque(const deque& x, const allocator_type &a)
      :  Base(a)
   {
      if(x.size()){
         this->prot_initialize_map_and_nodes(x.size());
         this->priv_segmented_uninitialized_copy_alloc_n(x.begin(), x.size(), this->begin());
      }
   }

   //! <b>Effects</b>: Move constructor using the specified allocator.
   //!                 Moves x's resources to *this if a == allocator_type().
   //!                 Otherwise copies values from x to *this.
   //!
   //! <b>Throws</b>: If allocation or T's copy constructor throws.
   //!
   //! <b>Complexity</b>: Constant if a == x.get_allocator(), linear otherwise.
   deque(BOOST_RV_REF(deque) x, const allocator_type &a)
      :  Base(a)
   {
      if(x.alloc() == a){
         this->swap_members(x);
      }
      else{
         if(x.size()){
            this->prot_initialize_map_and_nodes(x.size());
            this->priv_segmented_uninitialized_move_alloc_n(x.begin(), x.size(), this->begin());
         }
      }
   }

   //! <b>Effects</b>: Destroys the deque. All stored values are destroyed
   //!   and used memory is deallocated.
   //!
   //! <b>Throws</b>: Nothing.
   //!
   //! <b>Complexity</b>: Linear to the number of elements.
   inline ~deque() BOOST_NOEXCEPT_OR_NOTHROW
   {
      this->prot_destroy_range(this->prot_start(), this->prot_finish());
   }

   //! <b>Effects</b>: Makes *this contain the same elements as x.
   //!
   //! <b>Postcondition</b>: this->size() == x.size(). *this contains a copy
   //! of each of x's elements.
   //!
   //! <b>Throws</b>: If memory allocation throws or T's copy constructor throws.
   //!
   //! <b>Complexity</b>: Linear to the number of elements in x.
   deque& operator= (BOOST_COPY_ASSIGN_REF(deque) x)
   {
      if (BOOST_LIKELY(&x != this)){
         allocator_type &this_alloc     = this->alloc();
         const allocator_type &x_alloc  = x.alloc();
         dtl::bool_<allocator_traits_type::
            propagate_on_container_copy_assignment::value> flag;
         if(flag && this_alloc != x_alloc){
            this->clear();
            this->shrink_to_fit();
         }
         dtl::assign_alloc(this->alloc(), x.alloc(), flag);
         dtl::assign_alloc(this->ptr_alloc(), x.ptr_alloc(), flag);
         this->assign(x.cbegin(), x.cend());
      }
      return *this;
   }

   //! <b>Effects</b>: Move assignment. All x's values are transferred to *this.
   //!
   //! <b>Throws</b>: If allocator_traits_type::propagate_on_container_move_assignment
   //!   is false and (allocation throws or value_type's move constructor throws)
   //!
   //! <b>Complexity</b>: Constant if allocator_traits_type::
   //!   propagate_on_container_move_assignment is true or
   //!   this->get>allocator() == x.get_allocator(). Linear otherwise.
   deque& operator= (BOOST_RV_REF(deque) x)
      BOOST_NOEXCEPT_IF(allocator_traits_type::propagate_on_container_move_assignment::value
                                  || allocator_traits_type::is_always_equal::value)
   {
      if (BOOST_LIKELY(this != &x)) {
         //We know resources can be transferred at comiple time if both allocators are
         //always equal or the allocator is going to be propagated
         const bool can_steal_resources_alloc
            =  allocator_traits_type::propagate_on_container_move_assignment::value
            || allocator_traits_type::is_always_equal::value;
         dtl::bool_<can_steal_resources_alloc> flag;
         this->priv_move_assign(boost::move(x), flag);
      }
      return *this;
   }

#if !defined(BOOST_NO_CXX11_HDR_INITIALIZER_LIST)
   //! <b>Effects</b>: Makes *this contain the same elements as il.
   //!
   //! <b>Postcondition</b>: this->size() == il.size(). *this contains a copy
   //! of each of x's elements.
   //!
   //! <b>Throws</b>: If memory allocation throws or T's copy constructor throws.
   //!
   //! <b>Complexity</b>: Linear to the number of elements in il.
   inline deque& operator=(std::initializer_list<value_type> il)
   {
      this->assign(il.begin(), il.end());
      return *this;
   }
#endif

   //! <b>Effects</b>: Assigns the n copies of val to *this.
   //!
   //! <b>Throws</b>: If memory allocation throws or T's copy constructor throws.
   //!
   //! <b>Complexity</b>: Linear to n.
   inline void assign(size_type n, const T& val)
   {
      dtl::insert_n_copies_proxy<ValAllocator> proxy(val);
      this->priv_assign(n, proxy);
   }

   //! <b>Effects</b>: Assigns the the range [first, last) to *this.
   //!
   //! <b>Throws</b>: If memory allocation throws or
   //!   T's constructor from dereferencing InIt throws.
   //!
   //! <b>Complexity</b>: Linear to n.
   template <class InIt>
   void assign(InIt first, InIt last
      #if !defined(BOOST_CONTAINER_DOXYGEN_INVOKED)
      , typename dtl::disable_if_or
         < void
         , dtl::is_convertible<InIt, size_type>
         , dtl::is_not_input_iterator<InIt>
         >::type * = 0
      #endif
      )
   {
      iterator cur = this->begin();
      for ( ; first != last && cur != end(); ++cur, ++first){
         *cur = *first;
      }
      if (first == last){
         this->erase(cur, this->cend());
      }
      else{
         this->insert(this->cend(), first, last);
      }
   }

   #if !defined(BOOST_CONTAINER_DOXYGEN_INVOKED)
   template <class FwdIt>
   void assign(FwdIt first, FwdIt last
      , typename dtl::disable_if_or
         < void
         , dtl::is_convertible<FwdIt, size_type>
         , dtl::is_input_iterator<FwdIt>
         >::type * = 0
      )
   {
      typedef typename iter_size<FwdIt>::type it_size_type;
      const it_size_type sz = boost::container::iterator_udistance(first, last);
      if (BOOST_UNLIKELY(sz > size_type(-1))){
         boost::container::throw_length_error("vector::insert, FwdIt's max length reached");
      }
      const size_type n = static_cast<size_type>(sz);
      dtl::insert_range_proxy<ValAllocator, FwdIt> proxy(first);
      this->priv_assign(n, proxy);
   }
   #endif

#if !defined(BOOST_NO_CXX11_HDR_INITIALIZER_LIST)
   //! <b>Effects</b>: Assigns the the range [il.begin(), il.end()) to *this.
   //!
   //! <b>Throws</b>: If memory allocation throws or
   //!   T's constructor from dereferencing std::initializer_list iterator throws.
   //!
   //! <b>Complexity</b>: Linear to il.size().
   inline void assign(std::initializer_list<value_type> il)
   {   this->assign(il.begin(), il.end());   }
#endif

   //! <b>Effects</b>: Returns a copy of the internal allocator.
   //!
   //! <b>Throws</b>: If allocator's copy constructor throws.
   //!
   //! <b>Complexity</b>: Constant.
   BOOST_CONTAINER_ATTRIBUTE_NODISCARD inline
      allocator_type get_allocator() const BOOST_NOEXCEPT_OR_NOTHROW
   { return Base::alloc(); }

   //! <b>Effects</b>: Returns a reference to the internal allocator.
   //!
   //! <b>Throws</b>: Nothing
   //!
   //! <b>Complexity</b>: Constant.
   //!
   //! <b>Note</b>: Non-standard extension.
   BOOST_CONTAINER_ATTRIBUTE_NODISCARD inline
      const stored_allocator_type &get_stored_allocator() const BOOST_NOEXCEPT_OR_NOTHROW
   {  return Base::alloc(); }

   //////////////////////////////////////////////
   //
   //                iterators
   //
   //////////////////////////////////////////////

   //! <b>Effects</b>: Returns a reference to the internal allocator.
   //!
   //! <b>Throws</b>: Nothing
   //!
   //! <b>Complexity</b>: Constant.
   //!
   //! <b>Note</b>: Non-standard extension.
   BOOST_CONTAINER_ATTRIBUTE_NODISCARD inline
      stored_allocator_type &get_stored_allocator() BOOST_NOEXCEPT_OR_NOTHROW
   {  return Base::alloc(); }

   //! <b>Effects</b>: Returns an iterator to the first element contained in the deque.
   //!
   //! <b>Throws</b>: Nothing.
   //!
   //! <b>Complexity</b>: Constant.
   BOOST_CONTAINER_ATTRIBUTE_NODISCARD inline
      iterator begin() BOOST_NOEXCEPT_OR_NOTHROW
      { return this->prot_start(); }

   //! <b>Effects</b>: Returns a const_iterator to the first element contained in the deque.
   //!
   //! <b>Throws</b>: Nothing.
   //!
   //! <b>Complexity</b>: Constant.
   BOOST_CONTAINER_ATTRIBUTE_NODISCARD inline
      const_iterator begin() const BOOST_NOEXCEPT_OR_NOTHROW
      { return this->prot_start(); }

   //! <b>Effects</b>: Returns an iterator to the end of the deque.
   //!
   //! <b>Throws</b>: Nothing.
   //!
   //! <b>Complexity</b>: Constant.
   BOOST_CONTAINER_ATTRIBUTE_NODISCARD inline
      iterator end() BOOST_NOEXCEPT_OR_NOTHROW
      { return this->prot_finish(); }

   //! <b>Effects</b>: Returns a const_iterator to the end of the deque.
   //!
   //! <b>Throws</b>: Nothing.
   //!
   //! <b>Complexity</b>: Constant.
   BOOST_CONTAINER_ATTRIBUTE_NODISCARD inline
      const_iterator end() const BOOST_NOEXCEPT_OR_NOTHROW
      { return this->prot_finish(); }

   //! <b>Effects</b>: Returns a reverse_iterator pointing to the beginning
   //! of the reversed deque.
   //!
   //! <b>Throws</b>: Nothing.
   //!
   //! <b>Complexity</b>: Constant.
   BOOST_CONTAINER_ATTRIBUTE_NODISCARD inline
      reverse_iterator rbegin() BOOST_NOEXCEPT_OR_NOTHROW
      { return reverse_iterator(this->prot_finish()); }

   //! <b>Effects</b>: Returns a const_reverse_iterator pointing to the beginning
   //! of the reversed deque.
   //!
   //! <b>Throws</b>: Nothing.
   //!
   //! <b>Complexity</b>: Constant.
   BOOST_CONTAINER_ATTRIBUTE_NODISCARD inline
      const_reverse_iterator rbegin() const BOOST_NOEXCEPT_OR_NOTHROW
      { return const_reverse_iterator(this->prot_finish()); }

   //! <b>Effects</b>: Returns a reverse_iterator pointing to the end
   //! of the reversed deque.
   //!
   //! <b>Throws</b>: Nothing.
   //!
   //! <b>Complexity</b>: Constant.
   BOOST_CONTAINER_ATTRIBUTE_NODISCARD inline
      reverse_iterator rend() BOOST_NOEXCEPT_OR_NOTHROW
      { return reverse_iterator(this->prot_start()); }

   //! <b>Effects</b>: Returns a const_reverse_iterator pointing to the end
   //! of the reversed deque.
   //!
   //! <b>Throws</b>: Nothing.
   //!
   //! <b>Complexity</b>: Constant.
   BOOST_CONTAINER_ATTRIBUTE_NODISCARD inline
      const_reverse_iterator rend() const BOOST_NOEXCEPT_OR_NOTHROW
      { return const_reverse_iterator(this->prot_start()); }

   //! <b>Effects</b>: Returns a const_iterator to the first element contained in the deque.
   //!
   //! <b>Throws</b>: Nothing.
   //!
   //! <b>Complexity</b>: Constant.
   BOOST_CONTAINER_ATTRIBUTE_NODISCARD inline
      const_iterator cbegin() const BOOST_NOEXCEPT_OR_NOTHROW
      { return this->prot_start(); }

   //! <b>Effects</b>: Returns a const_iterator to the end of the deque.
   //!
   //! <b>Throws</b>: Nothing.
   //!
   //! <b>Complexity</b>: Constant.
   BOOST_CONTAINER_ATTRIBUTE_NODISCARD inline
      const_iterator cend() const BOOST_NOEXCEPT_OR_NOTHROW
      { return this->prot_finish(); }

   //! <b>Effects</b>: Returns a const_reverse_iterator pointing to the beginning
   //! of the reversed deque.
   //!
   //! <b>Throws</b>: Nothing.
   //!
   //! <b>Complexity</b>: Constant.
   BOOST_CONTAINER_ATTRIBUTE_NODISCARD inline
      const_reverse_iterator crbegin() const BOOST_NOEXCEPT_OR_NOTHROW
      { return const_reverse_iterator(this->prot_finish()); }

   //! <b>Effects</b>: Returns a const_reverse_iterator pointing to the end
   //! of the reversed deque.
   //!
   //! <b>Throws</b>: Nothing.
   //!
   //! <b>Complexity</b>: Constant.
   BOOST_CONTAINER_ATTRIBUTE_NODISCARD inline
      const_reverse_iterator crend() const BOOST_NOEXCEPT_OR_NOTHROW
      { return const_reverse_iterator(this->prot_start()); }

   //////////////////////////////////////////////
   //
   //                capacity
   //
   //////////////////////////////////////////////

   //! <b>Effects</b>: Returns true if the deque contains no elements.
   //!
   //! <b>Throws</b>: Nothing.
   //!
   //! <b>Complexity</b>: Constant.
   BOOST_CONTAINER_ATTRIBUTE_NODISCARD inline
      bool empty() const BOOST_NOEXCEPT_OR_NOTHROW
   { return this->prot_empty(); }

   //! <b>Effects</b>: Returns the number of the elements contained in the deque.
   //!
   //! <b>Throws</b>: Nothing.
   //!
   //! <b>Complexity</b>: Constant.
   BOOST_CONTAINER_ATTRIBUTE_NODISCARD inline
      size_type size() const BOOST_NOEXCEPT_OR_NOTHROW
      { return this->prot_size(); }

   //! <b>Effects</b>: Returns the number of the elements that can be inserted
   //!   at the back without allocating additional memory.
   //!
   //! <b>Throws</b>: Nothing.
   //!
   //! <b>Complexity</b>: Constant.
   //!
   //! <b>Note</b>: Non-standard extension.
   BOOST_CONTAINER_ATTRIBUTE_NODISCARD inline
      size_type back_capacity() const BOOST_NOEXCEPT_OR_NOTHROW
      { return size_type(this->size() + this->prot_back_free_capacity()); }

   //! <b>Effects</b>: Returns the number of the elements that can be inserted
   //!   at the front without allocating additional memory.
   //!
   //! <b>Throws</b>: Nothing.
   //!
   //! <b>Complexity</b>: Constant.
   //!
   //! <b>Note</b>: Non-standard extension.
   BOOST_CONTAINER_ATTRIBUTE_NODISCARD inline
      size_type front_capacity() const BOOST_NOEXCEPT_OR_NOTHROW
      { return size_type(this->size() + this->prot_front_free_capacity()); }

   //! <b>Requires</b>: The container must be "reservable" (is_reservable == true)
   //!
   //! <b>Effects</b>: If n is less than or equal to back_capacity() or the container is not reservable,
   //!   this call has no effect. Otherwise, if it is a request for allocation of additional memory.
   //!   If the request is successful, then back_capacity() is greater than or equal to
   //!   n; otherwise, back_capacity() is unchanged. In either case, size() is unchanged.
   //!
   //! <b>Throws</b>: If memory allocation throws.
   //!
   //! <b>Complexity</b>: Linear to n.
   //!
   //! <b>Note</b>: Non-standard extension.
   void reserve_back(size_type n)
   {
      (void)n;
      BOOST_IF_CONSTEXPR(is_reservable){
         const size_type cur_back_cap = this->back_capacity();
         if (this->back_capacity() < n)
            this->priv_reserve_elements_at_back(size_type(n - cur_back_cap) );
      }
   }

   //! <b>Requires</b>: The container must be "reservable" (is_reservable == true)
   //!
   //! <b>Effects</b>: If n is less than or equal to back_capacity() or the container is not reservable,
   //!   this call has no effect. Otherwise, if it is a request for allocation of additional memory.
   //!   If the request is successful, then back_capacity() is greater than or equal to
   //!   n; otherwise, back_capacity() is unchanged. In either case, size() is unchanged.
   //!
   //! <b>Throws</b>: If memory allocation throws.
   //!
   //! <b>Complexity</b>: Linear to n.
   //!
   //! <b>Note</b>: Non-standard extension.
   void reserve_front(size_type n)
   {
      (void)n;
      BOOST_IF_CONSTEXPR(is_reservable){
         const size_type cur_back_cap = this->front_capacity();
         if (this->front_capacity() < n)
            this->priv_reserve_elements_at_front(size_type(n - cur_back_cap) );
      }
   }

   //! <b>Effects</b>: Returns the largest possible size of the deque.
   //!
   //! <b>Throws</b>: Nothing.
   //!
   //! <b>Complexity</b>: Constant.
   BOOST_CONTAINER_ATTRIBUTE_NODISCARD inline
      size_type max_size() const BOOST_NOEXCEPT_OR_NOTHROW
      { return allocator_traits_type::max_size(this->alloc()); }

   //! <b>Effects</b>: Inserts or erases elements at the end such that
   //!   the size becomes n. New elements are value initialized.
   //!
   //! <b>Throws</b>: If memory allocation throws, or T's constructor throws.
   //!
   //! <b>Complexity</b>: Linear to the difference between size() and new_size.
   void resize(size_type new_size)
   {
      const size_type len = this->size();
      if (new_size < len)
         this->priv_erase_last_n(size_type(len - new_size));
      else{
         const size_type n = size_type(new_size - this->size());
         dtl::insert_value_initialized_n_proxy<ValAllocator> proxy;
         this->priv_insert_back_aux_impl(n, proxy);
      }
   }

   //! <b>Effects</b>: Inserts or erases elements at the end such that
   //!   the size becomes n. New elements are default initialized.
   //!
   //! <b>Throws</b>: If memory allocation throws, or T's constructor throws.
   //!
   //! <b>Complexity</b>: Linear to the difference between size() and new_size.
   //!
   //! <b>Note</b>: Non-standard extension
   void resize(size_type new_size, default_init_t)
   {
      const size_type len = size();
      if (new_size < len)
         this->priv_erase_last_n(size_type(len - new_size));
      else{
         const size_type n = size_type(new_size - this->size());
         dtl::insert_default_initialized_n_proxy<ValAllocator> proxy;
         this->priv_insert_back_aux_impl(n, proxy);
      }
   }

   //! <b>Effects</b>: Inserts or erases elements at the end such that
   //!   the size becomes n. New elements are copy constructed from x.
   //!
   //! <b>Throws</b>: If memory allocation throws, or T's copy constructor throws.
   //!
   //! <b>Complexity</b>: Linear to the difference between size() and new_size.
   void resize(size_type new_size, const value_type& x)
   {
      const size_type sz = this->size();
      if (new_size < sz)
         this->priv_erase_last_n(size_type(sz - new_size));
      else {
         const size_type n = size_type(new_size - sz);
         dtl::insert_n_copies_proxy<ValAllocator> proxy(x);
         this->priv_insert_back_aux_impl(n, proxy);
      }
   }

   //! <b>Effects</b>: Tries to deallocate the excess of memory created
   //!   with previous allocations. The size of the deque is unchanged
   //!
   //! <b>Throws</b>: If memory allocation throws.
   //!
   //! <b>Complexity</b>: Constant.
   void shrink_to_fit()
   {
      //This deque implementation already
      //deallocates excess nodes when erasing
      //so there is nothing to do except for
      //empty deque
      if(this->empty()){
         this->prot_clear_map();
      }
   }

   //////////////////////////////////////////////
   //
   //               element access
   //
   //////////////////////////////////////////////

   //! <b>Requires</b>: !empty()
   //!
   //! <b>Effects</b>: Returns a reference to the first
   //!   element of the container.
   //!
   //! <b>Throws</b>: Nothing.
   //!
   //! <b>Complexity</b>: Constant.
   BOOST_CONTAINER_ATTRIBUTE_NODISCARD inline
      reference front() BOOST_NOEXCEPT_OR_NOTHROW
   {
      BOOST_ASSERT(!this->empty());
      return *this->prot_start_cur_unchecked();
   }

   //! <b>Requires</b>: !empty()
   //!
   //! <b>Effects</b>: Returns a const reference to the first element
   //!   from the beginning of the container.
   //!
   //! <b>Throws</b>: Nothing.
   //!
   //! <b>Complexity</b>: Constant.
   BOOST_CONTAINER_ATTRIBUTE_NODISCARD inline
      const_reference front() const BOOST_NOEXCEPT_OR_NOTHROW
   {
      BOOST_ASSERT(!this->empty());
      return *this->prot_start_cur_unchecked();
   }

   //! <b>Requires</b>: !empty()
   //!
   //! <b>Effects</b>: Returns a reference to the last
   //!   element of the container.
   //!
   //! <b>Throws</b>: Nothing.
   //!
   //! <b>Complexity</b>: Constant.
   BOOST_CONTAINER_ATTRIBUTE_NODISCARD inline
      reference back() BOOST_NOEXCEPT_OR_NOTHROW
   {
      BOOST_ASSERT(!this->empty());
      return *this->prot_last_cur_unchecked();
   }

   //! <b>Requires</b>: !empty()
   //!
   //! <b>Effects</b>: Returns a const reference to the last
   //!   element of the container.
   //!
   //! <b>Throws</b>: Nothing.
   //!
   //! <b>Complexity</b>: Constant.
   BOOST_CONTAINER_ATTRIBUTE_NODISCARD inline
      const_reference back() const BOOST_NOEXCEPT_OR_NOTHROW
   {
      BOOST_ASSERT(!this->empty());
      return *this->prot_last_cur_unchecked();
   }

   //! <b>Requires</b>: size() > n.
   //!
   //! <b>Effects</b>: Returns a reference to the nth element
   //!   from the beginning of the container.
   //!
   //! <b>Throws</b>: Nothing.
   //!
   //! <b>Complexity</b>: Constant.
   BOOST_CONTAINER_ATTRIBUTE_NODISCARD inline
      reference operator[](size_type n) BOOST_NOEXCEPT_OR_NOTHROW
   {
      BOOST_ASSERT(this->size() > n);
      return *this->prot_nth(n);
   }

   //! <b>Requires</b>: size() > n.
   //!
   //! <b>Effects</b>: Returns a const reference to the nth element
   //!   from the beginning of the container.
   //!
   //! <b>Throws</b>: Nothing.
   //!
   //! <b>Complexity</b>: Constant.
   BOOST_CONTAINER_ATTRIBUTE_NODISCARD inline
      const_reference operator[](size_type n) const BOOST_NOEXCEPT_OR_NOTHROW
   {
      BOOST_ASSERT(this->size() > n);
      return *this->prot_nth(n);
   }

   //! <b>Requires</b>: size() >= n.
   //!
   //! <b>Effects</b>: Returns an iterator to the nth element
   //!   from the beginning of the container. Returns end()
   //!   if n == size().
   //!
   //! <b>Throws</b>: Nothing.
   //!
   //! <b>Complexity</b>: Constant.
   //!
   //! <b>Note</b>: Non-standard extension
   BOOST_CONTAINER_ATTRIBUTE_NODISCARD inline
      iterator nth(size_type n) BOOST_NOEXCEPT_OR_NOTHROW
   {
      BOOST_ASSERT(this->size() >= n);
      return this->prot_nth(n);
   }

   //! <b>Requires</b>: size() >= n.
   //!
   //! <b>Effects</b>: Returns a const_iterator to the nth element
   //!   from the beginning of the container. Returns end()
   //!   if n == size().
   //!
   //! <b>Throws</b>: Nothing.
   //!
   //! <b>Complexity</b>: Constant.
   //!
   //! <b>Note</b>: Non-standard extension
   BOOST_CONTAINER_ATTRIBUTE_NODISCARD inline
      const_iterator nth(size_type n) const BOOST_NOEXCEPT_OR_NOTHROW
   {
      BOOST_ASSERT(this->size() >= n);
      return this->prot_nth(n);
   }

   //! <b>Requires</b>: begin() <= p <= end().
   //!
   //! <b>Effects</b>: Returns the index of the element pointed by p
   //!   and size() if p == end().
   //!
   //! <b>Throws</b>: Nothing.
   //!
   //! <b>Complexity</b>: Constant.
   //!
   //! <b>Note</b>: Non-standard extension
   BOOST_CONTAINER_ATTRIBUTE_NODISCARD inline
      size_type index_of(iterator p) BOOST_NOEXCEPT_OR_NOTHROW
   {
      //Range checked priv_index_of
      return this->priv_index_of(p);
   }

   //! <b>Requires</b>: begin() <= p <= end().
   //!
   //! <b>Effects</b>: Returns the index of the element pointed by p
   //!   and size() if p == end().
   //!
   //! <b>Throws</b>: Nothing.
   //!
   //! <b>Complexity</b>: Constant.
   //!
   //! <b>Note</b>: Non-standard extension
   BOOST_CONTAINER_ATTRIBUTE_NODISCARD inline
      size_type index_of(const_iterator p) const BOOST_NOEXCEPT_OR_NOTHROW
   {
      //Range checked priv_index_of
      return this->priv_index_of(p);
   }

   //! <b>Requires</b>: size() > n.
   //!
   //! <b>Effects</b>: Returns a reference to the nth element
   //!   from the beginning of the container.
   //!
   //! <b>Throws</b>: range_error if n >= size()
   //!
   //! <b>Complexity</b>: Constant.
   BOOST_CONTAINER_ATTRIBUTE_NODISCARD inline
      reference at(size_type n)
   {
      this->priv_throw_if_out_of_range(n);
      return (*this)[n];
   }

   //! <b>Requires</b>: size() > n.
   //!
   //! <b>Effects</b>: Returns a const reference to the nth element
   //!   from the beginning of the container.
   //!
   //! <b>Throws</b>: range_error if n >= size()
   //!
   //! <b>Complexity</b>: Constant.
   BOOST_CONTAINER_ATTRIBUTE_NODISCARD inline
      const_reference at(size_type n) const
   {
      this->priv_throw_if_out_of_range(n);
      return (*this)[n];
   }

   //////////////////////////////////////////////
   //
   //                modifiers
   //
   //////////////////////////////////////////////

   #if !defined(BOOST_NO_CXX11_VARIADIC_TEMPLATES) || defined(BOOST_CONTAINER_DOXYGEN_INVOKED)

   //! <b>Effects</b>: Inserts an object of type T constructed with
   //!   std::forward<Args>(args)... in the beginning of the deque.
   //!
   //! <b>Returns</b>: A reference to the created object.
   //!
   //! <b>Throws</b>: If memory allocation throws or the in-place constructor throws.
   //!
   //! <b>Complexity</b>: Amortized constant time
   template <class... Args>
   reference emplace_front(BOOST_FWD_REF(Args)... args)
   {
      value_type *pr = this->prot_push_front_simple_pos();
      if(BOOST_LIKELY(pr != 0)){
         allocator_traits_type::construct
            ( this->alloc()
            , pr
            , boost::forward<Args>(args)...);
         this->prot_dec_start();
         return *pr;
      }
      else{
         typedef dtl::insert_nonmovable_emplace_proxy<ValAllocator, Args...> type;
         return *this->priv_insert_front_aux_impl(1, type(boost::forward<Args>(args)...));
      }
   }

   //! <b>Effects</b>: Inserts an object of type T constructed with
   //!   std::forward<Args>(args)... in the end of the deque.
   //!
   //! <b>Returns</b>: A reference to the created object.
   //!
   //! <b>Throws</b>: If memory allocation throws or the in-place constructor throws.
   //!
   //! <b>Complexity</b>: Amortized constant time
   template <class... Args>
   reference emplace_back(BOOST_FWD_REF(Args)... args)
   {
      value_type *pr = this->prot_push_back_simple_pos();
      if(BOOST_LIKELY(pr != 0)){
         allocator_traits_type::construct
            ( this->alloc(), pr, boost::forward<Args>(args)...);
         this->prot_inc_finish();
         return *pr;
      }
      else{
         typedef dtl::insert_nonmovable_emplace_proxy<ValAllocator, Args...> type;
         return *this->priv_insert_back_aux_impl(1, type(boost::forward<Args>(args)...));
      }
   }

   //! <b>Requires</b>: p must be a valid iterator of *this.
   //!
   //! <b>Effects</b>: Inserts an object of type T constructed with
   //!   std::forward<Args>(args)... before p
   //!
   //! <b>Throws</b>: If memory allocation throws or the in-place constructor throws.
   //!
   //! <b>Complexity</b>: If p is end(), amortized constant time
   //!   Linear time otherwise.
   template <class... Args>
   iterator emplace(const_iterator p, BOOST_FWD_REF(Args)... args)
   {
      const size_type elemsbefore = this->prot_it_to_start_off(p);
      const size_type length      = this->prot_size();

      if (!elemsbefore) {
         this->emplace_front(boost::forward<Args>(args)...);
         return this->begin();
      }
      else if (elemsbefore == length) {
         this->emplace_back(boost::forward<Args>(args)...);
         return this->prot_back_it();
      }
      else {
         typedef dtl::insert_emplace_proxy<ValAllocator, Args...> type;
         return this->priv_insert_middle_aux_impl(elemsbefore, 1, type(boost::forward<Args>(args)...));
      }
   }

   #else //!defined(BOOST_NO_CXX11_VARIADIC_TEMPLATES)

   #define BOOST_CONTAINER_DEQUE_EMPLACE_CODE(N) \
   BOOST_MOVE_TMPL_LT##N BOOST_MOVE_CLASS##N BOOST_MOVE_GT##N\
   reference emplace_front(BOOST_MOVE_UREF##N)\
   {\
      value_type *pr = this->prot_push_front_simple_pos();\
      if(BOOST_LIKELY(pr != 0)){\
         allocator_traits_type::construct\
            ( this->alloc(), pr BOOST_MOVE_I##N BOOST_MOVE_FWD##N);\
         this->prot_dec_start();\
         return *pr;\
      }\
      else{\
         typedef dtl::insert_nonmovable_emplace_proxy##N\
               <ValAllocator BOOST_MOVE_I##N BOOST_MOVE_TARG##N> type;\
         return *priv_insert_front_aux_impl(1, type(BOOST_MOVE_FWD##N));\
      }\
   }\
   \
   BOOST_MOVE_TMPL_LT##N BOOST_MOVE_CLASS##N BOOST_MOVE_GT##N\
   reference emplace_back(BOOST_MOVE_UREF##N)\
   {\
      value_type *pr = this->prot_push_back_simple_pos();\
      if(BOOST_LIKELY(pr != 0)){\
         allocator_traits_type::construct\
            ( this->alloc(), pr BOOST_MOVE_I##N BOOST_MOVE_FWD##N);\
         this->prot_inc_finish();\
         return *pr;\
      }\
      else{\
         typedef dtl::insert_nonmovable_emplace_proxy##N\
               <ValAllocator BOOST_MOVE_I##N BOOST_MOVE_TARG##N> type;\
         return *priv_insert_back_aux_impl(1, type(BOOST_MOVE_FWD##N));\
      }\
   }\
   \
   BOOST_MOVE_TMPL_LT##N BOOST_MOVE_CLASS##N BOOST_MOVE_GT##N\
   iterator emplace(const_iterator p BOOST_MOVE_I##N BOOST_MOVE_UREF##N)\
   {\
      const size_type elemsbefore = this->prot_it_to_start_off(p);\
      const size_type length      = this->prot_size();\
      \
      if (!elemsbefore) {\
         this->emplace_front(BOOST_MOVE_FWD##N);\
         return this->begin();\
      }\
      else if (elemsbefore == length) {\
         this->emplace_back(BOOST_MOVE_FWD##N);\
         return this->prot_back_it();\
      }\
      else {\
         typedef dtl::insert_emplace_proxy_arg##N\
               <ValAllocator BOOST_MOVE_I##N BOOST_MOVE_TARG##N> type;\
         return this->priv_insert_middle_aux_impl(elemsbefore, 1, type(BOOST_MOVE_FWD##N));\
      }\
   }\
   //
   BOOST_MOVE_ITERATE_0TO9(BOOST_CONTAINER_DEQUE_EMPLACE_CODE)
   #undef BOOST_CONTAINER_DEQUE_EMPLACE_CODE

   #endif   // !defined(BOOST_NO_CXX11_VARIADIC_TEMPLATES)

   #if defined(BOOST_CONTAINER_DOXYGEN_INVOKED)
   //! <b>Effects</b>: Inserts a copy of x at the front of the deque.
   //!
   //! <b>Throws</b>: If memory allocation throws or
   //!   T's copy constructor throws.
   //!
   //! <b>Complexity</b>: Amortized constant time.
   void push_front(const T &x);

   //! <b>Effects</b>: Constructs a new element in the front of the deque
   //!   and moves the resources of x to this new element.
   //!
   //! <b>Throws</b>: If memory allocation throws.
   //!
   //! <b>Complexity</b>: Amortized constant time.
   void push_front(T &&x);
   #else
   BOOST_MOVE_CONVERSION_AWARE_CATCH(push_front, T, void, priv_push_front)
   #endif

   #if defined(BOOST_CONTAINER_DOXYGEN_INVOKED)
   //! <b>Effects</b>: Inserts a copy of x at the end of the deque.
   //!
   //! <b>Throws</b>: If memory allocation throws or
   //!   T's copy constructor throws.
   //!
   //! <b>Complexity</b>: Amortized constant time.
   void push_back(const T &x);

   //! <b>Effects</b>: Constructs a new element in the end of the deque
   //!   and moves the resources of x to this new element.
   //!
   //! <b>Throws</b>: If memory allocation throws.
   //!
   //! <b>Complexity</b>: Amortized constant time.
   void push_back(T &&x);
   #else
   BOOST_MOVE_CONVERSION_AWARE_CATCH(push_back, T, void, priv_push_back)
   #endif

   #if defined(BOOST_CONTAINER_DOXYGEN_INVOKED)

   //! <b>Requires</b>: p must be a valid iterator of *this.
   //!
   //! <b>Effects</b>: Insert a copy of x before p.
   //!
   //! <b>Returns</b>: an iterator to the inserted element.
   //!
   //! <b>Throws</b>: If memory allocation throws or x's copy constructor throws.
   //!
   //! <b>Complexity</b>: If p is end(), amortized constant time
   //!   Linear time otherwise.
   iterator insert(const_iterator p, const T &x);

   //! <b>Requires</b>: p must be a valid iterator of *this.
   //!
   //! <b>Effects</b>: Insert a new element before p with x's resources.
   //!
   //! <b>Returns</b>: an iterator to the inserted element.
   //!
   //! <b>Throws</b>: If memory allocation throws.
   //!
   //! <b>Complexity</b>: If p is end(), amortized constant time
   //!   Linear time otherwise.
   iterator insert(const_iterator p, T &&x);
   #else
   BOOST_MOVE_CONVERSION_AWARE_CATCH_1ARG(insert, T, iterator, priv_insert, const_iterator, const_iterator)
   #endif

   //! <b>Requires</b>: pos must be a valid iterator of *this.
   //!
   //! <b>Effects</b>: Insert n copies of x before pos.
   //!
   //! <b>Returns</b>: an iterator to the first inserted element or pos if n is 0.
   //!
   //! <b>Throws</b>: If memory allocation throws or T's copy constructor throws.
   //!
   //! <b>Complexity</b>: Linear to n.
   inline iterator insert(const_iterator pos, size_type n, const value_type& x)
   {
      BOOST_ASSERT(this->priv_in_range_or_end(pos));
      dtl::insert_n_copies_proxy<ValAllocator> proxy(x);
      return this->priv_insert_aux_impl(pos, n, proxy);
   }

   //! <b>Requires</b>: pos must be a valid iterator of *this.
   //!
   //! <b>Effects</b>: Insert a copy of the [first, last) range before pos.
   //!
   //! <b>Returns</b>: an iterator to the first inserted element or pos if first == last.
   //!
   //! <b>Throws</b>: If memory allocation throws, T's constructor from a
   //!   dereferenced InIt throws or T's copy constructor throws.
   //!
   //! <b>Complexity</b>: Linear to distance [first, last).
   template <class InIt>
   iterator insert(const_iterator pos, InIt first, InIt last
      #if !defined(BOOST_CONTAINER_DOXYGEN_INVOKED)
      , typename dtl::disable_if_or
         < void
         , dtl::is_convertible<InIt, size_type>
         , dtl::is_not_input_iterator<InIt>
         >::type * = 0
      #endif
      )
   {
      BOOST_ASSERT(this->priv_in_range_or_end(pos));
      size_type n = 0;
      iterator it(pos.unconst());
      for(;first != last; ++first, ++n){
         it = this->emplace(it, *first);
         ++it;
      }
      it -= difference_type(n);
      return it;
   }

#if !defined(BOOST_NO_CXX11_HDR_INITIALIZER_LIST)
   //! <b>Requires</b>: pos must be a valid iterator of *this.
   //!
   //! <b>Effects</b>: Insert a copy of the [il.begin(), il.end()) range before pos.
   //!
   //! <b>Returns</b>: an iterator to the first inserted element or pos if il.begin() == il.end().
   //!
   //! <b>Throws</b>: If memory allocation throws, T's constructor from a
   //!   dereferenced std::initializer_list throws or T's copy constructor throws.
   //!
   //! <b>Complexity</b>: Linear to distance [il.begin(), il.end()).
   inline iterator insert(const_iterator pos, std::initializer_list<value_type> il)
   {
      //Range check os pos is done in insert()
      return insert(pos, il.begin(), il.end());
   }
#endif

   #if !defined(BOOST_CONTAINER_DOXYGEN_INVOKED)
   template <class FwdIt>
   inline iterator insert(const_iterator p, FwdIt first, FwdIt last
      #if !defined(BOOST_CONTAINER_DOXYGEN_INVOKED)
      , typename dtl::disable_if_or
         < void
         , dtl::is_convertible<FwdIt, size_type>
         , dtl::is_input_iterator<FwdIt>
         >::type * = 0
      #endif
      )
   {
      BOOST_ASSERT(this->priv_in_range_or_end(p));

      typedef typename iter_size<FwdIt>::type it_size_type;
      const it_size_type sz = boost::container::iterator_udistance(first, last);
      if (BOOST_UNLIKELY(sz > size_type(-1))){
         boost::container::throw_length_error("vector::insert, FwdIt's max length reached");
      }
      const size_type n = static_cast<size_type>(sz);

      dtl::insert_range_proxy<ValAllocator, FwdIt> proxy(first);
      return this->priv_insert_aux_impl( p, n, proxy);
   }
   #endif

   //! <b>Effects</b>: Removes the first element from the deque.
   //!
   //! <b>Throws</b>: Nothing.
   //!
   //! <b>Complexity</b>: Constant time.
   void pop_front() BOOST_NOEXCEPT_OR_NOTHROW
   {
      BOOST_ASSERT(!this->empty());
      if (BOOST_LIKELY(this->prot_pop_front_simple_available())) {
         allocator_traits_type::destroy
            ( this->alloc()
            , boost::movelib::to_raw_pointer(this->prot_start_cur_unchecked())
            );
         this->prot_inc_start();
      }
      else
         this->priv_pop_front_aux();
   }

   //! <b>Effects</b>: Removes the last element from the deque.
   //!
   //! <b>Throws</b>: Nothing.
   //!
   //! <b>Complexity</b>: Constant time.
   void pop_back() BOOST_NOEXCEPT_OR_NOTHROW
   {
      BOOST_ASSERT(!this->empty());
      if (BOOST_LIKELY(this->prot_pop_back_simple_available())) {
         this->prot_dec_finish();
         allocator_traits_type::destroy
            ( this->alloc()
            , boost::movelib::to_raw_pointer(this->prot_finish_cur_unchecked())
            );
      }
      else
         this->priv_pop_back_aux();
   }

   //! <b>Effects</b>: Erases the element at p.
   //!
   //! <b>Throws</b>: Nothing.
   //!
   //! <b>Complexity</b>: Linear to the elements between pos and the
   //!   last element (if pos is near the end) or the first element
   //!   if(pos is near the beginning).
   //!   Constant if pos is the first or the last element.
   iterator erase(const_iterator pos) BOOST_NOEXCEPT_OR_NOTHROW
   {
      BOOST_ASSERT(this->priv_in_range(pos));
      iterator next = pos.unconst();
      ++next;
      const size_type index = this->prot_it_to_start_off(pos);
      const size_type sz    = this->prot_size();
      if (index < sz/2u) {
         this->priv_segmented_move_backward_n(pos.unconst(), index, next);
         pop_front();
         return next;
      }
      else {
         this->priv_segmented_move_n(next, size_type(sz - size_type(index + 1u)), pos.unconst());
         pop_back();
         return pos.unconst();
      }
   }

   //! <b>Effects</b>: Erases the elements pointed by [first, last).
   //!
   //! <b>Throws</b>: Nothing.
   //!
   //! <b>Complexity</b>: Linear to the distance between first and
   //!   last plus the elements between pos and the
   //!   last element (if pos is near the end) or the first element
   //!   if(pos is near the beginning).
   iterator erase(const_iterator first, const_iterator last) BOOST_NOEXCEPT_OR_NOTHROW
   {
      BOOST_ASSERT(first == last ||
         (first < last && this->priv_in_range(first) && this->priv_in_range_or_end(last)));
      const size_type n = static_cast<size_type>(last - first);
      if (n == this->prot_size()) {
         this->clear();
         return this->end();
      }
      else {
         const size_type elems_before = this->prot_it_to_start_off(first);
         const size_type elems_after  = size_type(this->prot_size() - n - elems_before);
         if (elems_before < elems_after) {
            const iterator old_start = this->begin();
            iterator new_start = this->priv_segmented_move_backward_n(first.unconst(), elems_before, last.unconst());
            this->prot_destroy_range(old_start, new_start);
            this->prot_deallocate_nodes_if_not_reservable(old_start.get_node(), new_start.m_node);
            this->prot_inc_start(n);
         }
         else {
            const iterator old_finish = this->end();
            iterator new_finish = this->priv_segmented_move_n(last.unconst(), elems_after, first.unconst());
            this->prot_destroy_range(new_finish, old_finish);
            this->prot_deallocate_nodes_if_not_reservable(new_finish.m_node + 1, old_finish.get_node() + 1);
            this->prot_dec_finish(n);
         }
         return this->nth(elems_before);
      }
   }

   //! <b>Effects</b>: Swaps the contents of *this and x.
   //!
   //! <b>Throws</b>: Nothing.
   //!
   //! <b>Complexity</b>: Constant.
   inline void swap(deque &x)
      BOOST_NOEXCEPT_IF(allocator_traits_type::propagate_on_container_swap::value
                               || allocator_traits_type::is_always_equal::value)
   {
      this->swap_members(x);
      dtl::bool_<allocator_traits_type::propagate_on_container_swap::value> flag;
      dtl::swap_alloc(this->alloc(), x.alloc(), flag);
      dtl::swap_alloc(this->ptr_alloc(), x.ptr_alloc(), flag);
   }

   //! <b>Effects</b>: Erases all the elements of the deque.
   //!
   //! <b>Throws</b>: Nothing.
   //!
   //! <b>Complexity</b>: Linear to the number of elements in the deque.
   void clear() BOOST_NOEXCEPT_OR_NOTHROW
   {
      if (!this->empty()) {
         const iterator start = this->prot_start();
         const iterator finish = this->prot_finish();
         const index_pointer start_node =  start.get_node();
         const index_pointer finish_node = finish.get_node();
         for (index_pointer node = start_node + 1; node < finish_node; ++node) {
            this->prot_destroy_range(*node, *node + get_block_ssize());
            this->prot_deallocate_node_if_not_reservable(*node);
         }

         if (start_node != finish_node) {
            this->prot_destroy_range(start.get_cur(), start.get_last());
            this->prot_destroy_range(finish.get_first(), finish.get_cur());
            this->prot_deallocate_node_if_not_reservable(finish.get_first());
         }
         else
            this->prot_destroy_range(start.get_cur(), finish.get_cur());

         this->prot_reset_finish_to_start();
      }
   }

   //! <b>Effects</b>: Returns true if x and y are equal
   //!
   //! <b>Complexity</b>: Linear to the number of elements in the container.
   BOOST_CONTAINER_ATTRIBUTE_NODISCARD inline
      friend bool operator==(const deque& x, const deque& y)
   {  return x.size() == y.size() && ::boost::container::algo_equal(x.begin(), x.end(), y.begin());  }

   //! <b>Effects</b>: Returns true if x and y are unequal
   //!
   //! <b>Complexity</b>: Linear to the number of elements in the container.
   BOOST_CONTAINER_ATTRIBUTE_NODISCARD inline
      friend bool operator!=(const deque& x, const deque& y)
   {  return !(x == y); }

   //! <b>Effects</b>: Returns true if x is less than y
   //!
   //! <b>Complexity</b>: Linear to the number of elements in the container.
   BOOST_CONTAINER_ATTRIBUTE_NODISCARD inline
      friend bool operator<(const deque& x, const deque& y)
   {  return ::boost::container::algo_lexicographical_compare(x.begin(), x.end(), y.begin(), y.end());  }

   //! <b>Effects</b>: Returns true if x is greater than y
   //!
   //! <b>Complexity</b>: Linear to the number of elements in the container.
   BOOST_CONTAINER_ATTRIBUTE_NODISCARD inline
      friend bool operator>(const deque& x, const deque& y)
   {  return y < x;  }

   //! <b>Effects</b>: Returns true if x is equal or less than y
   //!
   //! <b>Complexity</b>: Linear to the number of elements in the container.
   BOOST_CONTAINER_ATTRIBUTE_NODISCARD inline
      friend bool operator<=(const deque& x, const deque& y)
   {  return !(y < x);  }

   //! <b>Effects</b>: Returns true if x is equal or greater than y
   //!
   //! <b>Complexity</b>: Linear to the number of elements in the container.
   BOOST_CONTAINER_ATTRIBUTE_NODISCARD inline
      friend bool operator>=(const deque& x, const deque& y)
   {  return !(x < y);  }

   //! <b>Effects</b>: x.swap(y)
   //!
   //! <b>Complexity</b>: Constant.
   inline friend void swap(deque& x, deque& y)
       BOOST_NOEXCEPT_IF(BOOST_NOEXCEPT(x.swap(y)))
   {  x.swap(y);  }

   #ifndef BOOST_CONTAINER_DOXYGEN_INVOKED
   private:

   template <class InsertProxy>
   void priv_assign(size_type n, InsertProxy proxy)
   {
      const size_type sz  = this->prot_size();
      this->priv_segmented_proxy_copy_n_and_update(this->begin(), sz < n ? sz : n, proxy);
      if (n > sz) {
         this->priv_insert_back_aux_impl(size_type(n - sz), proxy);
      }
      else{
         this->priv_erase_last_n(size_type(sz - n));
      }
   }

   void priv_move_assign(BOOST_RV_REF(deque) x, dtl::bool_<true> /*steal_resources*/)
   {
      //Destroy objects but retain memory in case x reuses it in the future
      this->clear();
      //Move allocator if needed
      dtl::bool_<allocator_traits_type::propagate_on_container_move_assignment::value> flag;
      dtl::move_alloc(this->alloc(), x.alloc(), flag);
      dtl::move_alloc(this->ptr_alloc(), x.ptr_alloc(), flag);
      //Nothrow swap
      this->swap_members(x);
   }

   void priv_move_assign(BOOST_RV_REF(deque) x, dtl::bool_<false> /*steal_resources*/)
   {
      //We can't guarantee a compile-time equal allocator or propagation so fallback to runtime
      //Resources can be transferred if both allocators are equal
      if (this->alloc() == x.alloc()) {
         this->priv_move_assign(boost::move(x), dtl::true_());
      }
      else {
         this->assign(boost::make_move_iterator(x.begin()), boost::make_move_iterator(x.end()));
      }
   }

   inline size_type priv_index_of(const_iterator p) const
   {
      BOOST_ASSERT(this->cbegin() <= p);
      BOOST_ASSERT(p <= this->cend());
      return this->prot_it_to_start_off(p);
   }

   void priv_erase_last_n(size_type n)
   {
      const size_type sz = this->prot_size();
      BOOST_ASSERT(n <= sz);

      if(sz) {
         const iterator old_finish = this->prot_finish();
         const iterator new_finish = old_finish - difference_type(n);
         this->prot_destroy_range(new_finish, old_finish);
         this->prot_deallocate_nodes_if_not_reservable(new_finish.get_node() + 1, old_finish.get_node() + 1);
         this->prot_dec_finish(n);
      }
   }

   void priv_throw_if_out_of_range(size_type n) const
   {
      if (n >= this->size())
         throw_out_of_range("deque::at out of range");
   }

   inline bool priv_in_range(const_iterator pos) const
   {
      return (this->begin() <= pos) && (pos < this->end());
   }

   inline bool priv_in_range_or_end(const_iterator pos) const
   {
      return (this->begin() <= pos) && (pos <= this->end());
   }

   template <class U>
   BOOST_CONTAINER_FORCEINLINE iterator priv_insert(const_iterator p, BOOST_FWD_REF(U) x)
   {
      return this->emplace(p, ::boost::forward<U>(x));
   }

   template <class U>
   BOOST_CONTAINER_FORCEINLINE void priv_push_back(BOOST_FWD_REF(U) u)
   {
      this->emplace_back(::boost::forward<U>(u));
   }

   template <class U>
   BOOST_CONTAINER_FORCEINLINE void priv_push_front(BOOST_FWD_REF(U) u)
   {
      this->emplace_front(::boost::forward<U>(u));
   }

   void prot_destroy_range(iterator start, iterator finish)
   {
      (void)start; (void)finish;
      BOOST_IF_CONSTEXPR(!Base::traits_t::trivial_dctr){
         const index_pointer start_node =  start.get_node();
         const index_pointer finish_node = finish.get_node();
      
         //In a sane deque both should be null or non-null
         BOOST_ASSERT(!start_node == !finish_node);
         if(start_node){
            for (index_pointer node = start_node + 1; node < finish_node; ++node) {
               this->prot_destroy_range(*node, *node + get_block_ssize());
            }

            if (start_node != finish_node) {
               this->prot_destroy_range(start.get_cur(), start.get_last());
               this->prot_destroy_range(finish.get_first(), finish.get_cur());
            }
            else
               this->prot_destroy_range(start.get_cur(), finish.get_cur());
         }
      }
   }

   void prot_destroy_range(pointer p, pointer p2)
   {
      (void)p; (void)p2;
      BOOST_IF_CONSTEXPR(!Base::traits_t::trivial_dctr){
         for(;p != p2; ++p){
            allocator_traits_type::destroy(this->alloc(), boost::movelib::iterator_to_raw_pointer(p));
         }
      }
   }

   template<class InsertProxy>
   iterator priv_insert_middle_aux_impl(const size_type elemsbefore, const size_type n, InsertProxy proxy)
   {
      typedef dtl::bool_<dtl::is_single_value_proxy<InsertProxy>::value> single_t;
      BOOST_ASSERT(!single_t::value || n == 1);

      const size_type length = this->prot_size();
      if (elemsbefore < length / 2) {
         this->priv_reserve_elements_at_front(n);
         const iterator old_start = this->prot_start();
         iterator new_start = old_start;
         priv_itsub(new_start, n, single_t());

         BOOST_ASSERT(!single_t::value || elemsbefore >= 1);

         if(single_t::value || elemsbefore >= n) {
            iterator start_n = old_start;
            priv_itadd(start_n, n, single_t());
            BOOST_CONTAINER_TRY {
               this->priv_segmented_uninitialized_move_alloc_n(old_start, n, new_start, single_t());
            }
            BOOST_CONTAINER_CATCH(...) {
               this->prot_deallocate_nodes_if_not_reservable(new_start.m_node, old_start.m_node);
               BOOST_CONTAINER_RETHROW
            }
            BOOST_CONTAINER_CATCH_END

            this->prot_dec_start(n);
            iterator p = this->priv_segmented_move_n(start_n, size_type(elemsbefore - n), old_start);
            this->priv_segmented_proxy_copy_n_and_update(p, n, proxy);
            return p;
         }
         else {
            const size_type mid_count = size_type(n - elemsbefore);
            iterator mid_start = old_start - difference_type(mid_count);

            BOOST_CONTAINER_TRY {
               this->priv_segmented_proxy_uninitialized_copy_n_and_update(mid_start, mid_count, proxy);
               this->prot_dec_start(mid_count);
               this->priv_segmented_uninitialized_move_alloc_n(old_start, elemsbefore, new_start);
            }
            BOOST_CONTAINER_CATCH(...) {
               this->prot_deallocate_nodes_if_not_reservable(new_start.m_node, old_start.get_node());
               BOOST_CONTAINER_RETHROW
            }
            BOOST_CONTAINER_CATCH_END
            this->prot_dec_start(size_type(n - mid_count));
            this->priv_segmented_proxy_copy_n_and_update(old_start, elemsbefore, proxy);
            return mid_start;
         }
      }
      else {
         this->priv_reserve_elements_at_back(n);
         const iterator old_finish  = this->prot_finish();
         const size_type elemsafter = size_type(length - elemsbefore);

         BOOST_ASSERT(!single_t::value || elemsafter >= 1);

         if(single_t::value || elemsafter >= n) {
            iterator finish_n = old_finish;
            priv_itsub(finish_n, n, single_t());
            BOOST_CONTAINER_TRY {
               this->priv_segmented_uninitialized_move_alloc_n(finish_n, n, old_finish, single_t());
            }
            BOOST_CONTAINER_CATCH(...) {
               this->prot_deallocate_nodes_if_not_reservable(old_finish.get_node() + 1, (old_finish + difference_type(n)).m_node + 1);
               BOOST_CONTAINER_RETHROW
            }
            BOOST_CONTAINER_CATCH_END

            this->prot_inc_finish(n);
            const size_type move_n = size_type(elemsafter - n);
            this->priv_segmented_move_backward_n(finish_n, move_n, old_finish);
            finish_n -= difference_type(move_n);
            this->priv_segmented_proxy_copy_n_and_update(finish_n, n, proxy);
            return finish_n;
         }
         else {
            const size_type raw_gap = size_type(n - elemsafter);
            iterator pos            = old_finish - difference_type(elemsafter);

            BOOST_CONTAINER_TRY{
               this->priv_segmented_uninitialized_move_alloc_n(pos, elemsafter, old_finish + difference_type(raw_gap));
               BOOST_CONTAINER_TRY{
                  this->priv_segmented_proxy_copy_n_and_update(pos, elemsafter, proxy);
                  this->priv_segmented_proxy_uninitialized_copy_n_and_update(old_finish, raw_gap, proxy);
               }
               BOOST_CONTAINER_CATCH(...) {
                  this->prot_destroy_range(old_finish, old_finish + difference_type(elemsafter));
                  BOOST_CONTAINER_RETHROW
               }
               BOOST_CONTAINER_CATCH_END
            }
            BOOST_CONTAINER_CATCH(...) {
               this->prot_deallocate_nodes_if_not_reservable(old_finish.get_node() + 1, (old_finish + difference_type(n)).m_node + 1);
               BOOST_CONTAINER_RETHROW
            }
            BOOST_CONTAINER_CATCH_END
            this->prot_inc_finish(n);
            return pos;
         }
      }
   }

   template<class InsertProxy>
   iterator priv_insert_aux_impl(const_iterator p, size_type n, InsertProxy proxy)
   {
      const size_type elemsbefore = this->prot_it_to_start_off(p);

      if (!elemsbefore) {
         return this->priv_insert_front_aux_impl(n, proxy);
      }
      else if (elemsbefore == this->prot_size()) {
         return this->priv_insert_back_aux_impl(n, proxy);
      }
      else {
         return this->priv_insert_middle_aux_impl(elemsbefore, n, proxy);
      }
   }

   template <class InsertProxy>
   void priv_segmented_proxy_uninitialized_copy_n_and_update(const iterator first, size_type n, InsertProxy &proxy)
   {
      BOOST_IF_CONSTEXPR (dtl::is_single_value_proxy<InsertProxy>::value) {
         (void)n;
         proxy.uninitialized_copy_n_and_update(this->alloc(), boost::movelib::to_raw_pointer(first.get_cur()), 1u);
      }
      else if (BOOST_LIKELY(n != 0)){ //We might initialize an empty range and current_node might be null
         BOOST_CONSTEXPR_OR_CONST size_type block_size = get_block_size();

         index_pointer current_node = first.get_node();

         BOOST_CONTAINER_TRY{
            const pointer cur = first.get_cur();
            const size_type block = size_type(block_size - size_type(cur - *current_node));
            size_type cnt = n < block ? n: block;
            proxy.uninitialized_copy_n_and_update(this->alloc(), boost::movelib::to_raw_pointer(cur), cnt);
            n = size_type(n - cnt);

            while (n) {
               ++current_node;
               cnt = n < block_size ? n: block_size;
               proxy.uninitialized_copy_n_and_update(this->alloc(), boost::movelib::to_raw_pointer(*current_node), cnt);
               n = size_type(n - cnt);
            }
         }
         BOOST_CONTAINER_CATCH(...) {
            this->prot_destroy_range(first, iterator(*current_node, current_node));
            BOOST_CONTAINER_RETHROW
         }
         BOOST_CONTAINER_CATCH_END
      }
   }


   template <class InsertProxy>
   void priv_segmented_proxy_copy_n_and_update(const iterator first, size_type n, InsertProxy &proxy)
   {
      BOOST_IF_CONSTEXPR (dtl::is_single_value_proxy<InsertProxy>::value) {
         (void)n;
         proxy.copy_n_and_update(this->alloc(), boost::movelib::to_raw_pointer(first.get_cur()), 1u);
      }
      else if (BOOST_LIKELY(n != 0)){ //We might assign an empty range in a default constructed deque
         BOOST_CONSTEXPR_OR_CONST size_type block_size = get_block_size();

         index_pointer current_node = first.get_node();

         const pointer cur = first.get_cur();
         const size_type block = size_type(block_size - size_type(cur - *current_node));
         size_type cnt = n < block ? n: block;
         proxy.copy_n_and_update(this->alloc(), boost::movelib::to_raw_pointer(cur), cnt);
         n = size_type(n - cnt);

         while (n) {
            ++current_node;
            cnt = n < block_size ? n: block_size;
            proxy.copy_n_and_update(this->alloc(), boost::movelib::to_raw_pointer(*current_node), cnt);
            n = size_type(n - cnt);
         }
      }
   }

   template <class InsertProxy>
   void priv_insert_segmented_uninitialized_copy_n_and_update(const iterator first, size_type n, InsertProxy proxy)
   {
      BOOST_CONTAINER_TRY{
         this->priv_segmented_proxy_uninitialized_copy_n_and_update(first, n, proxy);
      }
      BOOST_CONTAINER_CATCH(...) {
         this->prot_deallocate_nodes_if_not_reservable(first.get_node() + 1, (first+difference_type(n)).get_node() + 1);
         BOOST_CONTAINER_RETHROW
      }
      BOOST_CONTAINER_CATCH_END
   }

   template<class It>
   static BOOST_CONTAINER_FORCEINLINE void priv_itadd(It &it, size_type, dtl::bool_<true> /*single element*/)
   {  ++it; }

   template<class It>
   static BOOST_CONTAINER_FORCEINLINE void priv_itadd(It &it, size_type n, dtl::bool_<false> /*!single element*/)
   {  it += difference_type(n); }

   template<class It>
   static BOOST_CONTAINER_FORCEINLINE void priv_itsub(It &it, size_type, dtl::bool_<true> /*single element*/)
   {  --it; }

   template<class It>
   static BOOST_CONTAINER_FORCEINLINE void priv_itsub(It &it, size_type n, dtl::bool_<false> /*!single element*/)
   {  it -= difference_type(n); }

   void priv_segmented_uninitialized_move_alloc_n(iterator first, size_type n, iterator dest, dtl::bool_<true> /*single element*/)
   {
      BOOST_ASSERT(n == 1); (void)n;
      allocator_traits_type::construct
         ( this->alloc()
         , boost::movelib::to_raw_pointer(dest.get_cur())
         , boost::move(*first));
   }

   void priv_segmented_uninitialized_move_alloc_n(iterator first, size_type n, iterator dest, dtl::bool_<false> /*!single element*/)
   {
      if (BOOST_LIKELY(n != 0)) { //Check for empty range, current_node might be null
         BOOST_CONSTEXPR_OR_CONST size_type block_size = get_block_size();

         index_pointer current_node = first.get_node();

         BOOST_CONTAINER_TRY{
            const pointer cur = first.get_cur();
            const size_type block = size_type(block_size - size_type(cur - *current_node));
            size_type cnt = n < block ? n: block;
            dest = ::boost::container::uninitialized_move_alloc_n(this->alloc(), boost::movelib::to_raw_pointer(cur), cnt, dest);
            n = size_type(n - cnt);

            while (n) {
               ++current_node;
               cnt = n < block_size ? n: block_size;
               dest = ::boost::container::uninitialized_move_alloc_n(this->alloc(), boost::movelib::to_raw_pointer(*current_node), cnt, dest);
               n = size_type(n - cnt);
            }
         }
         BOOST_CONTAINER_CATCH(...) {
            this->prot_destroy_range(first, iterator(*current_node, current_node));
            BOOST_CONTAINER_RETHROW
         }
         BOOST_CONTAINER_CATCH_END
      }
   }

   BOOST_CONTAINER_FORCEINLINE void priv_segmented_uninitialized_move_alloc_n(iterator first, size_type n, iterator dest)
   {
      this->priv_segmented_uninitialized_move_alloc_n(first, n, dest, dtl::bool_<false>());
   }

   void priv_segmented_uninitialized_copy_alloc_n(const_iterator first, size_type n, iterator dest)
   {
      if (BOOST_LIKELY(n != 0)) { //We might initialize an empty range and current_node might be null
         BOOST_CONSTEXPR_OR_CONST size_type block_size = get_block_size();

         index_pointer current_node = first.get_node();

         BOOST_CONTAINER_TRY{
            const pointer cur = first.get_cur();
            const size_type block = size_type(block_size - size_type(cur - *current_node));
            size_type cnt = n < block ? n: block;
            dest = ::boost::container::uninitialized_copy_alloc_n(this->alloc(), boost::movelib::to_raw_pointer(cur), cnt, dest);
            n = size_type(n - cnt);

            while (n) {
               ++current_node;
               cnt = n < block_size ? n: block_size;
               dest = ::boost::container::uninitialized_copy_alloc_n(this->alloc(), boost::movelib::to_raw_pointer(*current_node), cnt, dest);
               n = size_type(n - cnt);
            }
         }
         BOOST_CONTAINER_CATCH(...) {
            this->prot_destroy_range(first.unconst(), iterator(*current_node, current_node));
            BOOST_CONTAINER_RETHROW
         }
         BOOST_CONTAINER_CATCH_END
      }
   }

   static iterator priv_segmented_move_n(const_iterator first, size_type n, iterator dest)
   {
      index_pointer current_node = first.get_node();
      BOOST_ASSERT(current_node != index_pointer());

      const pointer cur = first.get_cur();
      BOOST_CONSTEXPR_OR_CONST size_type block_size = get_block_size();

      const size_type block = size_type(block_size - size_type(cur - *current_node));
      size_type cnt = n < block ? n: block;
      dest = ::boost::container::move_n(boost::movelib::to_raw_pointer(cur), cnt, dest);
      n = size_type(n - cnt);

      while (n) {
         ++current_node;
         cnt = n < block_size ? n: block_size;
         dest = ::boost::container::move_n(boost::movelib::to_raw_pointer(*current_node), cnt, dest);
         n = size_type(n - cnt);
      }
      return dest;
   }

   static iterator priv_segmented_move_backward_n(iterator last, size_type n, iterator dest_last)
   {
      index_pointer current_node = last.get_node();
      BOOST_ASSERT(current_node != index_pointer());

      const pointer cur = last.get_cur();
      const size_type block = size_type(cur - *current_node);
      size_type cnt = n < block ? n: block;
      dest_last = ::boost::container::move_backward_n(boost::movelib::to_raw_pointer(cur), cnt, dest_last);
      n = size_type(n - cnt);

      BOOST_CONSTEXPR_OR_CONST size_type block_size = get_block_size();

      while (n) {
         --current_node;
         cnt = n < block_size ? n: block_size;
         dest_last = ::boost::container::move_backward_n(boost::movelib::to_raw_pointer(*current_node + difference_type(block_size)), cnt, dest_last);
         n = size_type(n - cnt);
      }
      return dest_last;
   }

   template <class InsertProxy>
   iterator priv_insert_back_aux_impl(size_type n, InsertProxy proxy)
   {
      this->priv_reserve_elements_at_back(n);
      const iterator old_finish = this->prot_finish();
      this->priv_insert_segmented_uninitialized_copy_n_and_update(old_finish, n, proxy);
      this->prot_inc_finish(n);
      return old_finish;
   }

   template <class InsertProxy>
   iterator priv_insert_front_aux_impl(size_type n, InsertProxy proxy)
   {
      this->priv_reserve_elements_at_front(n);
      const iterator old_start = this->prot_start();
      const iterator new_start = old_start - difference_type(n);
      this->priv_insert_segmented_uninitialized_copy_n_and_update(new_start, n, proxy);
      this->prot_dec_start(n);
      return new_start;
   }

   // Precondition: this->prot_start() and this->prot_finish() have already been initialized,
   // but none of the deque's elements have yet been constructed.
   void priv_fill_initialize(size_type n, const value_type& value)
   {
      dtl::insert_n_copies_proxy<ValAllocator> proxy(value);
      this->priv_segmented_proxy_uninitialized_copy_n_and_update(this->begin(), n, proxy);
   }

   template <class InIt>
   void priv_range_initialize(InIt first, InIt last, typename iterator_enable_if_tag<InIt, std::input_iterator_tag>::type* =0)
   {
      BOOST_CONTAINER_TRY {
         for ( ; first != last; ++first)
            this->emplace_back(*first);
      }
      BOOST_CONTAINER_CATCH(...){
         this->clear();
         BOOST_CONTAINER_RETHROW
      }
      BOOST_CONTAINER_CATCH_END
   }

   template <class FwdIt>
   void priv_range_initialize(FwdIt first, FwdIt last, typename iterator_disable_if_tag<FwdIt, std::input_iterator_tag>::type* =0)
   {
      typedef typename iter_size<FwdIt>::type it_size_type;
      const it_size_type sz = boost::container::iterator_udistance(first, last);
      if (BOOST_UNLIKELY(sz > size_type(-1))){
         boost::container::throw_length_error("vector::insert, FwdIt's max length reached");
      }
      const size_type n = static_cast<size_type>(sz);
      this->prot_initialize_map_and_nodes(n);

      dtl::insert_range_proxy<ValAllocator, FwdIt> proxy(first);
      this->priv_segmented_proxy_uninitialized_copy_n_and_update(this->begin(), n, proxy);
   }

   // Called only if this->prot_finish_cur() == this->prot_finish().get_first().
   void priv_pop_back_aux() BOOST_NOEXCEPT_OR_NOTHROW
   {
      index_pointer ip = this->prot_finish_node();
      this->prot_deallocate_node_if_not_reservable(*ip);
      this->prot_dec_finish();
      --ip;
      allocator_traits_type::destroy
         ( this->alloc()
         , boost::movelib::to_raw_pointer(this->prot_node_last(ip))
         );
   }

   // Called only if this->prot_start_cur() == this->prot_start().get_last() - 1.  Note that
   // if the deque has at least one element (a precondition for this member
   // function), and if this->prot_start_cur() == this->prot_start().get_last(), then the deque
   // must have at least two nodes.
   void priv_pop_front_aux() BOOST_NOEXCEPT_OR_NOTHROW
   {
      const index_pointer ip = this->prot_start_node();
      allocator_traits_type::destroy
         ( this->alloc()
         , boost::movelib::to_raw_pointer(this->prot_node_last(ip))
         );
      this->prot_deallocate_node_if_not_reservable(*ip);
      this->prot_inc_start();
   }

   void priv_reserve_elements_at_front(size_type n)
   {
      const size_type vacancies = this->prot_front_free_capacity();
      typedef dtl::bool_<is_reservable> res_t;

      if (n > vacancies){  //n == 0 handled in the else part
         if(this->members_.m_map){
            this->prot_reallocate_map_and_nodes(res_t(), size_type(n - vacancies), true);
         }
         else {
            this->prot_initialize_map_and_nodes(n);
            this->prot_reset_start_to_finish();
         }
      }
   }

   void priv_reserve_elements_at_back(size_type n)
   {
      const size_type vacancies = this->prot_back_free_capacity();
      typedef dtl::bool_<is_reservable> res_t;

      if (n > vacancies){  //n == 0 handled in the else part
         if(this->members_.m_map){
            this->prot_reallocate_map_and_nodes(res_t(), size_type(n - vacancies), false);
         }
         else{
            this->prot_initialize_map_and_nodes(n);
            this->prot_reset_finish_to_start();
         }
      }
   }

   #endif   //#ifndef BOOST_CONTAINER_DOXYGEN_INVOKED
};

#ifndef BOOST_CONTAINER_NO_CXX17_CTAD
template <typename InputIterator>
deque(InputIterator, InputIterator) -> deque<typename iterator_traits<InputIterator>::value_type>;
template <typename InputIterator, typename Allocator>
deque(InputIterator, InputIterator, Allocator const&) -> deque<typename iterator_traits<InputIterator>::value_type, Allocator>;
#endif

//! <b>Effects</b>: Erases all elements that compare equal to v from the container c.
//!
//! <b>Complexity</b>: Linear.
template <class T, class A, class O, class U>
inline typename deque<T, A, O>::size_type erase(deque<T, A, O>& c, const U& v)
{
   typename deque<T, A, O>::size_type old_size = c.size();
   c.erase(boost::container::remove(c.begin(), c.end(), v), c.end());
   return old_size - c.size();
}

//! <b>Effects</b>: Erases all elements that satisfy the predicate pred from the container c.
//!
//! <b>Complexity</b>: Linear.
template <class T, class A, class O, class Pred>
inline typename deque<T, A, O>::size_type erase_if(deque<T, A, O>& c, Pred pred)
{
   typename deque<T, A, O>::size_type old_size = c.size();
   c.erase(boost::container::remove_if(c.begin(), c.end(), pred), c.end());
   return old_size - c.size();
}

}  //namespace container
}  //namespace boost

#ifndef BOOST_CONTAINER_DOXYGEN_INVOKED

namespace boost {

//!has_trivial_destructor_after_move<> == true_type
//!specialization for optimizations
template <class T, class Allocator, class Options>
struct has_trivial_destructor_after_move<boost::container::deque<T, Allocator, Options> >
{
   typedef typename boost::container::deque<T, Allocator, Options>::allocator_type allocator_type;
   typedef typename ::boost::container::allocator_traits<allocator_type>::pointer pointer;
   BOOST_STATIC_CONSTEXPR bool value = ::boost::has_trivial_destructor_after_move<allocator_type>::value &&
                             ::boost::has_trivial_destructor_after_move<pointer>::value;
};

}

#endif   //#ifndef BOOST_CONTAINER_DOXYGEN_INVOKED

#include <boost/container/detail/config_end.hpp>

#endif //   #ifndef  BOOST_CONTAINER_DEQUE_HPP

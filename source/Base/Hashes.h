#pragma once

#include <tuple>
#include <bit>
#include <functional>

namespace std
{
    template <class A, class B>
    struct hash<pair<A, B>>
    {
        size_t operator()(pair<A, B> const& p) const { return std::rotl(hash<A>{}(p.first), 1) ^ hash<B>{}(p.second); }
    };
}


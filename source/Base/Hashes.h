#pragma once

#include <bit>
#include <functional>
#include <tuple>

namespace std
{
    template <class A, class B>
    struct hash<pair<A, B>>
    {
        size_t operator()(pair<A, B> const& p) const { return std::rotl(hash<A>{}(p.first), 1) ^ hash<B>{}(p.second); }
    };
}

template <typename... Types>
struct variant_hasher
{
    std::size_t operator()(std::variant<Types...> const& v) const
    {
        return std::visit([](const auto& val) -> std::size_t { return std::hash<std::decay_t<decltype(val)>>{}(val); }, v);
    }
};

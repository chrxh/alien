#pragma once

#include <bit>
#include <functional>
#include <tuple>
#include <variant>

template <class A, class B>
struct std::hash<std::pair<A, B>>
{
    size_t operator()(std::pair<A, B> const& p) const { return std::rotl(std::hash<A>{}(p.first), 1) ^ std::hash<B>{}(p.second); }
};

template <typename T>
inline void hash_combine(std::size_t& seed, const T& val)
{
    seed ^= std::hash<T>{}(val) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

template <typename... Types>
struct variant_hasher
{
    std::size_t operator()(std::variant<Types...> const& v) const
    {
        return std::visit([](const auto& val) -> std::size_t { return std::hash<std::decay_t<decltype(val)>>{}(val); }, v);
    }
};

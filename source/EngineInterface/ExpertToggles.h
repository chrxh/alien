#pragma once

#include <compare>

/**
 * NOTE: header is also included in kernel code
 */

struct ExpertToggles
{
    bool operator==(ExpertToggles const&) const = default;
};

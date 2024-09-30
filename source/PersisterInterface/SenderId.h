#pragma once

#include <string>

struct SenderId
{
    bool operator==(SenderId const&) const = default;

    std::string value;
};

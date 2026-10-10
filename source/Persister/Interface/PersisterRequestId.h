#pragma once

#include <string>

struct PersisterRequestId
{
    bool operator==(PersisterRequestId const&) const = default;

    std::string value;
};

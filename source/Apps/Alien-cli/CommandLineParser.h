#pragma once

#include <optional>

#include "CommandLineArguments.h"

class CommandLineParser
{
public:
    static std::optional<int> parse(CommandLineArguments& arguments, int argc, char** argv);
};

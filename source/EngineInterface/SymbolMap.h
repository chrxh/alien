#pragma once

#include "Definitions.h"

using SymbolMap = std::map<std::string, std::string>;

class SymbolMapHelper
{
public:
    static SymbolMap getDefaultSymbolMap();
};

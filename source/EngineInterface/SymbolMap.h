#pragma once

#include "Definitions.h"
#include "DllExport.h"

using SymbolMap = std::map<std::string, std::string>;

class SymbolMapHelper
{
public:
    ENGINEINTERFACE_EXPORT static SymbolMap getDefaultSymbolMap();
};

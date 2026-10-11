#pragma once

#include <string>

#include <Network/Interface/NetworkResourceTreeTO.h>

#include "Definitions.h"

class BrowserHelper
{
public:
    static auto constexpr RowHeight = 25.0f;
    static float calcFooterHeight();
    static bool ActionButton(std::string const& text);
    static void DownloadButton(BrowserData const& data, BrowserLeaf const& leaf);
};

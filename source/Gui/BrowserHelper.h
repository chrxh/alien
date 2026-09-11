#pragma once

#include <string>

#include <Network/NetworkResourceTreeTO.h>

#include "Definitions.h"

class BrowserHelper
{
public:
    static auto constexpr RowHeight = 25.0f;
    static auto constexpr WorkspaceBottomSpace = 34.0f;

    static bool ActionButton(std::string const& text);
    static void DownloadButton(BrowserData const& data, BrowserLeaf const& leaf);
};

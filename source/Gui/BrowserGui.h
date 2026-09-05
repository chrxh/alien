#pragma once

#include <string>

#include <Network/NetworkResourceTreeTO.h>

#include "Definitions.h"

// Small GUI elements shared by the browser widgets
class BrowserGui
{
public:
    static auto constexpr RowHeight = 25.0f;
    static auto constexpr WorkspaceBottomSpace = 34.0f;

    static void ShortenedText(std::string const& text, bool bold = false);
    static bool ActionButton(std::string const& text);
    static bool DetailButton();
    static void DownloadButton(BrowserData const& data, BrowserLeaf const& leaf);
};

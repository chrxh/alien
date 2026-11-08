#pragma once

#include <filesystem>
#include <string>

class WebLinkHelper
{
public:
    static void openInBrowser(std::string const& url);
    static void openInFileManager(std::filesystem::path const& folder);
};

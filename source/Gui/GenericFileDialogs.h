#pragma once

#include <functional>
#include <filesystem>

#include "Base/Singleton.h"
#include "EngineInterface/Definitions.h"
#include "Definitions.h"

class GenericFileDialogs
{
    MAKE_SINGLETON(GenericFileDialogs);
public:
    void showOpenFileDialog(
        std::string const& title,
        std::string const& filter,
        std::string startingPath,
        std::function<void(std::filesystem::path const&)> const& actionFunc);

    void showSaveFileDialog(
        std::string const& title,
        std::string const& filter,
        std::string startingPath,
        std::function<void(std::filesystem::path const&)> const& actionFunc);

    void process();

private:
    std::function<void(std::filesystem::path)> _actionFunc;
};

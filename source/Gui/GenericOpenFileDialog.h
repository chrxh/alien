#pragma once

#include <functional>
#include <filesystem>

#include "EngineInterface/Definitions.h"
#include "Definitions.h"

class GenericOpenFileDialog
{
public:
    static GenericOpenFileDialog& getInstance();

    void process();

    void show(std::string const& title, std::string const& filter, std::string startingPath, std::function<void(std::filesystem::path const&)> const& actionFunc);

private:
    std::function<void(std::filesystem::path)> _actionFunc;
};

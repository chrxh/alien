#pragma once

#include <functional>
#include <filesystem>

#include "Base/Singleton.h"
#include "EngineInterface/Definitions.h"
#include "Definitions.h"
#include "MainLoopEntity.h"

class GenericFileDialog : public MainLoopEntity<>
{
    MAKE_SINGLETON(GenericFileDialog);
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

private:
    void init() override {}
    void process() override;
    void shutdown() override {}

    std::function<void(std::filesystem::path)> _actionFunc;
};

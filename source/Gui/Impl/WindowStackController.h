#pragma once

#include <optional>
#include <string>
#include <unordered_set>
#include <vector>

#include <Base/Singleton.h>

#include "MainLoopEntity.h"

class WindowStackController : public MainLoopEntity
{
    MAKE_SINGLETON(WindowStackController);

public:
    void registerWindow(std::string const& title);

private:
    void init() override;
    void process() override;
    void shutdown() override;

    void restoreOrder(std::vector<std::string> const& order);

    std::unordered_set<std::string> _titles;
    std::optional<std::vector<std::string>> _orderToRestore;
    bool _isFirstFrameProcessed = false;
};

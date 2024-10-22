#pragma once

#include <chrono>

#include "Base/Singleton.h"

#include "Definitions.h"
#include "MainLoopEntity.h"

class UiController : public MainLoopEntity
{
    MAKE_SINGLETON(UiController);

public:
    void init();

    bool isOn() const;
    void setOn(bool value);

private:
    void process() override;
    void shutdown() override {}

    bool _on = true;
    std::optional<std::chrono::steady_clock::time_point> _lastChangeTimePoint;
};
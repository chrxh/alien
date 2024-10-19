#pragma once

#include <chrono>

#include "Base/Singleton.h"

#include "Definitions.h"

class UiController
{
    MAKE_SINGLETON(UiController);

public:
    bool isOn() const;
    void setOn(bool value);

    void process();

private:
    bool _on = true;
    std::optional<std::chrono::steady_clock::time_point> _lastChangeTimePoint;
};
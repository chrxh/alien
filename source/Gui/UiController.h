#pragma once

#include <chrono>

#include "Definitions.h"

class _UiController
{
public:
    bool isOn() const;
    void setOn(bool value);

    void process();

private:
    bool _on = true;
    boost::optional<std::chrono::steady_clock::time_point> _lastChangeTimePoint;
};
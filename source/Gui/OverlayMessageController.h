#pragma once

#include <chrono>

#include "Definitions.h"
#include "EngineInterface/Definitions.h"

class OverlayMessageController
{
public:
    static OverlayMessageController& getInstance();

    void process();

    void show(std::string const& message, bool withLightning = false);

    void setOn(bool value);

private:
    bool _show = false;
    bool _withLightning = false;
    bool _on = true;
    std::string _message;
    int _counter = 0;

    std::optional<std::chrono::steady_clock::time_point> _startTimePoint;
    std::optional<std::chrono::steady_clock::time_point> _ticksLaterTimePoint;
};

inline void printOverlayMessage(std::string const& message, bool withLightning = false)
{
    OverlayMessageController::getInstance().show(message, withLightning);
}
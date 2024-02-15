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

private:
    bool _show = false;
    bool _withLightning = false;
    std::string _message;

    std::optional<std::chrono::steady_clock::time_point> _startTimePoint;
};

inline void printOverlayMessage(std::string const& message, bool withLightning = false)
{
    OverlayMessageController::getInstance().show(message, withLightning);
}
#pragma once

#include <chrono>

#include "Definitions.h"
#include "EngineInterface/Definitions.h"
#include "PersisterInterface/PersisterController.h"

class OverlayMessageController
{
public:
    static OverlayMessageController& getInstance();

    void init(PersisterController const& persisterController);
    void process();

    void show(std::string const& message, bool withLightning = false);

    void setOn(bool value);

private:
    void processSpinner();
    void processMessage();

    PersisterController _persisterController;

    bool _show = false;
    bool _withLightning = false;
    bool _on = true;
    std::string _message;
    int _counter = 0;

    std::optional<std::chrono::steady_clock::time_point> _spinnerRefTimepoint;
    float _spinnerAngle = 0;
    std::optional<std::chrono::steady_clock::time_point> _startTimepoint;
    std::optional<std::chrono::steady_clock::time_point> _ticksLaterTimepoint;
};

inline void printOverlayMessage(std::string const& message, bool withLightning = false)
{
    OverlayMessageController::getInstance().show(message, withLightning);
}
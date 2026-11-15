#pragma once

#include <chrono>
#include <optional>
#include <string>

#include <Persister/Interface/PersisterFacade.h>

#include "Definitions.h"

class OverlayController
{
    MAKE_SINGLETON(OverlayController);

public:
    void setup();
    void process();

    void showMessage(std::string const& message, bool withLightning = false);

    // The status stays in place of the messages until it is reset. Messages that come in between are shown instead of it.
    void setStatus(std::optional<std::string> const& status);

    bool isOn() const;
    void setOn(bool value);

private:
    void processProgressAnimation();
    void advanceHelixRotation();
    void processMessage();
    void drawMessage(std::string const& message, float alpha);

    bool _show = false;
    bool _withLightning = false;
    bool _on = true;
    std::string _message;
    std::optional<std::string> _status;
    int _counter = 0;

    float _helixPhase = 0.0f;
    float _helixStartSpeedExponent = 0.0f;
    float _helixTargetSpeedExponent = 0.0f;
    float _helixSpeedChangeTime = 0.0f;
    float _helixSpeedChangeDuration = 0.0f;

    std::optional<std::chrono::steady_clock::time_point> _messageStartTimepoint;
    std::optional<std::chrono::steady_clock::time_point> _ticksLaterTimepoint;
    std::optional<std::chrono::steady_clock::time_point> _busyTimepoint;
};

inline void printOverlayMessage(std::string const& message, bool withLightning = false)
{
    OverlayController::get().showMessage(message, withLightning);
}

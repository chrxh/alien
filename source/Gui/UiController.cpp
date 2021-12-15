#include "UiController.h"

#include <imgui.h>

#include "Base/Definitions.h"

namespace
{
    constexpr std::chrono::milliseconds::rep FadeInOutDuration = 1000;
}

bool _UiController::isOn() const
{
    return _on;
}

void _UiController::setOn(bool value)
{
    if (!_lastChangeTimePoint) {
        _lastChangeTimePoint = std::chrono::steady_clock::now();
    } else {
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                            std::chrono::steady_clock::now() - *_lastChangeTimePoint)
                            .count();

        _lastChangeTimePoint = std::chrono::steady_clock::now() - std::chrono::milliseconds(FadeInOutDuration - duration);
    }
    _on = value;
}

void _UiController::process()
{
    if (_lastChangeTimePoint) {
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                            std::chrono::steady_clock::now() - *_lastChangeTimePoint)
                            .count();
        if (_on) {
            ImGui::GetStyle().Alpha = toFloat(std::min(duration, FadeInOutDuration)) / FadeInOutDuration;
            if (ImGui::GetStyle().Alpha == 1) {
                _lastChangeTimePoint = boost::none;
            }
        }
        else {
            ImGui::GetStyle().Alpha =
                toFloat(std::max(FadeInOutDuration - duration, std::chrono::milliseconds::rep(0))) / FadeInOutDuration;
            if (ImGui::GetStyle().Alpha == 0) {
                _lastChangeTimePoint = boost::none;
            }
        }
    }
}

#include "WindowStackController.h"

#include <algorithm>

#include <imgui.h>
#include <imgui_internal.h>

#include <Base/GlobalSettings.h>

namespace
{
    auto constexpr OrderSettingsKey = "controllers.window stack.order";
}

void WindowStackController::registerWindow(std::string const& title)
{
    _titles.insert(title);
}

void WindowStackController::init()
{
    _orderToRestore = GlobalSettings::get().getValue(OrderSettingsKey, std::vector<std::string>());
}

void WindowStackController::process()
{
    if (!_orderToRestore.has_value()) {
        return;
    }
    if (!_isFirstFrameProcessed) {
        _isFirstFrameProcessed = true;
        return;
    }
    restoreOrder(*_orderToRestore);
    _orderToRestore.reset();
}

namespace
{
    std::vector<ImGuiWindow*> getShownWindowsFromBackToFront(std::unordered_set<std::string> const& titles)
    {
        std::vector<ImGuiWindow*> result;
        for (auto const& window : ImGui::GetCurrentContext()->Windows) {
            if ((window->Active || window->WasActive) && titles.contains(window->Name)) {
                result.emplace_back(window);
            }
        }
        return result;
    }
}

void WindowStackController::shutdown()
{
    std::vector<std::string> order;
    for (auto const& window : getShownWindowsFromBackToFront(_titles)) {
        order.emplace_back(window->Name);
    }
    GlobalSettings::get().setValue(OrderSettingsKey, order);
}

void WindowStackController::restoreOrder(std::vector<std::string> const& order)
{
    auto windows = getShownWindowsFromBackToFront(_titles);
    std::ranges::stable_sort(windows, {}, [&](ImGuiWindow* window) { return std::ranges::find(order, window->Name) - order.begin(); });
    for (auto const& window : windows) {
        ImGui::FocusWindow(window, ImGuiFocusRequestFlags_UnlessBelowModal);
    }
}

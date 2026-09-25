#include "McpSettingsDialog.h"

#include <imgui.h>

#include "AlienGui.h"
#include "McpController.h"
#include "StyleService.h"

namespace
{
    auto constexpr RightColumnWidth = 150.0f;
}

McpSettingsDialog::McpSettingsDialog()
    : AlienDialog("MCP server settings")
{}

void McpSettingsDialog::processIntern()
{
    auto& controller = McpController::get();
    auto running = controller.isServerRunning();

    ImGui::BeginDisabled(running);
    AlienGui::InputInt(
        AlienGui::InputIntParameters()
            .name("Port")
            .textWidth(RightColumnWidth)
            .defaultValue(controller.getDefaultPort())
            .tooltip(running ? "Stop the server to change the port." : "Local port on which the MCP server accepts connections."),
        _port);
    ImGui::EndDisabled();

    ImGui::Dummy({0, ImGui::GetContentRegionAvail().y - scale(50.0f)});
    AlienGui::Separator();

    ImGui::BeginDisabled(running);
    if (AlienGui::Button("Adopt")) {
        close();
        controller.setPort(_port);
    }
    ImGui::EndDisabled();
    ImGui::SetItemDefaultFocus();

    ImGui::SameLine();
    if (AlienGui::Button("Cancel")) {
        close();
    }
}

void McpSettingsDialog::openIntern()
{
    _port = McpController::get().getPort();
}

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

    AlienGui::InputInt(
        AlienGui::InputIntParameters()
            .name("Port")
            .textWidth(RightColumnWidth)
            .defaultValue(controller.getDefaultPort())
            .tooltip("Local port on which the MCP server accepts connections. A running server is restarted on the new port."),
        _port);

    ImGui::Dummy({0, ImGui::GetContentRegionAvail().y - scale(50.0f)});
    AlienGui::Separator();

    if (AlienGui::Button("Adopt")) {
        close();
        controller.setPort(_port);
    }
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

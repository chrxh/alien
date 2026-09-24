#include "McpSettingsDialog.h"

#include <imgui.h>

#include <Network/McpService.h>

#include "AlienGui.h"
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
    auto& service = McpService::get();
    auto running = service.isServerRunning();

    ImGui::BeginDisabled(running);
    AlienGui::InputInt(
        AlienGui::InputIntParameters()
            .name("Port")
            .textWidth(RightColumnWidth)
            .defaultValue(service.getDefaultPort())
            .tooltip(running ? "Stop the server to change the port." : "Local port on which the MCP server accepts connections."),
        _port);
    ImGui::EndDisabled();

    ImGui::Dummy({0, ImGui::GetContentRegionAvail().y - scale(50.0f)});
    AlienGui::Separator();

    ImGui::BeginDisabled(running);
    if (AlienGui::Button("Adopt")) {
        close();
        service.setPort(_port);
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
    _port = McpService::get().getPort();
}

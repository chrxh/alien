#include "BrowserWindow.h"

#include <imgui.h>
#include <boost/property_tree/json_parser.hpp>

#define CPPHTTPLIB_OPENSSL_SUPPORT
#include <httplib.h>

#include "AlienImGui.h"
#include "GlobalSettings.h"
#include "StyleRepository.h"
#include "RemoteSimulationDataParser.h"

_BrowserWindow::_BrowserWindow(SimulationController const& simController)
    : _AlienWindow("Browser", "browser.network", false)
    , _simController(simController)
{
    if (_on) {
        processActivated();
    }
}

_BrowserWindow::~_BrowserWindow()
{
}

void _BrowserWindow::processIntern()
{
    auto styleRepository = StyleRepository::getInstance();
    static ImGuiTableFlags flags = ImGuiTableFlags_Resizable | ImGuiTableFlags_Reorderable | ImGuiTableFlags_Hideable | ImGuiTableFlags_Sortable
        | ImGuiTableFlags_SortMulti | ImGuiTableFlags_RowBg | ImGuiTableFlags_BordersOuter | ImGuiTableFlags_BordersV | ImGuiTableFlags_NoBordersInBody
        | ImGuiTableFlags_ScrollY;
    if (ImGui::BeginTable("table_sorting", 8, flags, ImVec2(0, ImGui::GetContentRegionAvail().y - styleRepository.scaleContent(40.0f)), 0.0f)) {
        ImGui::TableSetupColumn("Timestamp", ImGuiTableColumnFlags_DefaultSort | ImGuiTableColumnFlags_WidthFixed, 0.0f, RemoteSimulationDataColumnId_UserName);
        ImGui::TableSetupColumn("User name", ImGuiTableColumnFlags_DefaultSort | ImGuiTableColumnFlags_WidthFixed, 0.0f, RemoteSimulationDataColumnId_UserName);
        ImGui::TableSetupColumn(
            "Simulation name", ImGuiTableColumnFlags_DefaultSort | ImGuiTableColumnFlags_WidthFixed, 0.0f, RemoteSimulationDataColumnId_SimulationName);
        ImGui::TableSetupColumn("Likes", ImGuiTableColumnFlags_DefaultSort | ImGuiTableColumnFlags_WidthFixed, 0.0f, RemoteSimulationDataColumnId_UserName);
        ImGui::TableSetupColumn("Width", ImGuiTableColumnFlags_DefaultSort | ImGuiTableColumnFlags_WidthFixed, 0.0f, RemoteSimulationDataColumnId_Width);
        ImGui::TableSetupColumn("Height", ImGuiTableColumnFlags_DefaultSort | ImGuiTableColumnFlags_WidthStretch, 0.0f, RemoteSimulationDataColumnId_Height);
        ImGui::TableSetupColumn("Version", ImGuiTableColumnFlags_DefaultSort | ImGuiTableColumnFlags_WidthStretch, 0.0f, RemoteSimulationDataColumnId_Height);
        ImGui::TableSetupColumn(
            "Actions", ImGuiTableColumnFlags_PreferSortDescending | ImGuiTableColumnFlags_WidthStretch, 0.0f, RemoteSimulationDataColumnId_Height);
        ImGui::TableSetupScrollFreeze(0, 1);
        ImGui::TableHeadersRow();

        //sort our data if sort specs have been changed!
        if (ImGuiTableSortSpecs* sortSpecs = ImGui::TableGetSortSpecs())
            if (sortSpecs->SpecsDirty) {
                if (_remoteSimulationDatas.size() > 1) {
                    std::sort(_remoteSimulationDatas.begin(), _remoteSimulationDatas.end(), [&](auto const& left, auto const& right) {
                        return RemoteSimulationData::compare(&left, &right, sortSpecs) < 0;
                    });
                }
                sortSpecs->SpecsDirty = false;
            }

        ImGuiListClipper clipper;
        clipper.Begin(_remoteSimulationDatas.size());
        while (clipper.Step())
            for (int row_n = clipper.DisplayStart; row_n < clipper.DisplayEnd; row_n++) {
                RemoteSimulationData* item = &_remoteSimulationDatas[row_n];
                ImGui::PushID(row_n);
                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                AlienImGui::Text(item->timestamp);
                ImGui::TableNextColumn();
                AlienImGui::Text(item->userName);
                ImGui::TableNextColumn();
                AlienImGui::Text(item->simName);
                ImGui::TableNextColumn();
                ImGui::TableNextColumn();
                AlienImGui::Text(std::to_string(item->width));
                ImGui::TableNextColumn();
                AlienImGui::Text(std::to_string(item->height));
                ImGui::PopID();
            }
        ImGui::EndTable();
    }
    AlienImGui::InputText(AlienImGui::InputTextParameters().name("Filter"), _filter);
}

void _BrowserWindow::processActivated()
{
    _server = GlobalSettings::getInstance().getStringState("settings.server", "alien-project.org");

    httplib::SSLClient cli(_server);
    cli.set_ca_cert_path("./resources/ca-bundle.crt");
    cli.enable_server_certificate_verification(true);

    auto result = cli.get_openssl_verify_result();
/*
    if (result) {
        std::cout << "verify error: " << X509_verify_cert_error_string(result) << std::endl;
    }
*/

    if (auto res = cli.Get("/world-explorer/api/getsimulationinfos.php")) {
        std::stringstream stream(res->body);
        boost::property_tree::ptree tree;
        boost::property_tree::read_json(stream, tree);

        _remoteSimulationDatas = RemoteSimulationDataParser::decode(tree);
 
        _remoteSimulationDatas.resize(30, RemoteSimulationData());
        for (int n = 0; n < _remoteSimulationDatas.size(); n++) {
            RemoteSimulationData& item = _remoteSimulationDatas[n];
            item.userName = "chrxh";
            item.simName = "evo";
            item.width = 1000 * n;
            item.height = 500 * n;
            item.description = "test sim";
        }

    }
}

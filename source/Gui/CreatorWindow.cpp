#include "CreatorWindow.h"

#include <cstdlib>
#include <imgui.h>

#include "Fonts/IconsFontAwesome5.h"
#include "Fonts/AlienIconFont.h"

#include "EngineInterface/Descriptions.h"
#include "EngineInterface/SimulationController.h"

#include "GlobalSettings.h"
#include "StyleRepository.h"
#include "AlienImGui.h"
#include "Viewport.h"

namespace
{
    auto const ModeText = std::unordered_map<CreationMode, std::string>{
        {CreationMode::CreateParticle, "Create single particle"},
        {CreationMode::CreateCell, "Create single cell"},
        {CreationMode::CreateRect, "Create rectangular cell cluster"},
        {CreationMode::CreateHexagon, "Create hexagonal cell cluster"},
        {CreationMode::CreateDisc, "Create disc-shaped cell cluster"}};
}

_CreatorWindow::_CreatorWindow(SimulationController const& simController, Viewport const& viewport)
    : _simController(simController)
    , _viewport(viewport)
{
    _on = GlobalSettings::getInstance().getBoolState("windows.creator.active", true);
}

_CreatorWindow::~_CreatorWindow()
{
    GlobalSettings::getInstance().setBoolState("windows.creator.active", _on);
}

void _CreatorWindow::process()
{
    if (!_on) {
        return;
    }
    ImGui::SetNextWindowBgAlpha(Const::WindowAlpha * ImGui::GetStyle().Alpha);
    if (ImGui::Begin("Creator", &_on)) {
        if (AlienImGui::BeginToolbarButton(ICON_FA_SUN)) {
            _mode = CreationMode::CreateParticle;
        }
        AlienImGui::EndToolbarButton();

        ImGui::SameLine();
        if (AlienImGui::BeginToolbarButton(ICON_FA_ATOM)) {
            _mode = CreationMode::CreateCell;
        }
        AlienImGui::EndToolbarButton();

        ImGui::SameLine();
        if (AlienImGui::BeginToolbarButton(ICON_RECTANGLE)) {
            _mode = CreationMode::CreateRect;
        }
        AlienImGui::EndToolbarButton();

        ImGui::SameLine();
        if (AlienImGui::BeginToolbarButton(ICON_HEXAGON)) {
            _mode = CreationMode::CreateHexagon;
        }
        AlienImGui::EndToolbarButton();

        ImGui::SameLine();
        if (AlienImGui::BeginToolbarButton(ICON_DISC)) {
            _mode = CreationMode::CreateDisc;
        }
        AlienImGui::EndToolbarButton();

        AlienImGui::Group(ModeText.at(_mode));
        AlienImGui::InputFloat(AlienImGui::InputFloatParameters().name("Energy").format("%.2f"), _energy);

        if (_mode == CreationMode::CreateRect || _mode == CreationMode::CreateHexagon || _mode == CreationMode::CreateDisc) {
            AlienImGui::InputFloat(AlienImGui::InputFloatParameters().name("Cell distance").format("%.2f").step(0.1), _distance);
        }

        if (ImGui::Button("Build")) {
            if (_mode == CreationMode::CreateCell) {
                createCell();
            }
        }
    }
    ImGui::End();
}

bool _CreatorWindow::isOn() const
{
    return _on;
}

void _CreatorWindow::setOn(bool value)
{
    _on = value;
}

void _CreatorWindow::createCell()
{
    auto pos = _viewport->getCenterInWorldPos();
    pos.x += (toFloat(std::rand()) / RAND_MAX - 0.5f) * 8;
    pos.y += (toFloat(std::rand()) / RAND_MAX - 0.5f) * 8;
    auto cell = CellDescription().setPos(pos).setEnergy(_energy);
    auto cluster = ClusterDescription().addCell(cell);
    auto data = DataDescription().addCluster(cluster);
    _simController->addAndSelectSimulationData(data);
}

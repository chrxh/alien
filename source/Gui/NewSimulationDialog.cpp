#include "NewSimulationDialog.h"

#include <imgui.h>

#include <Base/GlobalSettings.h>
#include <Base/StringHelper.h>

#include <EngineInterface/NameGeneratorService.h>
#include <EngineInterface/SimulationFacade.h>

#include "AlienGui.h"
#include "NewSimulationService.h"
#include "StyleService.h"
#include <EngineInterface/SimulationFacade.h>

namespace
{
    auto const ContentTextInputWidth = 130.0f;
    auto const ProjectNameSize = sizeof(Char64) / sizeof(char);
}

void NewSimulationDialog::initIntern()
{

    _adoptSimulationParameters = GlobalSettings::get().getValue("dialogs.new simulation.adopt simulation parameters", true);
}

void NewSimulationDialog::shutdownIntern()
{
    GlobalSettings::get().setValue("dialogs.new simulation.adopt simulation parameters", _adoptSimulationParameters);
}

NewSimulationDialog::NewSimulationDialog()
    : AlienDialog("New simulation")
{}

void NewSimulationDialog::processIntern()
{
    AlienGui::InputText(
        AlienGui::InputTextParameters()
            .name("Project name")
            .textWidth(ContentTextInputWidth)
            .generateValueFunc([] { return NameGeneratorService::get().createSimulationName(); }),
        _projectName,
        ProjectNameSize);
    AlienGui::InputInt(AlienGui::InputIntParameters().name("Width").textWidth(ContentTextInputWidth), _width);
    AlienGui::InputInt(AlienGui::InputIntParameters().name("Height").textWidth(ContentTextInputWidth), _height);
    AlienGui::InputFloat(AlienGui::InputFloatParameters().name("Energy").textWidth(ContentTextInputWidth).format("%.0f").step(1000.0f), _externalEnergy);
    AlienGui::Checkbox(AlienGui::CheckboxParameters().name("Adopt parameters").textWidth(ContentTextInputWidth), &_adoptSimulationParameters);

    ImGui::Dummy({0, ImGui::GetContentRegionAvail().y - scale(50.0f)});
    AlienGui::Separator();
    if (AlienGui::Button("OK")) {
        ImGui::CloseCurrentPopup();
        onNewSimulation();
        close();
    }
    ImGui::SetItemDefaultFocus();

    ImGui::SameLine();
    if (AlienGui::Button("Cancel")) {
        ImGui::CloseCurrentPopup();
        close();
    }

    _width = std::max(1, _width);
    _height = std::max(1, _height);
    _externalEnergy = std::max(0.0f, _externalEnergy);
}

void NewSimulationDialog::openIntern()
{
    StringHelper::copy(_projectName, ProjectNameSize, NameGeneratorService::get().createSimulationName());

    auto worldSize = _SimulationFacade::get()->getWorldSize();
    _width = worldSize.x;
    _height = worldSize.y;
    _externalEnergy = _SimulationFacade::get()->getSimulationParameters().externalEnergy.value;
}

void NewSimulationDialog::onNewSimulation()
{
    NewSimulationService::get().createSimulation(NewSimulationService::Parameters()
                                                     .projectName(_projectName)
                                                     .worldSize({_width, _height})
                                                     .externalEnergy(_externalEnergy)
                                                     .adoptSimulationParameters(_adoptSimulationParameters));
}

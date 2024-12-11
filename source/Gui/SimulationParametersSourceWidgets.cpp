#include "SimulationParametersSourceWidgets.h"

#include <imgui.h>

#include "AlienImGui.h"
#include "EngineInterface/SimulationParametersEditService.h"
#include "EngineInterface/SimulationFacade.h"
#include "EngineInterface/SimulationParametersValidationService.h"
#include "LocationHelper.h"
#include "SimulationInteractionController.h"

namespace
{
    auto constexpr RightColumnWidth = 285.0f;
}

void _SimulationParametersSourceWidgets::init(SimulationFacade const& simulationFacade, int locationIndex)
{
    _simulationFacade = simulationFacade;
    _locationIndex = locationIndex;
}

void _SimulationParametersSourceWidgets::process()
{
    auto& editService = SimulationParametersEditService::get();

    auto parameters = _simulationFacade->getSimulationParameters();
    auto origParameters = _simulationFacade->getOriginalSimulationParameters();

    auto worldSize = _simulationFacade->getWorldSize();
    auto sourceIndex = LocationHelper::findLocationArrayIndex(parameters, _locationIndex);

    RadiationSource& source = parameters.radiationSource[sourceIndex];
    auto lastSource = source;
    RadiationSource& origSource = origParameters.radiationSource[sourceIndex];
    _sourceName = std::string(source.name);

    /**
     * General
     */
    if (AlienImGui::BeginTreeNode(AlienImGui::TreeNodeParameters().text("General"))) {
        AlienImGui::InputText(
            AlienImGui::InputTextParameters().name("Name").textWidth(RightColumnWidth).defaultValue(origSource.name),
            source.name,
            sizeof(Char64) / sizeof(char));

        AlienImGui::EndTreeNode();
    }

    /**
     * Location
     */
    if (AlienImGui::BeginTreeNode(AlienImGui::TreeNodeParameters().text("Location"))) {
        if (AlienImGui::Switcher(
                AlienImGui::SwitcherParameters()
                    .name("Shape")
                    .values({"Circular", "Rectangular"})
                    .textWidth(RightColumnWidth)
                    .defaultValue(origSource.shapeType),
                source.shapeType)) {
            if (source.shapeType == RadiationSourceShapeType_Circular) {
                source.shapeData.circularRadiationSource.radius = 1;
            } else {
                source.shapeData.rectangularRadiationSource.width = 40;
                source.shapeData.rectangularRadiationSource.height = 10;
            }
        }

        auto getMousePickerEnabledFunc = [&]() { return SimulationInteractionController::get().isPositionSelectionMode(); };
        auto setMousePickerEnabledFunc = [&](bool value) { SimulationInteractionController::get().setPositionSelectionMode(value); };
        auto getMousePickerPositionFunc = [&]() { return SimulationInteractionController::get().getPositionSelectionData(); };
        AlienImGui::SliderFloat2(
            AlienImGui::SliderFloat2Parameters()
                .name("Position (x,y)")
                .textWidth(RightColumnWidth)
                .min({0, 0})
                .max(toRealVector2D(worldSize))
                .defaultValue(RealVector2D{origSource.posX, origSource.posY})
                .format("%.2f")
                .getMousePickerEnabledFunc(getMousePickerEnabledFunc)
                .setMousePickerEnabledFunc(setMousePickerEnabledFunc)
                .getMousePickerPositionFunc(getMousePickerPositionFunc),
            source.posX,
            source.posY);
        AlienImGui::SliderFloat2(
            AlienImGui::SliderFloat2Parameters()
                .name("Velocity (x,y)")
                .textWidth(RightColumnWidth)
                .min({-4.0f, -4.0f})
                .max({4.0f, 4.0f})
                .defaultValue(RealVector2D{origSource.velX, origSource.velY})
                .format("%.2f"),
            source.velX,
            source.velY);
        if (source.shapeType == RadiationSourceShapeType_Circular) {
            auto maxRadius = toFloat(std::min(worldSize.x, worldSize.y));
            AlienImGui::SliderFloat(
                AlienImGui::SliderFloatParameters()
                    .name("Radius")
                    .textWidth(RightColumnWidth)
                    .min(1)
                    .max(maxRadius)
                    .format("%.0f")
                    .defaultValue(&origSource.shapeData.circularRadiationSource.radius),
                &source.shapeData.circularRadiationSource.radius);
        }
        if (source.shapeType == RadiationSourceShapeType_Rectangular) {
            AlienImGui::SliderFloat2(
                AlienImGui::SliderFloat2Parameters()
                    .name("Size (x,y)")
                    .textWidth(RightColumnWidth)
                    .min({0, 0})
                    .max({toFloat(worldSize.x), toFloat(worldSize.y)})
                    .defaultValue(RealVector2D{origSource.shapeData.rectangularRadiationSource.height, origSource.shapeData.rectangularRadiationSource.height})
                    .format("%.1f"),
                source.shapeData.rectangularRadiationSource.width,
                source.shapeData.rectangularRadiationSource.height);
        }
        AlienImGui::EndTreeNode();
    }

    /**
     * Radiation
     */
    if (AlienImGui::BeginTreeNode(AlienImGui::TreeNodeParameters().text("Radiation"))) {
        auto origStrengths = editService.getRadiationStrengths(parameters);
        if (AlienImGui::SliderFloat(
                AlienImGui::SliderFloatParameters()
                    .name("Relative strength")
                    .textWidth(RightColumnWidth)
                    .min(0.0f)
                    .max(1.0f)
                    .format("%.3f")
                    .defaultValue(&origSource.strength)
                    .tooltip("Cells can emit energy particles over time. A portion of this energy can be released directly near the cell, while the rest is "
                             "utilized by one of the available radiation sources. This parameter determines the fraction of energy assigned to the emitted "
                             "energy particle for the selected radiation source. Values between 0 and 1 are permitted."),
                &source.strength,
                nullptr,
                &source.strengthPinned)) {
            auto editedStrengths = origStrengths;
            editedStrengths.values.at(sourceIndex + 1) = source.strength;  // Set new strength
            editService.adaptRadiationStrengths(editedStrengths, origStrengths, sourceIndex + 1);
            editService.applyRadiationStrengths(parameters, editedStrengths);
        }

        AlienImGui::SliderFloat(
            AlienImGui::SliderFloatParameters()
                .name("Radiation angle")
                .textWidth(RightColumnWidth)
                .min(-180.0f)
                .max(180.0f)
                .defaultEnabledValue(&origSource.useAngle)
                .defaultValue(&origSource.angle)
                .disabledValue(&source.angle)
                .format("%.1f"),
            &source.angle,
            &source.useAngle);

        AlienImGui::EndTreeNode();
    }

    SimulationParametersValidationService::get().validateAndCorrect(source);

    if (source != lastSource) {
        auto isRunning = _simulationFacade->isSimulationRunning();
        _simulationFacade->setSimulationParameters(
            parameters, isRunning ? SimulationParametersUpdateConfig::AllExceptChangingPositions : SimulationParametersUpdateConfig::All);
    }
}

std::string _SimulationParametersSourceWidgets::getLocationName()
{
    return "Simulation parameters for '" + _sourceName + "'";
}

int _SimulationParametersSourceWidgets::getLocationIndex() const
{
    return _locationIndex;
}

void _SimulationParametersSourceWidgets::setLocationIndex(int locationIndex)
{
    _locationIndex = locationIndex;
}

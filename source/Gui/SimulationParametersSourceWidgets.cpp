#include "SimulationParametersSourceWidgets.h"

#include <imgui.h>

#include "EngineInterface/SimulationFacade.h"
#include "EngineInterface/ParametersValidationService.h"
#include "EngineInterface/LocationHelper.h"

#include "SimulationInteractionController.h"
#include "SpecificationGuiService.h"

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
    auto parameters = _simulationFacade->getSimulationParameters();
    auto origParameters = _simulationFacade->getOriginalSimulationParameters();
    auto lastParameters = parameters;

    auto sourceIndex = LocationHelper::findLocationArrayIndex(parameters, _locationIndex);

    _sourceName = std::string(parameters.sourceName.sourceValues[sourceIndex]);

    SpecificationGuiService::get().createWidgetsForParameters(parameters, origParameters, _simulationFacade, _locationIndex);

    ///**
    // * General
    // */
    //if (AlienImGui::BeginTreeNode(AlienImGui::TreeNodeParameters().name("General"))) {
    //    AlienImGui::InputText(
    //        AlienImGui::InputTextParameters().name("Name").textWidth(RightColumnWidth).defaultValue(origSource.name),
    //        source.name,
    //        sizeof(Char64) / sizeof(char));

    //}
    //AlienImGui::EndTreeNode();

    ///**
    // * Location
    // */
    //if (AlienImGui::BeginTreeNode(AlienImGui::TreeNodeParameters().name("Location"))) {
    //    if (AlienImGui::Switcher(
    //            AlienImGui::SwitcherParameters()
    //                .name("Shape")
    //                .values({"Circular", "Rectangular"})
    //                .textWidth(RightColumnWidth)
    //                .defaultValue(origSource.shape.type),
    //            source.shape.type)) {
    //        if (source.shape.type == RadiationSourceShapeType_Circular) {
    //            source.shape.alternatives.circularRadiationSource.radius = 1;
    //        } else {
    //            source.shape.alternatives.rectangularRadiationSource.width = 40;
    //            source.shape.alternatives.rectangularRadiationSource.height = 10;
    //        }
    //    }

    //    auto getMousePickerEnabledFunc = [&]() { return SimulationInteractionController::get().isPositionSelectionMode(); };
    //    auto setMousePickerEnabledFunc = [&](bool value) { SimulationInteractionController::get().setPositionSelectionMode(value); };
    //    auto getMousePickerPositionFunc = [&]() { return SimulationInteractionController::get().getPositionSelectionData(); };
    //    AlienImGui::SliderFloat2(
    //        AlienImGui::SliderFloat2Parameters()
    //            .name("Position (x,y)")
    //            .textWidth(RightColumnWidth)
    //            .min({0, 0})
    //            .max(toRealVector2D(worldSize))
    //            .defaultValue(RealVector2D{origSource.posX, origSource.posY})
    //            .format("%.2f")
    //            .getMousePickerEnabledFunc(getMousePickerEnabledFunc)
    //            .setMousePickerEnabledFunc(setMousePickerEnabledFunc)
    //            .getMousePickerPositionFunc(getMousePickerPositionFunc),
    //        source.posX,
    //        source.posY);
    //    AlienImGui::SliderFloat2(
    //        AlienImGui::SliderFloat2Parameters()
    //            .name("Velocity (x,y)")
    //            .textWidth(RightColumnWidth)
    //            .min({-4.0f, -4.0f})
    //            .max({4.0f, 4.0f})
    //            .defaultValue(RealVector2D{origSource.velX, origSource.velY})
    //            .format("%.2f"),
    //        source.velX,
    //        source.velY);
    //    if (source.shape.type == RadiationSourceShapeType_Circular) {
    //        auto maxRadius = toFloat(std::min(worldSize.x, worldSize.y));
    //        AlienImGui::SliderFloat(
    //            AlienImGui::SliderFloatParameters()
    //                .name("Radius")
    //                .textWidth(RightColumnWidth)
    //                .min(1)
    //                .max(maxRadius)
    //                .format("%.0f")
    //                .defaultValue(&origSource.shape.alternatives.circularRadiationSource.radius),
    //            &source.shape.alternatives.circularRadiationSource.radius);
    //    }
    //    if (source.shape.type == RadiationSourceShapeType_Rectangular) {
    //        AlienImGui::SliderFloat2(
    //            AlienImGui::SliderFloat2Parameters()
    //                .name("Size (x,y)")
    //                .textWidth(RightColumnWidth)
    //                .min({0, 0})
    //                .max({toFloat(worldSize.x), toFloat(worldSize.y)})
    //                .defaultValue(RealVector2D{
    //                    origSource.shape.alternatives.rectangularRadiationSource.height, origSource.shape.alternatives.rectangularRadiationSource.height})
    //                .format("%.1f"),
    //            source.shape.alternatives.rectangularRadiationSource.width,
    //            source.shape.alternatives.rectangularRadiationSource.height);
    //    }
    //}
    //AlienImGui::EndTreeNode();

    ///**
    // * Radiation
    // */
    //if (AlienImGui::BeginTreeNode(AlienImGui::TreeNodeParameters().name("Radiation"))) {
    //    auto origStrengths = editService.getRadiationStrengths(parameters);
    //    if (AlienImGui::SliderFloat(
    //            AlienImGui::SliderFloatParameters()
    //                .name("Relative strength")
    //                .textWidth(RightColumnWidth)
    //                .min(0.0f)
    //                .max(1.0f)
    //                .format("%.3f")
    //                .defaultValue(&origSource.strength)
    //                .tooltip("Cells can emit energy particles over time. A portion of this energy can be released directly near the cell, while the rest is "
    //                         "utilized by one of the available radiation sources. This parameter determines the fraction of energy assigned to the emitted "
    //                         "energy particle for the selected radiation source. Values between 0 and 1 are permitted."),
    //            &source.strength,
    //            nullptr,
    //            &source.strengthPinned)) {
    //        auto editedStrengths = origStrengths;
    //        editedStrengths.values.at(sourceIndex + 1) = source.strength;  // Set new strength
    //        editService.adaptRadiationStrengths(editedStrengths, origStrengths, sourceIndex + 1);
    //        editService.applyRadiationStrengths(parameters, editedStrengths);
    //    }

    //    AlienImGui::SliderFloat(
    //        AlienImGui::SliderFloatParameters()
    //            .name("Radiation angle")
    //            .textWidth(RightColumnWidth)
    //            .min(-180.0f)
    //            .max(180.0f)
    //            .defaultEnabledValue(&origSource.useAngle)
    //            .defaultValue(&origSource.angle)
    //            .disabledValue(&source.angle)
    //            .format("%.1f"),
    //        &source.angle,
    //        &source.useAngle);

    //}
    //AlienImGui::EndTreeNode();

    //ParametersValidationService::get().validateAndCorrect(source);

    if (parameters != lastParameters) {
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

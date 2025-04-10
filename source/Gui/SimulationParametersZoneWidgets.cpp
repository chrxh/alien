#include "SimulationParametersZoneWidgets.h"

#include "EngineInterface/LocationHelper.h"
#include "EngineInterface/SimulationFacade.h"
#include "EngineInterface/ParametersValidationService.h"
#include "EngineInterface/SimulationParametersZone.h"

#include "AlienImGui.h"
#include "LoginDialog.h"
#include "SpecificationGuiService.h"
#include "SimulationInteractionController.h"

void _SimulationParametersZoneWidgets::init(SimulationFacade const& simulationFacade, int locationIndex)
{
    _simulationFacade = simulationFacade;
    _locationIndex = locationIndex;
}

void _SimulationParametersZoneWidgets::process()
{
    auto parameters = _simulationFacade->getSimulationParameters();
    auto origParameters = _simulationFacade->getOriginalSimulationParameters();
    auto lastParameters = parameters;

    auto zoneIndex = LocationHelper::findLocationArrayIndex(parameters, _locationIndex);
    SimulationParametersZone& zone = parameters.zone[zoneIndex];
    _zoneName = std::string(parameters.zoneNames.zoneValues[zoneIndex]);

    SpecificationGuiService::get().createWidgetsForParameters(parameters, origParameters, _simulationFacade, _locationIndex);

    ///**
    // * General
    // */
    //if (AlienImGui::BeginTreeNode(AlienImGui::TreeNodeParameters().name("General"))) {
    //    AlienImGui::InputText(
    //        AlienImGui::InputTextParameters().name("Name").textWidth(RightColumnWidth).defaultValue(origParameters.zoneNames.zoneValues[zoneIndex]),
    //        parameters.zoneNames.zoneValues[zoneIndex],
    //        sizeof(Char64) / sizeof(char));

    //}
    //AlienImGui::EndTreeNode();

    ///**
    // * Location
    // */
    //if (AlienImGui::BeginTreeNode(AlienImGui::TreeNodeParameters().name("Location"))) {
    //    if (AlienImGui::Switcher(
    //            AlienImGui::SwitcherParameters().name("Shape").values({"Circular", "Rectangular"}).textWidth(RightColumnWidth).defaultValue(origZone.shape.type),
    //            zone.shape.type)) {
    //        setDefaultSpotData(zone);
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
    //            .defaultValue(RealVector2D{origZone.posX, origZone.posY})
    //            .format("%.2f")
    //            .getMousePickerEnabledFunc(getMousePickerEnabledFunc)
    //            .setMousePickerEnabledFunc(setMousePickerEnabledFunc)
    //            .getMousePickerPositionFunc(getMousePickerPositionFunc),
    //        zone.posX,
    //        zone.posY);
    //    AlienImGui::SliderFloat2(
    //        AlienImGui::SliderFloat2Parameters()
    //            .name("Velocity (x,y)")
    //            .textWidth(RightColumnWidth)
    //            .min({-4.0f, -4.0f})
    //            .max({4.0f, 4.0f})
    //            .defaultValue(RealVector2D{origZone.velX, origZone.velY})
    //            .format("%.2f"),
    //        zone.velX,
    //        zone.velY);
    //    auto maxRadius = toFloat(std::max(worldSize.x, worldSize.y));
    //    if (zone.shape.type == ZoneShapeType_Circular) {
    //        AlienImGui::SliderFloat(
    //            AlienImGui::SliderFloatParameters()
    //                .name("Core radius")
    //                .textWidth(RightColumnWidth)
    //                .min(0)
    //                .max(maxRadius)
    //                .defaultValue(&origZone.shape.alternatives.circularSpot.coreRadius)
    //                .format("%.1f"),
    //            &zone.shape.alternatives.circularSpot.coreRadius);
    //    }
    //    if (zone.shape.type == ZoneShapeType_Rectangular) {
    //        AlienImGui::SliderFloat2(
    //            AlienImGui::SliderFloat2Parameters()
    //                .name("Size (x,y)")
    //                .textWidth(RightColumnWidth)
    //                .min({0, 0})
    //                .max({toFloat(worldSize.x), toFloat(worldSize.y)})
    //                .defaultValue(RealVector2D{origZone.shape.alternatives.rectangularSpot.width, origZone.shape.alternatives.rectangularSpot.height})
    //                .format("%.1f"),
    //            zone.shape.alternatives.rectangularSpot.width,
    //            zone.shape.alternatives.rectangularSpot.height);
    //    }

    //    AlienImGui::SliderFloat(
    //        AlienImGui::SliderFloatParameters()
    //            .name("Fade-out radius")
    //            .textWidth(RightColumnWidth)
    //            .min(0)
    //            .max(maxRadius)
    //            .defaultValue(&origZone.fadeoutRadius)
    //            .format("%.1f"),
    //        &zone.fadeoutRadius);
    //}
    //AlienImGui::EndTreeNode();

    ///**
    // * Force field
    // */
    //if (AlienImGui::BeginTreeNode(AlienImGui::TreeNodeParameters().name("Force field"))) {
    //    {
    //        auto forceFieldTypeIntern = std::max(0, zone.flow.type - 1);  //FlowType_None should not be selectable in ComboBox
    //        auto origForceFieldTypeIntern = std::max(0, origZone.flow.type - 1);

    //        auto isEnabled = zone.flow.type != 0;
    //        auto origIsEnabled = origZone.flow.type != 0;
    //        if (AlienImGui::Combo(
    //                AlienImGui::ComboParameters()
    //                    .name("Field type")
    //                    .values({"Radial", "Central", "Linear"})
    //                    .textWidth(RightColumnWidth)
    //                    .defaultValue(origForceFieldTypeIntern)
    //                    .defaultEnabledValue(&origIsEnabled),
    //                forceFieldTypeIntern,
    //                &isEnabled)) {
    //            zone.flow.type = isEnabled ? forceFieldTypeIntern + 1 : FlowType_None;

    //            if (zone.flow.type == FlowType_Radial) {
    //                zone.flow.alternatives.radialFlow = RadialFlow();
    //            }
    //            if (zone.flow.type == FlowType_Central) {
    //                zone.flow.alternatives.centralFlow = CentralFlow();
    //            }
    //            if (zone.flow.type == FlowType_Linear) {
    //                zone.flow.alternatives.linearFlow = LinearFlow();
    //            }
    //        }
    //    }

    //    auto isForceFieldActive = zone.flow.type != FlowType_None;

    //    ImGui::BeginDisabled(!isForceFieldActive);
    //    auto posX = ImGui::GetCursorPos().x;
    //    if (zone.flow.type == FlowType_Radial) {
    //        ImGui::SetCursorPosX(posX);
    //        AlienImGui::Combo(
    //            AlienImGui::ComboParameters()
    //                .name("Orientation")
    //                .textWidth(RightColumnWidth)
    //                .defaultValue(origZone.flow.alternatives.radialFlow.orientation)
    //                .values({"Clockwise", "Counter clockwise"}),
    //            zone.flow.alternatives.radialFlow.orientation);
    //        ImGui::SetCursorPosX(posX);
    //        AlienImGui::SliderFloat(
    //            AlienImGui::SliderFloatParameters()
    //                .name("Strength")
    //                .textWidth(RightColumnWidth)
    //                .min(0)
    //                .max(0.5f)
    //                .logarithmic(true)
    //                .format("%.5f")
    //                .defaultValue(&origZone.flow.alternatives.radialFlow.strength),
    //            &zone.flow.alternatives.radialFlow.strength);
    //        ImGui::SetCursorPosX(posX);
    //        AlienImGui::SliderFloat(
    //            AlienImGui::SliderFloatParameters()
    //                .name("Drift angle")
    //                .textWidth(RightColumnWidth)
    //                .min(-180.0f)
    //                .max(180.0f)
    //                .format("%.1f")
    //                .defaultValue(&origZone.flow.alternatives.radialFlow.driftAngle),
    //            &zone.flow.alternatives.radialFlow.driftAngle);
    //    }
    //    if (zone.flow.type == FlowType_Central) {
    //        ImGui::SetCursorPosX(posX);
    //        AlienImGui::SliderFloat(
    //            AlienImGui::SliderFloatParameters()
    //                .name("Strength")
    //                .textWidth(RightColumnWidth)
    //                .min(0)
    //                .max(0.5f)
    //                .logarithmic(true)
    //                .format("%.5f")
    //                .defaultValue(&origZone.flow.alternatives.centralFlow.strength),
    //            &zone.flow.alternatives.centralFlow.strength);
    //    }
    //    if (zone.flow.type == FlowType_Linear) {
    //        ImGui::SetCursorPosX(posX);
    //        AlienImGui::SliderFloat(
    //            AlienImGui::SliderFloatParameters()
    //                .name("Angle")
    //                .textWidth(RightColumnWidth)
    //                .min(-180.0f)
    //                .max(180.0f)
    //                .format("%.1f")
    //                .defaultValue(&origZone.flow.alternatives.linearFlow.angle),
    //            &zone.flow.alternatives.linearFlow.angle);
    //        ImGui::SetCursorPosX(posX);
    //        AlienImGui::SliderFloat(
    //            AlienImGui::SliderFloatParameters()
    //                .name("Strength")
    //                .textWidth(RightColumnWidth)
    //                .min(0)
    //                .max(0.5f)
    //                .logarithmic(true)
    //                .format("%.5f")
    //                .defaultValue(&origZone.flow.alternatives.linearFlow.strength),
    //            &zone.flow.alternatives.linearFlow.strength);
    //    }
    //    ImGui::EndDisabled();
    //}
    //AlienImGui::EndTreeNode();

    ///**
    // * Physics: Motion
    // */
    //if (AlienImGui::BeginTreeNode(AlienImGui::TreeNodeParameters().name("Physics: Motion"))) {
    //    AlienImGui::SliderFloat(
    //        AlienImGui::SliderFloatParameters()
    //            .name("Friction")
    //            .textWidth(RightColumnWidth)
    //            .min(0)
    //            .max(1)
    //            .logarithmic(true)
    //            .defaultValue(&origParameters.friction.zoneValues[zoneIndex].value)
    //            .disabledValue(&parameters.friction.baseValue)
    //            .format("%.4f"),
    //        &parameters.friction.zoneValues[zoneIndex].value,
    //        &parameters.friction.zoneValues[zoneIndex].enabled);
    //    AlienImGui::SliderFloat(
    //        AlienImGui::SliderFloatParameters()
    //            .name("Rigidity")
    //            .textWidth(RightColumnWidth)
    //            .min(0)
    //            .max(1)
    //            .defaultValue(&origParameters.rigidity.zoneValues[zoneIndex].value)
    //            .disabledValue(&parameters.rigidity.baseValue)
    //            .format("%.2f"),
    //        &parameters.rigidity.zoneValues[zoneIndex].value,
    //        &parameters.rigidity.zoneValues[zoneIndex].enabled);
    //}
    //AlienImGui::EndTreeNode();

    ///**
    // * Physics: Thresholds
    // */
    //if (AlienImGui::BeginTreeNode(AlienImGui::TreeNodeParameters().name("Physics: Thresholds"))) {
    //    AlienImGui::SliderFloat(
    //        AlienImGui::SliderFloatParameters()
    //            .name("Maximum force")
    //            .textWidth(RightColumnWidth)
    //            .min(0)
    //            .max(3.0f)
    //            .colorDependence(true)
    //            .defaultValue(origParameters.maxForce.zoneValues[zoneIndex].value)
    //            .disabledValue(parameters.maxForce.zoneValues[zoneIndex].value),
    //        parameters.maxForce.zoneValues[zoneIndex].value,
    //        &parameters.maxForce.zoneValues[zoneIndex].enabled);
    //}
    //AlienImGui::EndTreeNode();

    ///**
    // * Physics: Binding
    // */
    //if (AlienImGui::BeginTreeNode(AlienImGui::TreeNodeParameters().name("Physics: Binding"))) {
    //    AlienImGui::SliderFloat(
    //        AlienImGui::SliderFloatParameters()
    //            .name("Binding creation velocity")
    //            .textWidth(RightColumnWidth)
    //            .min(0)
    //            .max(2.0f)
    //            .defaultValue(&origParameters.cellFusionVelocity.zoneValues[zoneIndex].value)
    //            .disabledValue(&parameters.cellFusionVelocity.baseValue),
    //        &parameters.cellFusionVelocity.zoneValues[zoneIndex].value,
    //        &parameters.cellFusionVelocity.zoneValues[zoneIndex].enabled);
    //    AlienImGui::SliderFloat(
    //        AlienImGui::SliderFloatParameters()
    //            .name("Maximum energy")
    //            .textWidth(RightColumnWidth)
    //            .min(50.0f)
    //            .max(10000000.0f)
    //            .logarithmic(true)
    //            .infinity(true)
    //            .format("%.0f")
    //            .defaultValue(&origParameters.cellMaxBindingEnergy.zoneValues[zoneIndex].value)
    //            .disabledValue(&parameters.cellMaxBindingEnergy.baseValue),
    //        &parameters.cellMaxBindingEnergy.zoneValues[zoneIndex].value,
    //        &parameters.cellMaxBindingEnergy.zoneValues[zoneIndex].enabled);
    //}
    //AlienImGui::EndTreeNode();


    ///**
    // * Physics: Radiation
    // */
    //if (AlienImGui::BeginTreeNode(AlienImGui::TreeNodeParameters().name("Physics: Radiation"))) {
    //    AlienImGui::Checkbox(
    //        AlienImGui::CheckboxParameters()
    //            .name("Disable radiation sources")
    //            .textWidth(RightColumnWidth)
    //            .defaultValue(origParameters.radiationDisableSources.zoneValues[zoneIndex])
    //            .tooltip("If activated, all radiation sources within this zone are deactivated."),
    //        parameters.radiationDisableSources.zoneValues[zoneIndex]);

    //    AlienImGui::SliderFloat(
    //        AlienImGui::SliderFloatParameters()
    //            .name("Absorption factor")
    //            .textWidth(RightColumnWidth)
    //            .logarithmic(true)
    //            .colorDependence(true)
    //            .min(0)
    //            .max(1.0)
    //            .format("%.4f")
    //            .defaultValue(origParameters.radiationAbsorption.zoneValues[zoneIndex].value)
    //            .disabledValue(parameters.radiationAbsorption.baseValue),
    //        parameters.radiationAbsorption.zoneValues[zoneIndex].value,
    //        &parameters.radiationAbsorption.zoneValues[zoneIndex].enabled);

    //    AlienImGui::SliderFloat(
    //        AlienImGui::SliderFloatParameters()
    //            .name("Radiation type 1: Strength")
    //            .textWidth(RightColumnWidth)
    //            .colorDependence(true)
    //            .min(0)
    //            .max(0.01f)
    //            .format("%.6f")
    //            .logarithmic(true)
    //            .defaultValue(origParameters.radiationType1_strength.zoneValues[zoneIndex].value)
    //            .disabledValue(parameters.radiationType1_strength.baseValue),
    //        parameters.radiationType1_strength.zoneValues[zoneIndex].value,
    //        &parameters.radiationType1_strength.zoneValues[zoneIndex].enabled);
    //}
    //AlienImGui::EndTreeNode();

    ///**
    // * Cell life cycle
    // */
    //if (AlienImGui::BeginTreeNode(AlienImGui::TreeNodeParameters().name("Cell life cycle"))) {
    //    AlienImGui::SliderFloat(
    //        AlienImGui::SliderFloatParameters()
    //            .name("Minimum energy")
    //            .textWidth(RightColumnWidth)
    //            .colorDependence(true)
    //            .min(10.0f)
    //            .max(200.0f)
    //            .defaultValue(origParameters.minCellEnergy.zoneValues[zoneIndex].value)
    //            .disabledValue(parameters.minCellEnergy.baseValue),
    //        parameters.minCellEnergy.zoneValues[zoneIndex].value,
    //        &parameters.minCellEnergy.zoneValues[zoneIndex].enabled);
    //    AlienImGui::SliderFloat(
    //        AlienImGui::SliderFloatParameters()
    //            .name("Decay rate of dying cells")
    //            .colorDependence(true)
    //            .textWidth(RightColumnWidth)
    //            .min(1e-6f)
    //            .max(0.1f)
    //            .format("%.6f")
    //            .logarithmic(true)
    //            .defaultValue(origParameters.cellDeathProbability.zoneValues[zoneIndex].value)
    //            .disabledValue(parameters.cellDeathProbability.baseValue),
    //        parameters.cellDeathProbability.zoneValues[zoneIndex].value,
    //        &parameters.cellDeathProbability.zoneValues[zoneIndex].enabled);
    //}
    //AlienImGui::EndTreeNode();

    ///**
    // * Mutation 
    // */
    //if (AlienImGui::BeginTreeNode(AlienImGui::TreeNodeParameters().name("Genome copy mutations"))) {
    //    AlienImGui::SliderFloat(
    //        AlienImGui::SliderFloatParameters()
    //            .name("Neuron weights and biases")
    //            .textWidth(RightColumnWidth)
    //            .min(0.0f)
    //            .max(1.0f)
    //            .format("%.7f")
    //            .colorDependence(true)
    //            .logarithmic(true)
    //            .defaultValue(origParameters.copyMutationNeuronData.zoneValues[zoneIndex].value)
    //            .disabledValue(parameters.copyMutationNeuronData.baseValue),
    //        parameters.copyMutationNeuronData.zoneValues[zoneIndex].value,
    //        &parameters.copyMutationNeuronData.zoneValues[zoneIndex].enabled);
    //    AlienImGui::SliderFloat(
    //        AlienImGui::SliderFloatParameters()
    //            .name("Cell properties")
    //            .textWidth(RightColumnWidth)
    //            .min(0.0f)
    //            .max(1.0f)
    //            .format("%.7f")
    //            .logarithmic(true)
    //            .colorDependence(true)
    //            .defaultValue(origParameters.copyMutationCellProperties.zoneValues[zoneIndex].value)
    //            .disabledValue(parameters.copyMutationCellProperties.baseValue),
    //        parameters.copyMutationCellProperties.zoneValues[zoneIndex].value,
    //        &parameters.copyMutationCellProperties.zoneValues[zoneIndex].enabled);
    //    AlienImGui::SliderFloat(
    //        AlienImGui::SliderFloatParameters()
    //            .name("Geometry")
    //            .textWidth(RightColumnWidth)
    //            .min(0.0f)
    //            .max(1.0f)
    //            .format("%.7f")
    //            .logarithmic(true)
    //            .colorDependence(true)
    //            .defaultValue(origParameters.copyMutationGeometry.zoneValues[zoneIndex].value)
    //            .disabledValue(parameters.copyMutationGeometry.baseValue),
    //        parameters.copyMutationGeometry.zoneValues[zoneIndex].value,
    //        &parameters.copyMutationGeometry.zoneValues[zoneIndex].enabled);
    //    AlienImGui::SliderFloat(
    //        AlienImGui::SliderFloatParameters()
    //            .name("Custom geometry")
    //            .textWidth(RightColumnWidth)
    //            .min(0.0f)
    //            .max(1.0f)
    //            .format("%.7f")
    //            .logarithmic(true)
    //            .colorDependence(true)
    //            .defaultValue(origParameters.copyMutationCustomGeometry.zoneValues[zoneIndex].value)
    //            .disabledValue(parameters.copyMutationCustomGeometry.baseValue),
    //        parameters.copyMutationCustomGeometry.zoneValues[zoneIndex].value,
    //        &parameters.copyMutationCustomGeometry.zoneValues[zoneIndex].enabled);
    //    AlienImGui::SliderFloat(
    //        AlienImGui::SliderFloatParameters()
    //            .name("Cell function type")
    //            .textWidth(RightColumnWidth)
    //            .min(0.0f)
    //            .max(1.0f)
    //            .format("%.7f")
    //            .logarithmic(true)
    //            .colorDependence(true)
    //            .defaultValue(origParameters.copyMutationCellType.zoneValues[zoneIndex].value)
    //            .disabledValue(parameters.copyMutationCellType.baseValue),
    //        parameters.copyMutationCellType.zoneValues[zoneIndex].value,
    //        &parameters.copyMutationCellType.zoneValues[zoneIndex].enabled);
    //    AlienImGui::SliderFloat(
    //        AlienImGui::SliderFloatParameters()
    //            .name("Cell insertion")
    //            .textWidth(RightColumnWidth)
    //            .min(0.0f)
    //            .max(1.0f)
    //            .format("%.7f")
    //            .logarithmic(true)
    //            .colorDependence(true)
    //            .defaultValue(origParameters.copyMutationInsertion.zoneValues[zoneIndex].value)
    //            .disabledValue(parameters.copyMutationInsertion.baseValue),
    //        parameters.copyMutationInsertion.zoneValues[zoneIndex].value,
    //        &parameters.copyMutationInsertion.zoneValues[zoneIndex].enabled);
    //    AlienImGui::SliderFloat(
    //        AlienImGui::SliderFloatParameters()
    //            .name("Cell deletion")
    //            .textWidth(RightColumnWidth)
    //            .min(0.0f)
    //            .max(1.0f)
    //            .format("%.7f")
    //            .logarithmic(true)
    //            .colorDependence(true)
    //            .defaultValue(origParameters.copyMutationDeletion.zoneValues[zoneIndex].value)
    //            .disabledValue(parameters.copyMutationDeletion.baseValue),
    //        parameters.copyMutationDeletion.zoneValues[zoneIndex].value,
    //        &parameters.copyMutationDeletion.zoneValues[zoneIndex].enabled);
    //    AlienImGui::SliderFloat(
    //        AlienImGui::SliderFloatParameters()
    //            .name("Translation")
    //            .textWidth(RightColumnWidth)
    //            .min(0.0f)
    //            .max(1.0f)
    //            .format("%.7f")
    //            .logarithmic(true)
    //            .colorDependence(true)
    //            .defaultValue(origParameters.copyMutationTranslation.zoneValues[zoneIndex].value)
    //            .disabledValue(parameters.copyMutationTranslation.baseValue),
    //        parameters.copyMutationTranslation.zoneValues[zoneIndex].value,
    //        &parameters.copyMutationTranslation.zoneValues[zoneIndex].enabled);
    //    AlienImGui::SliderFloat(
    //        AlienImGui::SliderFloatParameters()
    //            .name("Duplication")
    //            .textWidth(RightColumnWidth)
    //            .min(0.0f)
    //            .max(1.0f)
    //            .format("%.7f")
    //            .logarithmic(true)
    //            .colorDependence(true)
    //            .defaultValue(origParameters.copyMutationDuplication.zoneValues[zoneIndex].value)
    //            .disabledValue(parameters.copyMutationDuplication.baseValue),
    //        parameters.copyMutationDuplication.zoneValues[zoneIndex].value,
    //        &parameters.copyMutationDuplication.zoneValues[zoneIndex].enabled);
    //    AlienImGui::SliderFloat(
    //        AlienImGui::SliderFloatParameters()
    //            .name("Individual cell color")
    //            .textWidth(RightColumnWidth)
    //            .min(0.0f)
    //            .max(1.0f)
    //            .format("%.7f")
    //            .logarithmic(true)
    //            .colorDependence(true)
    //            .defaultValue(origParameters.copyMutationCellColor.zoneValues[zoneIndex].value)
    //            .disabledValue(parameters.copyMutationCellColor.baseValue),
    //        parameters.copyMutationCellColor.zoneValues[zoneIndex].value,
    //        &parameters.copyMutationCellColor.zoneValues[zoneIndex].enabled);
    //    AlienImGui::SliderFloat(
    //        AlienImGui::SliderFloatParameters()
    //            .name("Sub-genome color")
    //            .textWidth(RightColumnWidth)
    //            .min(0.0f)
    //            .max(1.0f)
    //            .format("%.7f")
    //            .logarithmic(true)
    //            .colorDependence(true)
    //            .defaultValue(origParameters.copyMutationSubgenomeColor.zoneValues[zoneIndex].value)
    //            .disabledValue(parameters.copyMutationSubgenomeColor.baseValue),
    //        parameters.copyMutationSubgenomeColor.zoneValues[zoneIndex].value,
    //        &parameters.copyMutationSubgenomeColor.zoneValues[zoneIndex].enabled);
    //    AlienImGui::SliderFloat(
    //        AlienImGui::SliderFloatParameters()
    //            .name("Genome color")
    //            .textWidth(RightColumnWidth)
    //            .min(0.0f)
    //            .max(1.0f)
    //            .format("%.7f")
    //            .logarithmic(true)
    //            .colorDependence(true)
    //            .defaultValue(origParameters.copyMutationGenomeColor.zoneValues[zoneIndex].value)
    //            .disabledValue(parameters.copyMutationGenomeColor.baseValue),
    //        parameters.copyMutationGenomeColor.zoneValues[zoneIndex].value,
    //        &parameters.copyMutationGenomeColor.zoneValues[zoneIndex].enabled);
    //}
    //AlienImGui::EndTreeNode();

    ///**
    // * Attacker
    // */
    //if (AlienImGui::BeginTreeNode(AlienImGui::TreeNodeParameters().name("Cell type: Attacker"))) {
    //    AlienImGui::InputFloatColorMatrix(
    //        AlienImGui::InputFloatColorMatrixParameters()
    //            .name("Food chain color matrix")
    //            .max(1)
    //            .textWidth(RightColumnWidth)
    //            .defaultValue(toVector<MAX_COLORS, MAX_COLORS>(origParameters.attackerFoodChainColorMatrix.zoneValues[zoneIndex].value))
    //            .disabledValue(toVector<MAX_COLORS, MAX_COLORS>(parameters.attackerFoodChainColorMatrix.baseValue)),
    //        parameters.attackerFoodChainColorMatrix.zoneValues[zoneIndex].value,
    //        &parameters.attackerFoodChainColorMatrix.zoneValues[zoneIndex].enabled);
    //    AlienImGui::InputFloatColorMatrix(
    //        AlienImGui::InputFloatColorMatrixParameters()
    //            .name("Complex creature protection")
    //            .textWidth(RightColumnWidth)
    //            .min(0)
    //            .max(20.0f)
    //            .defaultValue(toVector<MAX_COLORS, MAX_COLORS>(origParameters.attackerComplexCreatureProtection.zoneValues[zoneIndex].value))
    //            .disabledValue(toVector<MAX_COLORS, MAX_COLORS>(parameters.attackerComplexCreatureProtection.baseValue)),
    //        parameters.attackerComplexCreatureProtection.zoneValues[zoneIndex].value,
    //        &parameters.attackerComplexCreatureProtection.zoneValues[zoneIndex].enabled);
    //    AlienImGui::SliderFloat(
    //        AlienImGui::SliderFloatParameters()
    //            .name("Energy cost")
    //            .textWidth(RightColumnWidth)
    //            .colorDependence(true)
    //            .min(0)
    //            .max(1.0f)
    //            .format("%.5f")
    //            .logarithmic(true)
    //            .defaultValue(origParameters.attackerEnergyCost.zoneValues[zoneIndex].value)
    //            .disabledValue(parameters.attackerEnergyCost.baseValue),
    //        parameters.attackerEnergyCost.zoneValues[zoneIndex].value,
    //        &parameters.attackerEnergyCost.zoneValues[zoneIndex].enabled);
    //}
    //AlienImGui::EndTreeNode();

    ///**
    // * Expert settings: Advanced absorption control
    // */
    //if (AlienImGui::BeginTreeNode(AlienImGui::TreeNodeParameters()
    //                                  .name("Expert settings: Advanced energy absorption control")
    //                                  .visible(parameters.advancedAbsorptionControlToggle.value)
    //                                  .blinkWhenActivated(true))) {
    //    AlienImGui::SliderFloat(
    //        AlienImGui::SliderFloatParameters()
    //            .name("Low velocity penalty")
    //            .textWidth(RightColumnWidth)
    //            .colorDependence(true)
    //            .min(0)
    //            .max(1.0f)
    //            .format("%.2f")
    //            .defaultValue(origParameters.radiationAbsorptionLowVelocityPenalty.zoneValues[zoneIndex].value)
    //            .disabledValue(parameters.radiationAbsorptionLowVelocityPenalty.baseValue),
    //        parameters.radiationAbsorptionLowVelocityPenalty.zoneValues[zoneIndex].value,
    //        &parameters.radiationAbsorptionLowVelocityPenalty.zoneValues[zoneIndex].enabled);
    //    AlienImGui::SliderFloat(
    //        AlienImGui::SliderFloatParameters()
    //            .name("Low genome complexity penalty")
    //            .textWidth(RightColumnWidth)
    //            .colorDependence(true)
    //            .min(0)
    //            .max(1.0f)
    //            .format("%.2f")
    //            .defaultValue(origParameters.radiationAbsorptionLowGenomeComplexityPenalty.zoneValues[zoneIndex].value)
    //            .disabledValue(parameters.radiationAbsorptionLowGenomeComplexityPenalty.baseValue),
    //        parameters.radiationAbsorptionLowGenomeComplexityPenalty.zoneValues[zoneIndex].value,
    //        &parameters.radiationAbsorptionLowGenomeComplexityPenalty.zoneValues[zoneIndex].enabled);
    //}
    //AlienImGui::EndTreeNode();

    ///**
    // * Expert settings: Advanced attacker control
    // */
    //if (AlienImGui::BeginTreeNode(AlienImGui::TreeNodeParameters()
    //                                  .name("Expert settings: Advanced attacker control")
    //                                  .visible(parameters.advancedAttackerControlToggle.value)
    //                                  .blinkWhenActivated(true))) {
    //    AlienImGui::InputFloatColorMatrix(
    //        AlienImGui::InputFloatColorMatrixParameters()
    //            .name("New complex mutant protection")
    //            .textWidth(RightColumnWidth)
    //            .min(0)
    //            .max(1.0f)
    //            .defaultValue(toVector<MAX_COLORS, MAX_COLORS>(origParameters.attackerNewComplexMutantProtection.zoneValues[zoneIndex].value))
    //            .disabledValue(toVector<MAX_COLORS, MAX_COLORS>(parameters.attackerNewComplexMutantProtection.baseValue)),
    //        parameters.attackerNewComplexMutantProtection.zoneValues[zoneIndex].value,
    //        &parameters.attackerNewComplexMutantProtection.zoneValues[zoneIndex].enabled);
    //    AlienImGui::SliderFloat(
    //        AlienImGui::SliderFloatParameters()
    //            .name("Geometry penalty")
    //            .textWidth(RightColumnWidth)
    //            .colorDependence(true)
    //            .min(0)
    //            .max(5.0f)
    //            .defaultValue(origParameters.attackerGeometryDeviationProtection.zoneValues[zoneIndex].value)
    //            .disabledValue(parameters.attackerGeometryDeviationProtection.baseValue),
    //        parameters.attackerGeometryDeviationProtection.zoneValues[zoneIndex].value,
    //        &parameters.attackerGeometryDeviationProtection.zoneValues[zoneIndex].enabled);
    //    AlienImGui::SliderFloat(
    //        AlienImGui::SliderFloatParameters()
    //            .name("Connections mismatch penalty")
    //            .textWidth(RightColumnWidth)
    //            .colorDependence(true)
    //            .min(0)
    //            .max(1.0f)
    //            .defaultValue(origParameters.attackerConnectionsMismatchProtection.zoneValues[zoneIndex].value)
    //            .disabledValue(parameters.attackerConnectionsMismatchProtection.baseValue),
    //        parameters.attackerConnectionsMismatchProtection.zoneValues[zoneIndex].value,
    //        &parameters.attackerConnectionsMismatchProtection.zoneValues[zoneIndex].enabled);
    //}
    //AlienImGui::EndTreeNode();

    ///**
    // * Expert settings: Cell age limiter
    // */
    //if (AlienImGui::BeginTreeNode(AlienImGui::TreeNodeParameters()
    //                                  .name("Expert settings: Cell age limiter")
    //                                  .visible(parameters.cellAgeLimiterToggle.value)
    //                                  .blinkWhenActivated(true))) {
    //    AlienImGui::SliderFloat(
    //        AlienImGui::SliderFloatParameters()
    //            .name("Maximum inactive cell age")
    //            .textWidth(RightColumnWidth)
    //            .colorDependence(true)
    //            .min(1.0f)
    //            .max(10000000.0f)
    //            .logarithmic(true)
    //            .infinity(true)
    //            .format("%.0f")
    //            .defaultValue(origParameters.maxAgeForInactiveCells.zoneValues[zoneIndex].value)
    //            .disabledValue(parameters.maxAgeForInactiveCells.baseValue),
    //        parameters.maxAgeForInactiveCells.zoneValues[zoneIndex].value,
    //        &parameters.maxAgeForInactiveCells.zoneValues[zoneIndex].enabled);
    //}
    //AlienImGui::EndTreeNode();


    ParametersValidationService::get().validateAndCorrect(zone, parameters);

    if (parameters != lastParameters) {
        auto isRunning = _simulationFacade->isSimulationRunning();
        _simulationFacade->setSimulationParameters(
            parameters, isRunning ? SimulationParametersUpdateConfig::AllExceptChangingPositions : SimulationParametersUpdateConfig::All);
    }
}

std::string _SimulationParametersZoneWidgets::getLocationName()
{
    return "Simulation parameters for '" + _zoneName + "'";
}

int _SimulationParametersZoneWidgets::getLocationIndex() const
{
    return _locationIndex;
}

void _SimulationParametersZoneWidgets::setLocationIndex(int locationIndex)
{
    _locationIndex = locationIndex;
}

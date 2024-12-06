#include "SimulationParametersZoneWidgets.h"

#include "EngineInterface/SimulationFacade.h"
#include "EngineInterface/SimulationParametersValidationService.h"
#include "EngineInterface/SimulationParametersZone.h"

#include "AlienImGui.h"
#include "LoginDialog.h"
#include "SimulationInteractionController.h"

namespace
{
    auto constexpr RightColumnWidth = 285.0f;

    template <int numRows, int numCols, typename T>
    std::vector<std::vector<T>> toVector(T const v[numRows][numCols])
    {
        std::vector<std::vector<T>> result;
        for (int row = 0; row < numRows; ++row) {
            std::vector<T> rowVector;
            for (int col = 0; col < numCols; ++col) {
                rowVector.emplace_back(v[row][col]);
            }
            result.emplace_back(rowVector);
        }
        return result;
    }
}

void _SimulationParametersZoneWidgets::init(SimulationFacade const& simulationFacade, int zoneIndex)
{
    _simulationFacade = simulationFacade;
    _zoneIndex = zoneIndex;
}

void _SimulationParametersZoneWidgets::process()
{
    auto parameters = _simulationFacade->getSimulationParameters();
    auto origParameters = _simulationFacade->getOriginalSimulationParameters();
    auto lastParameters = parameters;

    SimulationParametersZone& spot = parameters.zone[_zoneIndex];
    SimulationParametersZone const& origSpot = origParameters.zone[_zoneIndex];
    SimulationParametersZone const& lastSpot = lastParameters.zone[_zoneIndex];

    if (ImGui::BeginChild("##", ImVec2(0, 0), false, ImGuiWindowFlags_HorizontalScrollbar)) {
        auto worldSize = _simulationFacade->getWorldSize();

        /**
         * Colors and location
         */
        if (AlienImGui::BeginTreeNode(AlienImGui::TreeNodeParameters().text("Visualization"))) {
            AlienImGui::ColorButtonWithPicker(
                AlienImGui::ColorButtonWithPickerParameters().name("Background color").textWidth(RightColumnWidth).defaultValue(origSpot.color),
                spot.color,
                _backupColor,
                _zoneColorPalette.getReference());
            AlienImGui::EndTreeNode();
        }

        if (AlienImGui::BeginTreeNode(AlienImGui::TreeNodeParameters().text("Location"))) {
            if (AlienImGui::Switcher(
                    AlienImGui::SwitcherParameters()
                        .name("Shape")
                        .values({"Circular", "Rectangular"})
                        .textWidth(RightColumnWidth)
                        .defaultValue(origSpot.shapeType),
                    spot.shapeType)) {
                setDefaultSpotData(spot);
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
                    .defaultValue(RealVector2D{origSpot.posX, origSpot.posY})
                    .format("%.2f")
                    .getMousePickerEnabledFunc(getMousePickerEnabledFunc)
                    .setMousePickerEnabledFunc(setMousePickerEnabledFunc)
                    .getMousePickerPositionFunc(getMousePickerPositionFunc),
                spot.posX,
                spot.posY);
            AlienImGui::SliderFloat2(
                AlienImGui::SliderFloat2Parameters()
                    .name("Velocity (x,y)")
                    .textWidth(RightColumnWidth)
                    .min({-4.0f, -4.0f})
                    .max({4.0f, 4.0f})
                    .defaultValue(RealVector2D{origSpot.velX, origSpot.velY})
                    .format("%.2f"),
                spot.velX,
                spot.velY);
            auto maxRadius = toFloat(std::max(worldSize.x, worldSize.y));
            if (spot.shapeType == SpotShapeType_Circular) {
                AlienImGui::SliderFloat(
                    AlienImGui::SliderFloatParameters()
                        .name("Core radius")
                        .textWidth(RightColumnWidth)
                        .min(0)
                        .max(maxRadius)
                        .defaultValue(&origSpot.shapeData.circularSpot.coreRadius)
                        .format("%.1f"),
                    &spot.shapeData.circularSpot.coreRadius);
            }
            if (spot.shapeType == SpotShapeType_Rectangular) {
                AlienImGui::SliderFloat2(
                    AlienImGui::SliderFloat2Parameters()
                        .name("Size (x,y)")
                        .textWidth(RightColumnWidth)
                        .min({0, 0})
                        .max({toFloat(worldSize.x), toFloat(worldSize.y)})
                        .defaultValue(RealVector2D{origSpot.shapeData.rectangularSpot.width, origSpot.shapeData.rectangularSpot.height})
                        .format("%.1f"),
                    spot.shapeData.rectangularSpot.width,
                    spot.shapeData.rectangularSpot.height);
            }

            AlienImGui::SliderFloat(
                AlienImGui::SliderFloatParameters()
                    .name("Fade-out radius")
                    .textWidth(RightColumnWidth)
                    .min(0)
                    .max(maxRadius)
                    .defaultValue(&origSpot.fadeoutRadius)
                    .format("%.1f"),
                &spot.fadeoutRadius);
            AlienImGui::EndTreeNode();
        }

        /**
         * Flow
         */
        if (AlienImGui::BeginTreeNode(AlienImGui::TreeNodeParameters().text("Force field"))) {
            auto isForceFieldActive = spot.flowType != FlowType_None;

            auto forceFieldTypeIntern = std::max(0, spot.flowType - 1);  //FlowType_None should not be selectable in ComboBox
            auto origForceFieldTypeIntern = std::max(0, origSpot.flowType - 1);
            if (ImGui::Checkbox("##forceField", &isForceFieldActive)) {
                spot.flowType = isForceFieldActive ? FlowType_Radial : FlowType_None;
            }
            ImGui::SameLine();
            ImGui::BeginDisabled(!isForceFieldActive);
            auto posX = ImGui::GetCursorPos().x;
            if (AlienImGui::Combo(
                    AlienImGui::ComboParameters()
                        .name("Type")
                        .values({"Radial", "Central", "Linear"})
                        .textWidth(RightColumnWidth)
                        .defaultValue(origForceFieldTypeIntern),
                    forceFieldTypeIntern)) {
                spot.flowType = forceFieldTypeIntern + 1;
                if (spot.flowType == FlowType_Radial) {
                    spot.flowData.radialFlow = RadialFlow();
                }
                if (spot.flowType == FlowType_Central) {
                    spot.flowData.centralFlow = CentralFlow();
                }
                if (spot.flowType == FlowType_Linear) {
                    spot.flowData.linearFlow = LinearFlow();
                }
            }
            if (spot.flowType == FlowType_Radial) {
                ImGui::SetCursorPosX(posX);
                AlienImGui::Combo(
                    AlienImGui::ComboParameters()
                        .name("Orientation")
                        .textWidth(RightColumnWidth)
                        .defaultValue(origSpot.flowData.radialFlow.orientation)
                        .values({"Clockwise", "Counter clockwise"}),
                    spot.flowData.radialFlow.orientation);
                ImGui::SetCursorPosX(posX);
                AlienImGui::SliderFloat(
                    AlienImGui::SliderFloatParameters()
                        .name("Strength")
                        .textWidth(RightColumnWidth)
                        .min(0)
                        .max(0.5f)
                        .logarithmic(true)
                        .format("%.5f")
                        .defaultValue(&origSpot.flowData.radialFlow.strength),
                    &spot.flowData.radialFlow.strength);
                ImGui::SetCursorPosX(posX);
                AlienImGui::SliderFloat(
                    AlienImGui::SliderFloatParameters()
                        .name("Drift angle")
                        .textWidth(RightColumnWidth)
                        .min(-180.0f)
                        .max(180.0f)
                        .format("%.1f")
                        .defaultValue(&origSpot.flowData.radialFlow.driftAngle),
                    &spot.flowData.radialFlow.driftAngle);
            }
            if (spot.flowType == FlowType_Central) {
                ImGui::SetCursorPosX(posX);
                AlienImGui::SliderFloat(
                    AlienImGui::SliderFloatParameters()
                        .name("Strength")
                        .textWidth(RightColumnWidth)
                        .min(0)
                        .max(0.5f)
                        .logarithmic(true)
                        .format("%.5f")
                        .defaultValue(&origSpot.flowData.centralFlow.strength),
                    &spot.flowData.centralFlow.strength);
            }
            if (spot.flowType == FlowType_Linear) {
                ImGui::SetCursorPosX(posX);
                AlienImGui::SliderFloat(
                    AlienImGui::SliderFloatParameters()
                        .name("Angle")
                        .textWidth(RightColumnWidth)
                        .min(-180.0f)
                        .max(180.0f)
                        .format("%.1f")
                        .defaultValue(&origSpot.flowData.linearFlow.angle),
                    &spot.flowData.linearFlow.angle);
                ImGui::SetCursorPosX(posX);
                AlienImGui::SliderFloat(
                    AlienImGui::SliderFloatParameters()
                        .name("Strength")
                        .textWidth(RightColumnWidth)
                        .min(0)
                        .max(0.5f)
                        .logarithmic(true)
                        .format("%.5f")
                        .defaultValue(&origSpot.flowData.linearFlow.strength),
                    &spot.flowData.linearFlow.strength);
            }
            ImGui::EndDisabled();
            AlienImGui::EndTreeNode();
        }

        /**
         * Physics: Motion
         */
        if (AlienImGui::BeginTreeNode(AlienImGui::TreeNodeParameters().text("Physics: Motion"))) {
            AlienImGui::SliderFloat(
                AlienImGui::SliderFloatParameters()
                    .name("Friction")
                    .textWidth(RightColumnWidth)
                    .min(0)
                    .max(1)
                    .logarithmic(true)
                    .defaultValue(&origSpot.values.friction)
                    .disabledValue(&parameters.baseValues.friction)
                    .format("%.4f"),
                &spot.values.friction,
                &spot.activatedValues.friction);
            AlienImGui::SliderFloat(
                AlienImGui::SliderFloatParameters()
                    .name("Rigidity")
                    .textWidth(RightColumnWidth)
                    .min(0)
                    .max(1)
                    .defaultValue(&origSpot.values.rigidity)
                    .disabledValue(&parameters.baseValues.rigidity)
                    .format("%.2f"),
                &spot.values.rigidity,
                &spot.activatedValues.rigidity);
            AlienImGui::EndTreeNode();
        }

        /**
         * Physics: Thresholds
         */
        if (AlienImGui::BeginTreeNode(AlienImGui::TreeNodeParameters().text("Physics: Thresholds"))) {
            AlienImGui::SliderFloat(
                AlienImGui::SliderFloatParameters()
                    .name("Maximum force")
                    .textWidth(RightColumnWidth)
                    .min(0)
                    .max(3.0f)
                    .colorDependence(true)
                    .defaultValue(origSpot.values.cellMaxForce)
                    .disabledValue(parameters.baseValues.cellMaxForce),
                spot.values.cellMaxForce,
                &spot.activatedValues.cellMaxForce);
            AlienImGui::EndTreeNode();
        }

        /**
         * Physics: Binding
         */
        if (AlienImGui::BeginTreeNode(AlienImGui::TreeNodeParameters().text("Physics: Binding"))) {
            AlienImGui::SliderFloat(
                AlienImGui::SliderFloatParameters()
                    .name("Binding creation velocity")
                    .textWidth(RightColumnWidth)
                    .min(0)
                    .max(2.0f)
                    .defaultValue(&origSpot.values.cellFusionVelocity)
                    .disabledValue(&parameters.baseValues.cellFusionVelocity),
                &spot.values.cellFusionVelocity,
                &spot.activatedValues.cellFusionVelocity);
            AlienImGui::SliderFloat(
                AlienImGui::SliderFloatParameters()
                    .name("Maximum energy")
                    .textWidth(RightColumnWidth)
                    .min(50.0f)
                    .max(10000000.0f)
                    .logarithmic(true)
                    .infinity(true)
                    .format("%.0f")
                    .defaultValue(&origSpot.values.cellMaxBindingEnergy)
                    .disabledValue(&parameters.baseValues.cellMaxBindingEnergy),
                &spot.values.cellMaxBindingEnergy,
                &spot.activatedValues.cellMaxBindingEnergy);
            AlienImGui::EndTreeNode();
        }

        /**
         * Physics: Radiation
         */
        if (AlienImGui::BeginTreeNode(AlienImGui::TreeNodeParameters().text("Physics: Radiation"))) {
            AlienImGui::Checkbox(
                AlienImGui::CheckboxParameters()
                    .name("Disable radiation sources")
                    .textWidth(RightColumnWidth)
                    .defaultValue(origSpot.values.radiationDisableSources)
                    .tooltip("If activated, all radiation sources within this zone are deactivated."),
                spot.values.radiationDisableSources);
            spot.activatedValues.radiationDisableSources = spot.values.radiationDisableSources;

            AlienImGui::SliderFloat(
                AlienImGui::SliderFloatParameters()
                    .name("Absorption factor")
                    .textWidth(RightColumnWidth)
                    .logarithmic(true)
                    .colorDependence(true)
                    .min(0)
                    .max(1.0)
                    .format("%.4f")
                    .defaultValue(origSpot.values.radiationAbsorption)
                    .disabledValue(parameters.baseValues.radiationAbsorption),
                spot.values.radiationAbsorption,
                &spot.activatedValues.radiationAbsorption);

            AlienImGui::SliderFloat(
                AlienImGui::SliderFloatParameters()
                    .name("Radiation type 1: Strength")
                    .textWidth(RightColumnWidth)
                    .colorDependence(true)
                    .min(0)
                    .max(0.01f)
                    .logarithmic(true)
                    .defaultValue(origSpot.values.radiationCellAgeStrength)
                    .disabledValue(parameters.baseValues.radiationCellAgeStrength)
                    .format("%.6f"),
                spot.values.radiationCellAgeStrength,
                &spot.activatedValues.radiationCellAgeStrength);
            AlienImGui::EndTreeNode();
        }

        /**
         * Cell life cycle
         */
        if (AlienImGui::BeginTreeNode(AlienImGui::TreeNodeParameters().text("Cell life cycle"))) {
            AlienImGui::SliderFloat(
                AlienImGui::SliderFloatParameters()
                    .name("Minimum energy")
                    .textWidth(RightColumnWidth)
                    .colorDependence(true)
                    .min(10.0f)
                    .max(200.0f)
                    .defaultValue(origSpot.values.cellMinEnergy)
                    .disabledValue(parameters.baseValues.cellMinEnergy),
                spot.values.cellMinEnergy,
                &spot.activatedValues.cellMinEnergy);
            AlienImGui::SliderFloat(
                AlienImGui::SliderFloatParameters()
                    .name("Decay rate of dying cells")
                    .colorDependence(true)
                    .textWidth(RightColumnWidth)
                    .min(1e-6f)
                    .max(0.1f)
                    .format("%.6f")
                    .logarithmic(true)
                    .defaultValue(origSpot.values.cellDeathProbability)
                    .disabledValue(parameters.baseValues.cellDeathProbability),
                spot.values.cellDeathProbability,
                &spot.activatedValues.cellDeathProbability);
            AlienImGui::EndTreeNode();
        }

        /**
         * Mutation 
         */
        if (AlienImGui::BeginTreeNode(AlienImGui::TreeNodeParameters().text("Genome copy mutations"))) {
            AlienImGui::SliderFloat(
                AlienImGui::SliderFloatParameters()
                    .name("Neuron weights and biases")
                    .textWidth(RightColumnWidth)
                    .min(0.0f)
                    .max(1.0f)
                    .format("%.7f")
                    .colorDependence(true)
                    .logarithmic(true)
                    .defaultValue(origSpot.values.cellCopyMutationNeuronData)
                    .disabledValue(parameters.baseValues.cellCopyMutationNeuronData),
                spot.values.cellCopyMutationNeuronData,
                &spot.activatedValues.cellCopyMutationNeuronData);
            AlienImGui::SliderFloat(
                AlienImGui::SliderFloatParameters()
                    .name("Cell properties")
                    .textWidth(RightColumnWidth)
                    .min(0.0f)
                    .max(1.0f)
                    .format("%.7f")
                    .logarithmic(true)
                    .colorDependence(true)
                    .defaultValue(origSpot.values.cellCopyMutationCellProperties)
                    .disabledValue(parameters.baseValues.cellCopyMutationCellProperties),
                spot.values.cellCopyMutationCellProperties,
                &spot.activatedValues.cellCopyMutationCellProperties);
            AlienImGui::SliderFloat(
                AlienImGui::SliderFloatParameters()
                    .name("Geometry")
                    .textWidth(RightColumnWidth)
                    .min(0.0f)
                    .max(1.0f)
                    .format("%.7f")
                    .logarithmic(true)
                    .colorDependence(true)
                    .defaultValue(origSpot.values.cellCopyMutationGeometry)
                    .disabledValue(parameters.baseValues.cellCopyMutationGeometry),
                spot.values.cellCopyMutationGeometry,
                &spot.activatedValues.cellCopyMutationGeometry);
            AlienImGui::SliderFloat(
                AlienImGui::SliderFloatParameters()
                    .name("Custom geometry")
                    .textWidth(RightColumnWidth)
                    .min(0.0f)
                    .max(1.0f)
                    .format("%.7f")
                    .logarithmic(true)
                    .colorDependence(true)
                    .defaultValue(origSpot.values.cellCopyMutationCustomGeometry)
                    .disabledValue(parameters.baseValues.cellCopyMutationCustomGeometry),
                spot.values.cellCopyMutationCustomGeometry,
                &spot.activatedValues.cellCopyMutationCustomGeometry);
            AlienImGui::SliderFloat(
                AlienImGui::SliderFloatParameters()
                    .name("Cell function type")
                    .textWidth(RightColumnWidth)
                    .min(0.0f)
                    .max(1.0f)
                    .format("%.7f")
                    .logarithmic(true)
                    .colorDependence(true)
                    .defaultValue(origSpot.values.cellCopyMutationCellFunction)
                    .disabledValue(parameters.baseValues.cellCopyMutationCellFunction),
                spot.values.cellCopyMutationCellFunction,
                &spot.activatedValues.cellCopyMutationCellFunction);
            AlienImGui::SliderFloat(
                AlienImGui::SliderFloatParameters()
                    .name("Cell insertion")
                    .textWidth(RightColumnWidth)
                    .min(0.0f)
                    .max(1.0f)
                    .format("%.7f")
                    .logarithmic(true)
                    .colorDependence(true)
                    .defaultValue(origSpot.values.cellCopyMutationInsertion)
                    .disabledValue(parameters.baseValues.cellCopyMutationInsertion),
                spot.values.cellCopyMutationInsertion,
                &spot.activatedValues.cellCopyMutationInsertion);
            AlienImGui::SliderFloat(
                AlienImGui::SliderFloatParameters()
                    .name("Cell deletion")
                    .textWidth(RightColumnWidth)
                    .min(0.0f)
                    .max(1.0f)
                    .format("%.7f")
                    .logarithmic(true)
                    .colorDependence(true)
                    .defaultValue(origSpot.values.cellCopyMutationDeletion)
                    .disabledValue(parameters.baseValues.cellCopyMutationDeletion),
                spot.values.cellCopyMutationDeletion,
                &spot.activatedValues.cellCopyMutationDeletion);
            AlienImGui::SliderFloat(
                AlienImGui::SliderFloatParameters()
                    .name("Translation")
                    .textWidth(RightColumnWidth)
                    .min(0.0f)
                    .max(1.0f)
                    .format("%.7f")
                    .logarithmic(true)
                    .colorDependence(true)
                    .defaultValue(origSpot.values.cellCopyMutationTranslation)
                    .disabledValue(parameters.baseValues.cellCopyMutationTranslation),
                spot.values.cellCopyMutationTranslation,
                &spot.activatedValues.cellCopyMutationTranslation);
            AlienImGui::SliderFloat(
                AlienImGui::SliderFloatParameters()
                    .name("Duplication")
                    .textWidth(RightColumnWidth)
                    .min(0.0f)
                    .max(1.0f)
                    .format("%.7f")
                    .logarithmic(true)
                    .colorDependence(true)
                    .defaultValue(origSpot.values.cellCopyMutationDuplication)
                    .disabledValue(parameters.baseValues.cellCopyMutationDuplication),
                spot.values.cellCopyMutationDuplication,
                &spot.activatedValues.cellCopyMutationDuplication);
            AlienImGui::SliderFloat(
                AlienImGui::SliderFloatParameters()
                    .name("Individual cell color")
                    .textWidth(RightColumnWidth)
                    .min(0.0f)
                    .max(1.0f)
                    .format("%.7f")
                    .logarithmic(true)
                    .colorDependence(true)
                    .defaultValue(origSpot.values.cellCopyMutationCellColor)
                    .disabledValue(parameters.baseValues.cellCopyMutationCellColor),
                spot.values.cellCopyMutationCellColor,
                &spot.activatedValues.cellCopyMutationCellColor);
            AlienImGui::SliderFloat(
                AlienImGui::SliderFloatParameters()
                    .name("Sub-genome color")
                    .textWidth(RightColumnWidth)
                    .min(0.0f)
                    .max(1.0f)
                    .format("%.7f")
                    .logarithmic(true)
                    .colorDependence(true)
                    .defaultValue(origSpot.values.cellCopyMutationSubgenomeColor)
                    .disabledValue(parameters.baseValues.cellCopyMutationSubgenomeColor),
                spot.values.cellCopyMutationSubgenomeColor,
                &spot.activatedValues.cellCopyMutationSubgenomeColor);
            AlienImGui::SliderFloat(
                AlienImGui::SliderFloatParameters()
                    .name("Genome color")
                    .textWidth(RightColumnWidth)
                    .min(0.0f)
                    .max(1.0f)
                    .format("%.7f")
                    .logarithmic(true)
                    .colorDependence(true)
                    .defaultValue(origSpot.values.cellCopyMutationGenomeColor)
                    .disabledValue(parameters.baseValues.cellCopyMutationGenomeColor),
                spot.values.cellCopyMutationGenomeColor,
                &spot.activatedValues.cellCopyMutationGenomeColor);
            AlienImGui::EndTreeNode();
        }

        /**
         * Attacker
         */
        if (AlienImGui::BeginTreeNode(AlienImGui::TreeNodeParameters().text("Cell function: Attacker"))) {
            AlienImGui::InputFloatColorMatrix(
                AlienImGui::InputFloatColorMatrixParameters()
                    .name("Food chain color matrix")
                    .max(1)
                    .textWidth(RightColumnWidth)
                    .defaultValue(toVector<MAX_COLORS, MAX_COLORS>(origSpot.values.cellFunctionAttackerFoodChainColorMatrix))
                    .disabledValue(toVector<MAX_COLORS, MAX_COLORS>(parameters.baseValues.cellFunctionAttackerFoodChainColorMatrix)),
                spot.values.cellFunctionAttackerFoodChainColorMatrix,
                &spot.activatedValues.cellFunctionAttackerFoodChainColorMatrix);
            AlienImGui::InputFloatColorMatrix(
                AlienImGui::InputFloatColorMatrixParameters()
                    .name("Complex creature protection")
                    .textWidth(RightColumnWidth)
                    .min(0)
                    .max(20.0f)
                    .defaultValue(toVector<MAX_COLORS, MAX_COLORS>(origSpot.values.cellFunctionAttackerGenomeComplexityBonus))
                    .disabledValue(toVector<MAX_COLORS, MAX_COLORS>(parameters.baseValues.cellFunctionAttackerGenomeComplexityBonus)),
                spot.values.cellFunctionAttackerGenomeComplexityBonus,
                &spot.activatedValues.cellFunctionAttackerGenomeComplexityBonus);
            AlienImGui::SliderFloat(
                AlienImGui::SliderFloatParameters()
                    .name("Energy cost")
                    .textWidth(RightColumnWidth)
                    .colorDependence(true)
                    .min(0)
                    .max(1.0f)
                    .format("%.5f")
                    .logarithmic(true)
                    .defaultValue(origSpot.values.cellFunctionAttackerEnergyCost)
                    .disabledValue(parameters.baseValues.cellFunctionAttackerEnergyCost),
                spot.values.cellFunctionAttackerEnergyCost,
                &spot.activatedValues.cellFunctionAttackerEnergyCost);
            AlienImGui::EndTreeNode();
        }

        /**
         * Addon: Advanced absorption control
         */
        if (parameters.features.advancedAbsorptionControl) {
            if (AlienImGui::BeginTreeNode(AlienImGui::TreeNodeParameters().text("Addon: Advanced energy absorption control"))) {
                AlienImGui::SliderFloat(
                    AlienImGui::SliderFloatParameters()
                        .name("Low velocity penalty")
                        .textWidth(RightColumnWidth)
                        .colorDependence(true)
                        .min(0)
                        .max(1.0f)
                        .format("%.2f")
                        .defaultValue(origSpot.values.radiationAbsorptionLowVelocityPenalty)
                        .disabledValue(parameters.baseValues.radiationAbsorptionLowVelocityPenalty),
                    spot.values.radiationAbsorptionLowVelocityPenalty,
                    &spot.activatedValues.radiationAbsorptionLowVelocityPenalty);
                AlienImGui::SliderFloat(
                    AlienImGui::SliderFloatParameters()
                        .name("Low genome complexity penalty")
                        .textWidth(RightColumnWidth)
                        .colorDependence(true)
                        .min(0)
                        .max(1.0f)
                        .format("%.2f")
                        .defaultValue(origSpot.values.radiationAbsorptionLowGenomeComplexityPenalty)
                        .disabledValue(parameters.baseValues.radiationAbsorptionLowGenomeComplexityPenalty),
                    spot.values.radiationAbsorptionLowGenomeComplexityPenalty,
                    &spot.activatedValues.radiationAbsorptionLowGenomeComplexityPenalty);
                AlienImGui::EndTreeNode();
            }
        }

        /**
         * Addon: Advanced attacker control
         */
        if (parameters.features.advancedAttackerControl) {
            if (AlienImGui::BeginTreeNode(AlienImGui::TreeNodeParameters().text("Addon: Advanced attacker control"))) {
                AlienImGui::InputFloatColorMatrix(
                    AlienImGui::InputFloatColorMatrixParameters()
                        .name("New complex mutant protection")
                        .textWidth(RightColumnWidth)
                        .min(0)
                        .max(1.0f)
                        .defaultValue(toVector<MAX_COLORS, MAX_COLORS>(origSpot.values.cellFunctionAttackerNewComplexMutantPenalty))
                        .disabledValue(toVector<MAX_COLORS, MAX_COLORS>(parameters.baseValues.cellFunctionAttackerNewComplexMutantPenalty)),
                    spot.values.cellFunctionAttackerNewComplexMutantPenalty,
                    &spot.activatedValues.cellFunctionAttackerNewComplexMutantPenalty);

                AlienImGui::SliderFloat(
                    AlienImGui::SliderFloatParameters()
                        .name("Geometry penalty")
                        .textWidth(RightColumnWidth)
                        .colorDependence(true)
                        .min(0)
                        .max(5.0f)
                        .defaultValue(origSpot.values.cellFunctionAttackerGeometryDeviationExponent)
                        .disabledValue(parameters.baseValues.cellFunctionAttackerGeometryDeviationExponent),
                    spot.values.cellFunctionAttackerGeometryDeviationExponent,
                    &spot.activatedValues.cellFunctionAttackerGeometryDeviationExponent);
                AlienImGui::SliderFloat(
                    AlienImGui::SliderFloatParameters()
                        .name("Connections mismatch penalty")
                        .textWidth(RightColumnWidth)
                        .colorDependence(true)
                        .min(0)
                        .max(1.0f)
                        .defaultValue(origSpot.values.cellFunctionAttackerConnectionsMismatchPenalty)
                        .disabledValue(parameters.baseValues.cellFunctionAttackerConnectionsMismatchPenalty),
                    spot.values.cellFunctionAttackerConnectionsMismatchPenalty,
                    &spot.activatedValues.cellFunctionAttackerConnectionsMismatchPenalty);
                AlienImGui::EndTreeNode();
            }
        }

        /**
         * Addon: Cell age limiter
         */
        if (parameters.features.cellAgeLimiter) {
            if (AlienImGui::BeginTreeNode(AlienImGui::TreeNodeParameters().text("Addon: Cell age limiter"))) {
                AlienImGui::SliderFloat(
                    AlienImGui::SliderFloatParameters()
                        .name("Maximum inactive cell age")
                        .textWidth(RightColumnWidth)
                        .colorDependence(true)
                        .min(1.0f)
                        .max(10000000.0f)
                        .logarithmic(true)
                        .infinity(true)
                        .format("%.0f")
                        .disabledValue(parameters.baseValues.cellInactiveMaxAge)
                        .defaultValue(origSpot.values.cellInactiveMaxAge),
                    spot.values.cellInactiveMaxAge,
                    &spot.activatedValues.cellInactiveMaxAge);
                AlienImGui::EndTreeNode();
            }
        }

        /**
         * Addon: Cell color transition rules
         */
        if (parameters.features.cellColorTransitionRules) {
            if (AlienImGui::BeginTreeNode(AlienImGui::TreeNodeParameters().text("Addon: Cell color transition rules"))) {
                ImGui::Checkbox("##cellColorTransition", &spot.activatedValues.cellColorTransition);
                ImGui::SameLine();
                ImGui::BeginDisabled(!spot.activatedValues.cellColorTransition);
                auto posX = ImGui::GetCursorPos().x;
                for (int color = 0; color < MAX_COLORS; ++color) {
                    ImGui::SetCursorPosX(posX);
                    ImGui::PushID(color);
                    auto parameters = AlienImGui::InputColorTransitionParameters()
                                          .textWidth(RightColumnWidth)
                                          .color(color)
                                          .defaultTargetColor(origSpot.values.cellColorTransitionTargetColor[color])
                                          .defaultTransitionAge(origSpot.values.cellColorTransitionDuration[color])
                                          .logarithmic(true)
                                          .infinity(true);
                    if (0 == color) {
                        parameters.name("Target color and duration");
                    }
                    AlienImGui::InputColorTransition(
                        parameters, color, spot.values.cellColorTransitionTargetColor[color], spot.values.cellColorTransitionDuration[color]);
                    ImGui::PopID();
                }
                ImGui::EndDisabled();
                AlienImGui::EndTreeNode();
                if (!spot.activatedValues.cellColorTransition) {
                    for (int color = 0; color < MAX_COLORS; ++color) {
                        spot.values.cellColorTransitionTargetColor[color] = parameters.baseValues.cellColorTransitionTargetColor[color];
                        spot.values.cellColorTransitionDuration[color] = parameters.baseValues.cellColorTransitionDuration[color];
                    }
                }
            }
        }
    }
    ImGui::EndChild();
    SimulationParametersValidationService::get().validateAndCorrect(parameters);

    if (spot != lastSpot) {
        auto isRunning = _simulationFacade->isSimulationRunning();
        _simulationFacade->setSimulationParameters(
            parameters, isRunning ? SimulationParametersUpdateConfig::AllExceptChangingPositions : SimulationParametersUpdateConfig::All);
    }
}

void _SimulationParametersZoneWidgets::setDefaultSpotData(SimulationParametersZone& spot) const
{
    auto worldSize = _simulationFacade->getWorldSize();

    auto maxRadius = toFloat(std::min(worldSize.x, worldSize.y)) / 2;
    if (spot.shapeType == SpotShapeType_Circular) {
        spot.shapeData.circularSpot.coreRadius = maxRadius / 3;
    } else {
        spot.shapeData.rectangularSpot.height = maxRadius / 3;
        spot.shapeData.rectangularSpot.width = maxRadius / 3;
    }
}

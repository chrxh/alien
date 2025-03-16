#include "SimulationParametersZoneWidgets.h"

#include "EngineInterface/SimulationFacade.h"
#include "EngineInterface/SimulationParametersValidationService.h"
#include "EngineInterface/SimulationParametersZone.h"

#include "AlienImGui.h"
#include "LocationHelper.h"
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

void _SimulationParametersZoneWidgets::init(SimulationFacade const& simulationFacade, int locationIndex)
{
    _simulationFacade = simulationFacade;
    _locationIndex = locationIndex;
}

void _SimulationParametersZoneWidgets::process()
{
    auto parameters = _simulationFacade->getSimulationParameters();
    auto origParameters = _simulationFacade->getOriginalSimulationParameters();

    auto zoneIndex = LocationHelper::findLocationArrayIndex(parameters, _locationIndex);
    SimulationParametersZone& zone = parameters.zone[zoneIndex];
    SimulationParametersZone const& origZone = origParameters.zone[zoneIndex];
    auto lastZone = zone;
    _zoneName = std::string(zone.name);

    auto worldSize = _simulationFacade->getWorldSize();

    /**
     * General
     */
    if (AlienImGui::BeginTreeNode(AlienImGui::TreeNodeParameters().name("General"))) {
        AlienImGui::InputText(
            AlienImGui::InputTextParameters().name("Name").textWidth(RightColumnWidth).defaultValue(origZone.name), zone.name, sizeof(Char64) / sizeof(char));

    }
    AlienImGui::EndTreeNode();

    /**
     * Visualization and location
     */
    if (AlienImGui::BeginTreeNode(AlienImGui::TreeNodeParameters().name("Visualization"))) {
        AlienImGui::ColorButtonWithPicker(
            AlienImGui::ColorButtonWithPickerParameters().name("Background color").textWidth(RightColumnWidth).defaultValue(origZone.color),
            zone.color,
            _backupColor,
            _zoneColorPalette.getReference());
    }
    AlienImGui::EndTreeNode();

    if (AlienImGui::BeginTreeNode(AlienImGui::TreeNodeParameters().name("Location"))) {
        if (AlienImGui::Switcher(
                AlienImGui::SwitcherParameters().name("Shape").values({"Circular", "Rectangular"}).textWidth(RightColumnWidth).defaultValue(origZone.shape.type),
                zone.shape.type)) {
            setDefaultSpotData(zone);
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
                .defaultValue(RealVector2D{origZone.posX, origZone.posY})
                .format("%.2f")
                .getMousePickerEnabledFunc(getMousePickerEnabledFunc)
                .setMousePickerEnabledFunc(setMousePickerEnabledFunc)
                .getMousePickerPositionFunc(getMousePickerPositionFunc),
            zone.posX,
            zone.posY);
        AlienImGui::SliderFloat2(
            AlienImGui::SliderFloat2Parameters()
                .name("Velocity (x,y)")
                .textWidth(RightColumnWidth)
                .min({-4.0f, -4.0f})
                .max({4.0f, 4.0f})
                .defaultValue(RealVector2D{origZone.velX, origZone.velY})
                .format("%.2f"),
            zone.velX,
            zone.velY);
        auto maxRadius = toFloat(std::max(worldSize.x, worldSize.y));
        if (zone.shape.type == ZoneShapeType_Circular) {
            AlienImGui::SliderFloat(
                AlienImGui::SliderFloatParameters()
                    .name("Core radius")
                    .textWidth(RightColumnWidth)
                    .min(0)
                    .max(maxRadius)
                    .defaultValue(&origZone.shape.alternatives.circularSpot.coreRadius)
                    .format("%.1f"),
                &zone.shape.alternatives.circularSpot.coreRadius);
        }
        if (zone.shape.type == ZoneShapeType_Rectangular) {
            AlienImGui::SliderFloat2(
                AlienImGui::SliderFloat2Parameters()
                    .name("Size (x,y)")
                    .textWidth(RightColumnWidth)
                    .min({0, 0})
                    .max({toFloat(worldSize.x), toFloat(worldSize.y)})
                    .defaultValue(RealVector2D{origZone.shape.alternatives.rectangularSpot.width, origZone.shape.alternatives.rectangularSpot.height})
                    .format("%.1f"),
                zone.shape.alternatives.rectangularSpot.width,
                zone.shape.alternatives.rectangularSpot.height);
        }

        AlienImGui::SliderFloat(
            AlienImGui::SliderFloatParameters()
                .name("Fade-out radius")
                .textWidth(RightColumnWidth)
                .min(0)
                .max(maxRadius)
                .defaultValue(&origZone.fadeoutRadius)
                .format("%.1f"),
            &zone.fadeoutRadius);
    }
    AlienImGui::EndTreeNode();

    /**
     * Force field
     */
    if (AlienImGui::BeginTreeNode(AlienImGui::TreeNodeParameters().name("Force field"))) {
        {
            auto forceFieldTypeIntern = std::max(0, zone.flow.type - 1);  //FlowType_None should not be selectable in ComboBox
            auto origForceFieldTypeIntern = std::max(0, origZone.flow.type - 1);

            auto isEnabled = zone.flow.type != 0;
            auto origIsEnabled = origZone.flow.type != 0;
            if (AlienImGui::Combo(
                    AlienImGui::ComboParameters()
                        .name("Field type")
                        .values({"Radial", "Central", "Linear"})
                        .textWidth(RightColumnWidth)
                        .defaultValue(origForceFieldTypeIntern)
                        .defaultEnabledValue(&origIsEnabled),
                    forceFieldTypeIntern,
                    &isEnabled)) {
                zone.flow.type = isEnabled ? forceFieldTypeIntern + 1 : FlowType_None;

                if (zone.flow.type == FlowType_Radial) {
                    zone.flow.alternatives.radialFlow = RadialFlow();
                }
                if (zone.flow.type == FlowType_Central) {
                    zone.flow.alternatives.centralFlow = CentralFlow();
                }
                if (zone.flow.type == FlowType_Linear) {
                    zone.flow.alternatives.linearFlow = LinearFlow();
                }
            }
        }

        auto isForceFieldActive = zone.flow.type != FlowType_None;

        ImGui::BeginDisabled(!isForceFieldActive);
        auto posX = ImGui::GetCursorPos().x;
        if (zone.flow.type == FlowType_Radial) {
            ImGui::SetCursorPosX(posX);
            AlienImGui::Combo(
                AlienImGui::ComboParameters()
                    .name("Orientation")
                    .textWidth(RightColumnWidth)
                    .defaultValue(origZone.flow.alternatives.radialFlow.orientation)
                    .values({"Clockwise", "Counter clockwise"}),
                zone.flow.alternatives.radialFlow.orientation);
            ImGui::SetCursorPosX(posX);
            AlienImGui::SliderFloat(
                AlienImGui::SliderFloatParameters()
                    .name("Strength")
                    .textWidth(RightColumnWidth)
                    .min(0)
                    .max(0.5f)
                    .logarithmic(true)
                    .format("%.5f")
                    .defaultValue(&origZone.flow.alternatives.radialFlow.strength),
                &zone.flow.alternatives.radialFlow.strength);
            ImGui::SetCursorPosX(posX);
            AlienImGui::SliderFloat(
                AlienImGui::SliderFloatParameters()
                    .name("Drift angle")
                    .textWidth(RightColumnWidth)
                    .min(-180.0f)
                    .max(180.0f)
                    .format("%.1f")
                    .defaultValue(&origZone.flow.alternatives.radialFlow.driftAngle),
                &zone.flow.alternatives.radialFlow.driftAngle);
        }
        if (zone.flow.type == FlowType_Central) {
            ImGui::SetCursorPosX(posX);
            AlienImGui::SliderFloat(
                AlienImGui::SliderFloatParameters()
                    .name("Strength")
                    .textWidth(RightColumnWidth)
                    .min(0)
                    .max(0.5f)
                    .logarithmic(true)
                    .format("%.5f")
                    .defaultValue(&origZone.flow.alternatives.centralFlow.strength),
                &zone.flow.alternatives.centralFlow.strength);
        }
        if (zone.flow.type == FlowType_Linear) {
            ImGui::SetCursorPosX(posX);
            AlienImGui::SliderFloat(
                AlienImGui::SliderFloatParameters()
                    .name("Angle")
                    .textWidth(RightColumnWidth)
                    .min(-180.0f)
                    .max(180.0f)
                    .format("%.1f")
                    .defaultValue(&origZone.flow.alternatives.linearFlow.angle),
                &zone.flow.alternatives.linearFlow.angle);
            ImGui::SetCursorPosX(posX);
            AlienImGui::SliderFloat(
                AlienImGui::SliderFloatParameters()
                    .name("Strength")
                    .textWidth(RightColumnWidth)
                    .min(0)
                    .max(0.5f)
                    .logarithmic(true)
                    .format("%.5f")
                    .defaultValue(&origZone.flow.alternatives.linearFlow.strength),
                &zone.flow.alternatives.linearFlow.strength);
        }
        ImGui::EndDisabled();
    }
    AlienImGui::EndTreeNode();

    /**
     * Physics: Motion
     */
    if (AlienImGui::BeginTreeNode(AlienImGui::TreeNodeParameters().name("Physics: Motion"))) {
        AlienImGui::SliderFloat(
            AlienImGui::SliderFloatParameters()
                .name("Friction")
                .textWidth(RightColumnWidth)
                .min(0)
                .max(1)
                .logarithmic(true)
                .defaultValue(&origZone.values.friction)
                .disabledValue(&parameters.baseValues.friction)
                .format("%.4f"),
            &zone.values.friction,
            &zone.activatedValues.friction);
        AlienImGui::SliderFloat(
            AlienImGui::SliderFloatParameters()
                .name("Rigidity")
                .textWidth(RightColumnWidth)
                .min(0)
                .max(1)
                .defaultValue(&origZone.values.rigidity)
                .disabledValue(&parameters.baseValues.rigidity)
                .format("%.2f"),
            &zone.values.rigidity,
            &zone.activatedValues.rigidity);
    }
    AlienImGui::EndTreeNode();

    /**
     * Physics: Thresholds
     */
    if (AlienImGui::BeginTreeNode(AlienImGui::TreeNodeParameters().name("Physics: Thresholds"))) {
        AlienImGui::SliderFloat(
            AlienImGui::SliderFloatParameters()
                .name("Maximum force")
                .textWidth(RightColumnWidth)
                .min(0)
                .max(3.0f)
                .colorDependence(true)
                .defaultValue(origZone.values.cellMaxForce)
                .disabledValue(parameters.baseValues.cellMaxForce),
            zone.values.cellMaxForce,
            &zone.activatedValues.cellMaxForce);
    }
    AlienImGui::EndTreeNode();

    /**
     * Physics: Binding
     */
    if (AlienImGui::BeginTreeNode(AlienImGui::TreeNodeParameters().name("Physics: Binding"))) {
        AlienImGui::SliderFloat(
            AlienImGui::SliderFloatParameters()
                .name("Binding creation velocity")
                .textWidth(RightColumnWidth)
                .min(0)
                .max(2.0f)
                .defaultValue(&origZone.values.cellFusionVelocity)
                .disabledValue(&parameters.baseValues.cellFusionVelocity),
            &zone.values.cellFusionVelocity,
            &zone.activatedValues.cellFusionVelocity);
        AlienImGui::SliderFloat(
            AlienImGui::SliderFloatParameters()
                .name("Maximum energy")
                .textWidth(RightColumnWidth)
                .min(50.0f)
                .max(10000000.0f)
                .logarithmic(true)
                .infinity(true)
                .format("%.0f")
                .defaultValue(&origZone.values.cellMaxBindingEnergy)
                .disabledValue(&parameters.baseValues.cellMaxBindingEnergy),
            &zone.values.cellMaxBindingEnergy,
            &zone.activatedValues.cellMaxBindingEnergy);
    }
    AlienImGui::EndTreeNode();


    /**
     * Physics: Radiation
     */
    if (AlienImGui::BeginTreeNode(AlienImGui::TreeNodeParameters().name("Physics: Radiation"))) {
        AlienImGui::Checkbox(
            AlienImGui::CheckboxParameters()
                .name("Disable radiation sources")
                .textWidth(RightColumnWidth)
                .defaultValue(origZone.values.radiationDisableSources)
                .tooltip("If activated, all radiation sources within this zone are deactivated."),
            zone.values.radiationDisableSources);
        zone.activatedValues.radiationDisableSources = zone.values.radiationDisableSources;

        AlienImGui::SliderFloat(
            AlienImGui::SliderFloatParameters()
                .name("Absorption factor")
                .textWidth(RightColumnWidth)
                .logarithmic(true)
                .colorDependence(true)
                .min(0)
                .max(1.0)
                .format("%.4f")
                .defaultValue(origZone.values.radiationAbsorption)
                .disabledValue(parameters.baseValues.radiationAbsorption),
            zone.values.radiationAbsorption,
            &zone.activatedValues.radiationAbsorption);

        AlienImGui::SliderFloat(
            AlienImGui::SliderFloatParameters()
                .name("Radiation type 1: Strength")
                .textWidth(RightColumnWidth)
                .colorDependence(true)
                .min(0)
                .max(0.01f)
                .logarithmic(true)
                .defaultValue(origZone.values.radiationCellAgeStrength)
                .disabledValue(parameters.baseValues.radiationCellAgeStrength)
                .format("%.6f"),
            zone.values.radiationCellAgeStrength,
            &zone.activatedValues.radiationCellAgeStrength);
    }
    AlienImGui::EndTreeNode();

    /**
     * Cell life cycle
     */
    if (AlienImGui::BeginTreeNode(AlienImGui::TreeNodeParameters().name("Cell life cycle"))) {
        AlienImGui::SliderFloat(
            AlienImGui::SliderFloatParameters()
                .name("Minimum energy")
                .textWidth(RightColumnWidth)
                .colorDependence(true)
                .min(10.0f)
                .max(200.0f)
                .defaultValue(origZone.values.cellMinEnergy)
                .disabledValue(parameters.baseValues.cellMinEnergy),
            zone.values.cellMinEnergy,
            &zone.activatedValues.cellMinEnergy);
        AlienImGui::SliderFloat(
            AlienImGui::SliderFloatParameters()
                .name("Decay rate of dying cells")
                .colorDependence(true)
                .textWidth(RightColumnWidth)
                .min(1e-6f)
                .max(0.1f)
                .format("%.6f")
                .logarithmic(true)
                .defaultValue(origZone.values.cellDeathProbability)
                .disabledValue(parameters.baseValues.cellDeathProbability),
            zone.values.cellDeathProbability,
            &zone.activatedValues.cellDeathProbability);
    }
    AlienImGui::EndTreeNode();

    /**
     * Mutation 
     */
    if (AlienImGui::BeginTreeNode(AlienImGui::TreeNodeParameters().name("Genome copy mutations"))) {
        AlienImGui::SliderFloat(
            AlienImGui::SliderFloatParameters()
                .name("Neuron weights and biases")
                .textWidth(RightColumnWidth)
                .min(0.0f)
                .max(1.0f)
                .format("%.7f")
                .colorDependence(true)
                .logarithmic(true)
                .defaultValue(origZone.values.cellCopyMutationNeuronData)
                .disabledValue(parameters.baseValues.cellCopyMutationNeuronData),
            zone.values.cellCopyMutationNeuronData,
            &zone.activatedValues.cellCopyMutationNeuronData);
        AlienImGui::SliderFloat(
            AlienImGui::SliderFloatParameters()
                .name("Cell properties")
                .textWidth(RightColumnWidth)
                .min(0.0f)
                .max(1.0f)
                .format("%.7f")
                .logarithmic(true)
                .colorDependence(true)
                .defaultValue(origZone.values.cellCopyMutationCellProperties)
                .disabledValue(parameters.baseValues.cellCopyMutationCellProperties),
            zone.values.cellCopyMutationCellProperties,
            &zone.activatedValues.cellCopyMutationCellProperties);
        AlienImGui::SliderFloat(
            AlienImGui::SliderFloatParameters()
                .name("Geometry")
                .textWidth(RightColumnWidth)
                .min(0.0f)
                .max(1.0f)
                .format("%.7f")
                .logarithmic(true)
                .colorDependence(true)
                .defaultValue(origZone.values.cellCopyMutationGeometry)
                .disabledValue(parameters.baseValues.cellCopyMutationGeometry),
            zone.values.cellCopyMutationGeometry,
            &zone.activatedValues.cellCopyMutationGeometry);
        AlienImGui::SliderFloat(
            AlienImGui::SliderFloatParameters()
                .name("Custom geometry")
                .textWidth(RightColumnWidth)
                .min(0.0f)
                .max(1.0f)
                .format("%.7f")
                .logarithmic(true)
                .colorDependence(true)
                .defaultValue(origZone.values.cellCopyMutationCustomGeometry)
                .disabledValue(parameters.baseValues.cellCopyMutationCustomGeometry),
            zone.values.cellCopyMutationCustomGeometry,
            &zone.activatedValues.cellCopyMutationCustomGeometry);
        AlienImGui::SliderFloat(
            AlienImGui::SliderFloatParameters()
                .name("Cell function type")
                .textWidth(RightColumnWidth)
                .min(0.0f)
                .max(1.0f)
                .format("%.7f")
                .logarithmic(true)
                .colorDependence(true)
                .defaultValue(origZone.values.cellCopyMutationCellType)
                .disabledValue(parameters.baseValues.cellCopyMutationCellType),
            zone.values.cellCopyMutationCellType,
            &zone.activatedValues.cellCopyMutationCellType);
        AlienImGui::SliderFloat(
            AlienImGui::SliderFloatParameters()
                .name("Cell insertion")
                .textWidth(RightColumnWidth)
                .min(0.0f)
                .max(1.0f)
                .format("%.7f")
                .logarithmic(true)
                .colorDependence(true)
                .defaultValue(origZone.values.cellCopyMutationInsertion)
                .disabledValue(parameters.baseValues.cellCopyMutationInsertion),
            zone.values.cellCopyMutationInsertion,
            &zone.activatedValues.cellCopyMutationInsertion);
        AlienImGui::SliderFloat(
            AlienImGui::SliderFloatParameters()
                .name("Cell deletion")
                .textWidth(RightColumnWidth)
                .min(0.0f)
                .max(1.0f)
                .format("%.7f")
                .logarithmic(true)
                .colorDependence(true)
                .defaultValue(origZone.values.cellCopyMutationDeletion)
                .disabledValue(parameters.baseValues.cellCopyMutationDeletion),
            zone.values.cellCopyMutationDeletion,
            &zone.activatedValues.cellCopyMutationDeletion);
        AlienImGui::SliderFloat(
            AlienImGui::SliderFloatParameters()
                .name("Translation")
                .textWidth(RightColumnWidth)
                .min(0.0f)
                .max(1.0f)
                .format("%.7f")
                .logarithmic(true)
                .colorDependence(true)
                .defaultValue(origZone.values.cellCopyMutationTranslation)
                .disabledValue(parameters.baseValues.cellCopyMutationTranslation),
            zone.values.cellCopyMutationTranslation,
            &zone.activatedValues.cellCopyMutationTranslation);
        AlienImGui::SliderFloat(
            AlienImGui::SliderFloatParameters()
                .name("Duplication")
                .textWidth(RightColumnWidth)
                .min(0.0f)
                .max(1.0f)
                .format("%.7f")
                .logarithmic(true)
                .colorDependence(true)
                .defaultValue(origZone.values.cellCopyMutationDuplication)
                .disabledValue(parameters.baseValues.cellCopyMutationDuplication),
            zone.values.cellCopyMutationDuplication,
            &zone.activatedValues.cellCopyMutationDuplication);
        AlienImGui::SliderFloat(
            AlienImGui::SliderFloatParameters()
                .name("Individual cell color")
                .textWidth(RightColumnWidth)
                .min(0.0f)
                .max(1.0f)
                .format("%.7f")
                .logarithmic(true)
                .colorDependence(true)
                .defaultValue(origZone.values.cellCopyMutationCellColor)
                .disabledValue(parameters.baseValues.cellCopyMutationCellColor),
            zone.values.cellCopyMutationCellColor,
            &zone.activatedValues.cellCopyMutationCellColor);
        AlienImGui::SliderFloat(
            AlienImGui::SliderFloatParameters()
                .name("Sub-genome color")
                .textWidth(RightColumnWidth)
                .min(0.0f)
                .max(1.0f)
                .format("%.7f")
                .logarithmic(true)
                .colorDependence(true)
                .defaultValue(origZone.values.cellCopyMutationSubgenomeColor)
                .disabledValue(parameters.baseValues.cellCopyMutationSubgenomeColor),
            zone.values.cellCopyMutationSubgenomeColor,
            &zone.activatedValues.cellCopyMutationSubgenomeColor);
        AlienImGui::SliderFloat(
            AlienImGui::SliderFloatParameters()
                .name("Genome color")
                .textWidth(RightColumnWidth)
                .min(0.0f)
                .max(1.0f)
                .format("%.7f")
                .logarithmic(true)
                .colorDependence(true)
                .defaultValue(origZone.values.cellCopyMutationGenomeColor)
                .disabledValue(parameters.baseValues.cellCopyMutationGenomeColor),
            zone.values.cellCopyMutationGenomeColor,
            &zone.activatedValues.cellCopyMutationGenomeColor);
    }
    AlienImGui::EndTreeNode();

    /**
     * Attacker
     */
    if (AlienImGui::BeginTreeNode(AlienImGui::TreeNodeParameters().name("Cell type: Attacker"))) {
        AlienImGui::InputFloatColorMatrix(
            AlienImGui::InputFloatColorMatrixParameters()
                .name("Food chain color matrix")
                .max(1)
                .textWidth(RightColumnWidth)
                .defaultValue(toVector<MAX_COLORS, MAX_COLORS>(origZone.values.cellTypeAttackerFoodChainColorMatrix))
                .disabledValue(toVector<MAX_COLORS, MAX_COLORS>(parameters.baseValues.cellTypeAttackerFoodChainColorMatrix)),
            zone.values.cellTypeAttackerFoodChainColorMatrix,
            &zone.activatedValues.cellTypeAttackerFoodChainColorMatrix);
        AlienImGui::InputFloatColorMatrix(
            AlienImGui::InputFloatColorMatrixParameters()
                .name("Complex creature protection")
                .textWidth(RightColumnWidth)
                .min(0)
                .max(20.0f)
                .defaultValue(toVector<MAX_COLORS, MAX_COLORS>(origZone.values.cellTypeAttackerGenomeComplexityBonus))
                .disabledValue(toVector<MAX_COLORS, MAX_COLORS>(parameters.baseValues.cellTypeAttackerGenomeComplexityBonus)),
            zone.values.cellTypeAttackerGenomeComplexityBonus,
            &zone.activatedValues.cellTypeAttackerGenomeComplexityBonus);
        AlienImGui::SliderFloat(
            AlienImGui::SliderFloatParameters()
                .name("Energy cost")
                .textWidth(RightColumnWidth)
                .colorDependence(true)
                .min(0)
                .max(1.0f)
                .format("%.5f")
                .logarithmic(true)
                .defaultValue(origZone.values.cellTypeAttackerEnergyCost)
                .disabledValue(parameters.baseValues.cellTypeAttackerEnergyCost),
            zone.values.cellTypeAttackerEnergyCost,
            &zone.activatedValues.cellTypeAttackerEnergyCost);
    }
    AlienImGui::EndTreeNode();

    /**
     * Expert settings: Advanced absorption control
     */
    if (AlienImGui::BeginTreeNode(AlienImGui::TreeNodeParameters()
                                      .name("Expert settings: Advanced energy absorption control")
                                      .visible(parameters.features.advancedAbsorptionControl)
                                      .blinkWhenActivated(true))) {
        AlienImGui::SliderFloat(
            AlienImGui::SliderFloatParameters()
                .name("Low velocity penalty")
                .textWidth(RightColumnWidth)
                .colorDependence(true)
                .min(0)
                .max(1.0f)
                .format("%.2f")
                .defaultValue(origZone.values.radiationAbsorptionLowVelocityPenalty)
                .disabledValue(parameters.baseValues.radiationAbsorptionLowVelocityPenalty),
            zone.values.radiationAbsorptionLowVelocityPenalty,
            &zone.activatedValues.radiationAbsorptionLowVelocityPenalty);
        AlienImGui::SliderFloat(
            AlienImGui::SliderFloatParameters()
                .name("Low genome complexity penalty")
                .textWidth(RightColumnWidth)
                .colorDependence(true)
                .min(0)
                .max(1.0f)
                .format("%.2f")
                .defaultValue(origZone.values.radiationAbsorptionLowGenomeComplexityPenalty)
                .disabledValue(parameters.baseValues.radiationAbsorptionLowGenomeComplexityPenalty),
            zone.values.radiationAbsorptionLowGenomeComplexityPenalty,
            &zone.activatedValues.radiationAbsorptionLowGenomeComplexityPenalty);
    }
    AlienImGui::EndTreeNode();

    /**
     * Expert settings: Advanced attacker control
     */
    if (AlienImGui::BeginTreeNode(AlienImGui::TreeNodeParameters()
                                      .name("Expert settings: Advanced attacker control")
                                      .visible(parameters.features.advancedAttackerControl)
                                      .blinkWhenActivated(true))) {
        AlienImGui::InputFloatColorMatrix(
            AlienImGui::InputFloatColorMatrixParameters()
                .name("New complex mutant protection")
                .textWidth(RightColumnWidth)
                .min(0)
                .max(1.0f)
                .defaultValue(toVector<MAX_COLORS, MAX_COLORS>(origZone.values.cellTypeAttackerNewComplexMutantPenalty))
                .disabledValue(toVector<MAX_COLORS, MAX_COLORS>(parameters.baseValues.cellTypeAttackerNewComplexMutantPenalty)),
            zone.values.cellTypeAttackerNewComplexMutantPenalty,
            &zone.activatedValues.cellTypeAttackerNewComplexMutantPenalty);

        AlienImGui::SliderFloat(
            AlienImGui::SliderFloatParameters()
                .name("Geometry penalty")
                .textWidth(RightColumnWidth)
                .colorDependence(true)
                .min(0)
                .max(5.0f)
                .defaultValue(origZone.values.cellTypeAttackerGeometryDeviationExponent)
                .disabledValue(parameters.baseValues.cellTypeAttackerGeometryDeviationExponent),
            zone.values.cellTypeAttackerGeometryDeviationExponent,
            &zone.activatedValues.cellTypeAttackerGeometryDeviationExponent);
        AlienImGui::SliderFloat(
            AlienImGui::SliderFloatParameters()
                .name("Connections mismatch penalty")
                .textWidth(RightColumnWidth)
                .colorDependence(true)
                .min(0)
                .max(1.0f)
                .defaultValue(origZone.values.cellTypeAttackerConnectionsMismatchPenalty)
                .disabledValue(parameters.baseValues.cellTypeAttackerConnectionsMismatchPenalty),
            zone.values.cellTypeAttackerConnectionsMismatchPenalty,
            &zone.activatedValues.cellTypeAttackerConnectionsMismatchPenalty);
    }
    AlienImGui::EndTreeNode();

    /**
     * Expert settings: Cell age limiter
     */
    if (AlienImGui::BeginTreeNode(AlienImGui::TreeNodeParameters()
                                      .name("Expert settings: Cell age limiter")
                                      .visible(parameters.features.cellAgeLimiter)
                                      .blinkWhenActivated(true))) {
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
                .disabledValue(parameters.baseValues.maxAgeForInactiveCells)
                .defaultValue(origZone.values.maxAgeForInactiveCells),
            zone.values.maxAgeForInactiveCells,
            &zone.activatedValues.cellInactiveMaxAge);
    }
    AlienImGui::EndTreeNode();

    /**
     * Expert settings: Cell color transition rules
     */
    if (AlienImGui::BeginTreeNode(AlienImGui::TreeNodeParameters()
                                      .name("Expert settings: Cell color transition rules")
                                      .visible(parameters.features.cellColorTransitionRules)
                                      .blinkWhenActivated(true))) {
        ImGui::Checkbox("##cellColorTransition", &zone.activatedValues.cellColorTransition);
        ImGui::SameLine();
        ImGui::BeginDisabled(!zone.activatedValues.cellColorTransition);
        auto posX = ImGui::GetCursorPos().x;
        for (int color = 0; color < MAX_COLORS; ++color) {
            ImGui::SetCursorPosX(posX);
            ImGui::PushID(color);
            auto parameters = AlienImGui::InputColorTransitionParameters()
                                  .textWidth(RightColumnWidth)
                                  .color(color)
                                  .defaultTargetColor(origZone.values.cellColorTransitionTargetColor[color])
                                  .defaultTransitionAge(origZone.values.cellColorTransitionDuration[color])
                                  .logarithmic(true)
                                  .infinity(true);
            if (0 == color) {
                parameters.name("Target color and duration");
            }
            AlienImGui::InputColorTransition(
                parameters, color, zone.values.cellColorTransitionTargetColor[color], zone.values.cellColorTransitionDuration[color]);
            ImGui::PopID();
        }
        ImGui::EndDisabled();
        if (!zone.activatedValues.cellColorTransition) {
            for (int color = 0; color < MAX_COLORS; ++color) {
                zone.values.cellColorTransitionTargetColor[color] = parameters.baseValues.cellColorTransitionTargetColor[color];
                zone.values.cellColorTransitionDuration[color] = parameters.baseValues.cellColorTransitionDuration[color];
            }
        }
    }
    AlienImGui::EndTreeNode();

    SimulationParametersValidationService::get().validateAndCorrect(zone, parameters);

    if (zone != lastZone) {
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

void _SimulationParametersZoneWidgets::setDefaultSpotData(SimulationParametersZone& spot) const
{
    auto worldSize = _simulationFacade->getWorldSize();

    auto maxRadius = toFloat(std::min(worldSize.x, worldSize.y)) / 2;
    if (spot.shape.type == ZoneShapeType_Circular) {
        spot.shape.alternatives.circularSpot.coreRadius = maxRadius / 3;
    } else {
        spot.shape.alternatives.rectangularSpot.height = maxRadius / 3;
        spot.shape.alternatives.rectangularSpot.width = maxRadius / 3;
    }
}

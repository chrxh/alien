#include "SimulationParametersZoneWidgets.h"

#include "EngineInterface/LocationHelper.h"
#include "EngineInterface/SimulationFacade.h"
#include "EngineInterface/ParametersValidationService.h"
#include "EngineInterface/SimulationParametersZone.h"

#include "AlienImGui.h"
#include "LoginDialog.h"
#include "SpecificationGuiService.h"
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

    SpecificationGuiService::get().createWidgetsForParameters(parameters, origParameters, _locationIndex);
    AlienImGui::Separator();
    AlienImGui::Separator();
    AlienImGui::Separator();

    /**
     * General
     */
    if (AlienImGui::BeginTreeNode(AlienImGui::TreeNodeParameters().name("General"))) {
        AlienImGui::InputText(
            AlienImGui::InputTextParameters().name("Name").textWidth(RightColumnWidth).defaultValue(origZone.name), zone.name, sizeof(Char64) / sizeof(char));

    }
    AlienImGui::EndTreeNode();

    /**
     * Location
     */
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
            &zone.enabledValues.friction);
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
            &zone.enabledValues.rigidity);
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
            &zone.enabledValues.cellMaxForce);
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
            &zone.enabledValues.cellFusionVelocity);
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
            &zone.enabledValues.cellMaxBindingEnergy);
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
        zone.enabledValues.radiationDisableSources = zone.values.radiationDisableSources;

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
            &zone.enabledValues.radiationAbsorption);

        AlienImGui::SliderFloat(
            AlienImGui::SliderFloatParameters()
                .name("Radiation type 1: Strength")
                .textWidth(RightColumnWidth)
                .colorDependence(true)
                .min(0)
                .max(0.01f)
                .logarithmic(true)
                .defaultValue(origZone.values.radiationType1_strength)
                .disabledValue(parameters.baseValues.radiationType1_strength)
                .format("%.6f"),
            zone.values.radiationType1_strength,
            &zone.enabledValues.radiationType1_strength);
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
                .defaultValue(origZone.values.minCellEnergy)
                .disabledValue(parameters.baseValues.minCellEnergy),
            zone.values.minCellEnergy,
            &zone.enabledValues.minCellEnergy);
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
            &zone.enabledValues.cellDeathProbability);
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
                .defaultValue(origZone.values.copyMutationNeuronData)
                .disabledValue(parameters.baseValues.copyMutationNeuronData),
            zone.values.copyMutationNeuronData,
            &zone.enabledValues.copyMutationNeuronData);
        AlienImGui::SliderFloat(
            AlienImGui::SliderFloatParameters()
                .name("Cell properties")
                .textWidth(RightColumnWidth)
                .min(0.0f)
                .max(1.0f)
                .format("%.7f")
                .logarithmic(true)
                .colorDependence(true)
                .defaultValue(origZone.values.copyMutationCellProperties)
                .disabledValue(parameters.baseValues.copyMutationCellProperties),
            zone.values.copyMutationCellProperties,
            &zone.enabledValues.copyMutationCellProperties);
        AlienImGui::SliderFloat(
            AlienImGui::SliderFloatParameters()
                .name("Geometry")
                .textWidth(RightColumnWidth)
                .min(0.0f)
                .max(1.0f)
                .format("%.7f")
                .logarithmic(true)
                .colorDependence(true)
                .defaultValue(origZone.values.copyMutationGeometry)
                .disabledValue(parameters.baseValues.copyMutationGeometry),
            zone.values.copyMutationGeometry,
            &zone.enabledValues.copyMutationGeometry);
        AlienImGui::SliderFloat(
            AlienImGui::SliderFloatParameters()
                .name("Custom geometry")
                .textWidth(RightColumnWidth)
                .min(0.0f)
                .max(1.0f)
                .format("%.7f")
                .logarithmic(true)
                .colorDependence(true)
                .defaultValue(origZone.values.copyMutationCustomGeometry)
                .disabledValue(parameters.baseValues.copyMutationCustomGeometry),
            zone.values.copyMutationCustomGeometry,
            &zone.enabledValues.copyMutationCustomGeometry);
        AlienImGui::SliderFloat(
            AlienImGui::SliderFloatParameters()
                .name("Cell function type")
                .textWidth(RightColumnWidth)
                .min(0.0f)
                .max(1.0f)
                .format("%.7f")
                .logarithmic(true)
                .colorDependence(true)
                .defaultValue(origZone.values.copyMutationCellType)
                .disabledValue(parameters.baseValues.copyMutationCellType),
            zone.values.copyMutationCellType,
            &zone.enabledValues.copyMutationCellType);
        AlienImGui::SliderFloat(
            AlienImGui::SliderFloatParameters()
                .name("Cell insertion")
                .textWidth(RightColumnWidth)
                .min(0.0f)
                .max(1.0f)
                .format("%.7f")
                .logarithmic(true)
                .colorDependence(true)
                .defaultValue(origZone.values.copyMutationInsertion)
                .disabledValue(parameters.baseValues.copyMutationInsertion),
            zone.values.copyMutationInsertion,
            &zone.enabledValues.copyMutationInsertion);
        AlienImGui::SliderFloat(
            AlienImGui::SliderFloatParameters()
                .name("Cell deletion")
                .textWidth(RightColumnWidth)
                .min(0.0f)
                .max(1.0f)
                .format("%.7f")
                .logarithmic(true)
                .colorDependence(true)
                .defaultValue(origZone.values.copyMutationDeletion)
                .disabledValue(parameters.baseValues.copyMutationDeletion),
            zone.values.copyMutationDeletion,
            &zone.enabledValues.copyMutationDeletion);
        AlienImGui::SliderFloat(
            AlienImGui::SliderFloatParameters()
                .name("Translation")
                .textWidth(RightColumnWidth)
                .min(0.0f)
                .max(1.0f)
                .format("%.7f")
                .logarithmic(true)
                .colorDependence(true)
                .defaultValue(origZone.values.copyMutationTranslation)
                .disabledValue(parameters.baseValues.copyMutationTranslation),
            zone.values.copyMutationTranslation,
            &zone.enabledValues.copyMutationTranslation);
        AlienImGui::SliderFloat(
            AlienImGui::SliderFloatParameters()
                .name("Duplication")
                .textWidth(RightColumnWidth)
                .min(0.0f)
                .max(1.0f)
                .format("%.7f")
                .logarithmic(true)
                .colorDependence(true)
                .defaultValue(origZone.values.copyMutationDuplication)
                .disabledValue(parameters.baseValues.copyMutationDuplication),
            zone.values.copyMutationDuplication,
            &zone.enabledValues.copyMutationDuplication);
        AlienImGui::SliderFloat(
            AlienImGui::SliderFloatParameters()
                .name("Individual cell color")
                .textWidth(RightColumnWidth)
                .min(0.0f)
                .max(1.0f)
                .format("%.7f")
                .logarithmic(true)
                .colorDependence(true)
                .defaultValue(origZone.values.copyMutationCellColor)
                .disabledValue(parameters.baseValues.copyMutationCellColor),
            zone.values.copyMutationCellColor,
            &zone.enabledValues.copyMutationCellColor);
        AlienImGui::SliderFloat(
            AlienImGui::SliderFloatParameters()
                .name("Sub-genome color")
                .textWidth(RightColumnWidth)
                .min(0.0f)
                .max(1.0f)
                .format("%.7f")
                .logarithmic(true)
                .colorDependence(true)
                .defaultValue(origZone.values.copyMutationSubgenomeColor)
                .disabledValue(parameters.baseValues.copyMutationSubgenomeColor),
            zone.values.copyMutationSubgenomeColor,
            &zone.enabledValues.copyMutationSubgenomeColor);
        AlienImGui::SliderFloat(
            AlienImGui::SliderFloatParameters()
                .name("Genome color")
                .textWidth(RightColumnWidth)
                .min(0.0f)
                .max(1.0f)
                .format("%.7f")
                .logarithmic(true)
                .colorDependence(true)
                .defaultValue(origZone.values.copyMutationGenomeColor)
                .disabledValue(parameters.baseValues.copyMutationGenomeColor),
            zone.values.copyMutationGenomeColor,
            &zone.enabledValues.copyMutationGenomeColor);
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
                .defaultValue(toVector<MAX_COLORS, MAX_COLORS>(origZone.values.attackerFoodChainColorMatrix))
                .disabledValue(toVector<MAX_COLORS, MAX_COLORS>(parameters.baseValues.attackerFoodChainColorMatrix)),
            zone.values.attackerFoodChainColorMatrix,
            &zone.enabledValues.attackerFoodChainColorMatrix);
        AlienImGui::InputFloatColorMatrix(
            AlienImGui::InputFloatColorMatrixParameters()
                .name("Complex creature protection")
                .textWidth(RightColumnWidth)
                .min(0)
                .max(20.0f)
                .defaultValue(toVector<MAX_COLORS, MAX_COLORS>(origZone.values.attackerComplexCreatureProtection))
                .disabledValue(toVector<MAX_COLORS, MAX_COLORS>(parameters.baseValues.attackerComplexCreatureProtection)),
            zone.values.attackerComplexCreatureProtection,
            &zone.enabledValues.attackerComplexCreatureProtection);
        AlienImGui::SliderFloat(
            AlienImGui::SliderFloatParameters()
                .name("Energy cost")
                .textWidth(RightColumnWidth)
                .colorDependence(true)
                .min(0)
                .max(1.0f)
                .format("%.5f")
                .logarithmic(true)
                .defaultValue(origZone.values.attackerEnergyCost)
                .disabledValue(parameters.baseValues.attackerEnergyCost),
            zone.values.attackerEnergyCost,
            &zone.enabledValues.attackerEnergyCost);
    }
    AlienImGui::EndTreeNode();

    /**
     * Expert settings: Advanced absorption control
     */
    if (AlienImGui::BeginTreeNode(AlienImGui::TreeNodeParameters()
                                      .name("Expert settings: Advanced energy absorption control")
                                      .visible(parameters.expertToggles.advancedAbsorptionControl)
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
            &zone.enabledValues.radiationAbsorptionLowVelocityPenalty);
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
            &zone.enabledValues.radiationAbsorptionLowGenomeComplexityPenalty);
    }
    AlienImGui::EndTreeNode();

    /**
     * Expert settings: Advanced attacker control
     */
    if (AlienImGui::BeginTreeNode(AlienImGui::TreeNodeParameters()
                                      .name("Expert settings: Advanced attacker control")
                                      .visible(parameters.expertToggles.advancedAttackerControl)
                                      .blinkWhenActivated(true))) {
        AlienImGui::InputFloatColorMatrix(
            AlienImGui::InputFloatColorMatrixParameters()
                .name("New complex mutant protection")
                .textWidth(RightColumnWidth)
                .min(0)
                .max(1.0f)
                .defaultValue(toVector<MAX_COLORS, MAX_COLORS>(origZone.values.attackerNewComplexMutantProtection))
                .disabledValue(toVector<MAX_COLORS, MAX_COLORS>(parameters.baseValues.attackerNewComplexMutantProtection)),
            zone.values.attackerNewComplexMutantProtection,
            &zone.enabledValues.attackerNewComplexMutantProtection);

        AlienImGui::SliderFloat(
            AlienImGui::SliderFloatParameters()
                .name("Geometry penalty")
                .textWidth(RightColumnWidth)
                .colorDependence(true)
                .min(0)
                .max(5.0f)
                .defaultValue(origZone.values.attackerGeometryDeviationProtection)
                .disabledValue(parameters.baseValues.attackerGeometryDeviationProtection),
            zone.values.attackerGeometryDeviationProtection,
            &zone.enabledValues.attackerGeometryDeviationProtection);
        AlienImGui::SliderFloat(
            AlienImGui::SliderFloatParameters()
                .name("Connections mismatch penalty")
                .textWidth(RightColumnWidth)
                .colorDependence(true)
                .min(0)
                .max(1.0f)
                .defaultValue(origZone.values.attackerConnectionsMismatchProtection)
                .disabledValue(parameters.baseValues.attackerConnectionsMismatchProtection),
            zone.values.attackerConnectionsMismatchProtection,
            &zone.enabledValues.attackerConnectionsMismatchProtection);
    }
    AlienImGui::EndTreeNode();

    /**
     * Expert settings: Cell age limiter
     */
    if (AlienImGui::BeginTreeNode(AlienImGui::TreeNodeParameters()
                                      .name("Expert settings: Cell age limiter")
                                      .visible(parameters.expertToggles.cellAgeLimiter)
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
            &zone.enabledValues.maxAgeForInactiveCellsEnabled);
    }
    AlienImGui::EndTreeNode();

    /**
     * Expert settings: Cell color transition rules
     */
    if (AlienImGui::BeginTreeNode(AlienImGui::TreeNodeParameters()
                                      .name("Expert settings: Cell color transition rules")
                                      .visible(parameters.expertToggles.cellColorTransitionRules)
                                      .blinkWhenActivated(true))) {
        ImGui::Checkbox("##cellColorTransition", &zone.enabledValues.colorTransitionRules);
        ImGui::SameLine();
        ImGui::BeginDisabled(!zone.enabledValues.colorTransitionRules);
        auto posX = ImGui::GetCursorPos().x;
        for (int color = 0; color < MAX_COLORS; ++color) {
            ImGui::SetCursorPosX(posX);
            ImGui::PushID(color);
            auto parameters = AlienImGui::InputColorTransitionParameters()
                                  .textWidth(RightColumnWidth)
                                  .color(color)
                                  .defaultTargetColor(origZone.values.colorTransitionRules.cellColorTransitionTargetColor[color])
                                  .defaultTransitionAge(origZone.values.colorTransitionRules.cellColorTransitionDuration[color])
                                  .logarithmic(true)
                                  .infinity(true);
            if (0 == color) {
                parameters.name("Target color and duration");
            }
            AlienImGui::InputColorTransition(
                parameters,
                color,
                zone.values.colorTransitionRules.cellColorTransitionTargetColor[color],
                zone.values.colorTransitionRules.cellColorTransitionDuration[color]);
            ImGui::PopID();
        }
        ImGui::EndDisabled();
        if (!zone.enabledValues.colorTransitionRules) {
            for (int color = 0; color < MAX_COLORS; ++color) {
                zone.values.colorTransitionRules.cellColorTransitionTargetColor[color] =
                    parameters.baseValues.colorTransitionRules.cellColorTransitionTargetColor[color];
                zone.values.colorTransitionRules.cellColorTransitionDuration[color] =
                    parameters.baseValues.colorTransitionRules.cellColorTransitionDuration[color];
            }
        }
    }
    AlienImGui::EndTreeNode();

    ParametersValidationService::get().validateAndCorrect(zone, parameters);

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

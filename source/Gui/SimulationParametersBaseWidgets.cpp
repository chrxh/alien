#include "SimulationParametersBaseWidgets.h"

#include <imgui.h>

#include "EngineInterface/Descriptions.h"
#include "EngineInterface/SimulationFacade.h"
#include "EngineInterface/SimulationParametersEditService.h"
#include "EngineInterface/SimulationParametersTypes.h"
#include "EngineInterface/SimulationParametersUpdateConfig.h"
#include "EngineInterface/SimulationParametersValidationService.h"
#include "EngineInterface/SimulationParametersSpecificationService.h"
#include "EngineInterface/CellTypeStrings.h"

#include "AlienImGui.h"
#include "HelpStrings.h"
#include "ParametersSpecGuiService.h"
#include "SimulationParametersMainWindow.h"

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

void _SimulationParametersBaseWidgets::init(SimulationFacade const& simulationFacade)
{
    _simulationFacade = simulationFacade;
    for (int i = 0; i < CellType_Count; ++i) {
        _cellTypeStrings.emplace_back(Const::CellTypeToStringMap.at(i));
    }
    _parametersSpecs = SimulationParametersSpecificationService::get().createParametersSpec();
}

void _SimulationParametersBaseWidgets::process()
{
    auto parameters = _simulationFacade->getSimulationParameters();
    auto origParameters = _simulationFacade->getOriginalSimulationParameters();
    auto lastParameters = parameters;

    ParametersSpecGuiService::get().createWidgetsFromSpec(_parametersSpecs, 0, parameters, origParameters);

    AlienImGui::Separator();
    AlienImGui::Separator();
    AlienImGui::Separator();

    /**
     * Mutations
     */
    if (AlienImGui::BeginTreeNode(AlienImGui::TreeNodeParameters().name("Genome copy mutations"))) {
        //----
        AlienImGui::CheckboxColorMatrix(
            AlienImGui::CheckboxColorMatrixParameters()
                .name("Color transitions")
                .textWidth(RightColumnWidth)
                .defaultValue(toVector<MAX_COLORS, MAX_COLORS>(origParameters.copyMutationColorTransitions))
                .tooltip("The color transitions are used for color mutations. The row index indicates the source color and the column index the target "
                         "color."),
            parameters.copyMutationColorTransitions);
        AlienImGui::Checkbox(
            AlienImGui::CheckboxParameters()
                .name("Prevent genome depth increase")
                .textWidth(RightColumnWidth)
                .defaultValue(origParameters.copyMutationPreventDepthIncrease)
                .tooltip(std::string("A genome has a tree-like structure because it can contain sub-genomes. If this flag is activated, the mutations will "
                                     "not increase the depth of the genome structure.")),
            parameters.copyMutationPreventDepthIncrease);
        auto preserveSelfReplication = !parameters.copyMutationSelfReplication;
        AlienImGui::Checkbox(
            AlienImGui::CheckboxParameters()
                .name("Preserve self-replication")
                .textWidth(RightColumnWidth)
                .defaultValue(!origParameters.copyMutationSelfReplication)
                .tooltip("If deactivated, a mutation can also alter self-replication capabilities in the genome by changing a constructor cell to "
                         "something else or vice versa."),
            preserveSelfReplication);
        parameters.copyMutationSelfReplication = !preserveSelfReplication;
    }
    AlienImGui::EndTreeNode();

    /**
     * Attacker
     */
    ImGui::PushID("Attacker");
    if (AlienImGui::BeginTreeNode(AlienImGui::TreeNodeParameters().name("Cell type: Attacker"))) {
        AlienImGui::InputFloatColorMatrix(
            AlienImGui::InputFloatColorMatrixParameters()
                .name("Food chain color matrix")
                .max(1)
                .textWidth(RightColumnWidth)
                .tooltip("This matrix can be used to determine how well one cell can attack another cell. The color of the attacking cell correspond to the "
                         "row "
                         "number and the color of the attacked cell to the column number. A value of 0 means that the attacked cell cannot be digested, "
                         "i.e. no energy can be obtained. A value of 1 means that the maximum energy can be obtained in the digestion process.\n\nExample: "
                         "If a "
                         "zero is entered in row 2 (red) and column 3 (green), it means that red cells cannot eat green cells.")
                .defaultValue(toVector<MAX_COLORS, MAX_COLORS>(origParameters.baseValues.cellTypeAttackerFoodChainColorMatrix)),
            parameters.baseValues.cellTypeAttackerFoodChainColorMatrix);
        AlienImGui::SliderFloat(
            AlienImGui::SliderFloatParameters()
                .name("Attack strength")
                .textWidth(RightColumnWidth)
                .colorDependence(true)
                .logarithmic(true)
                .min(0)
                .max(0.5f)
                .defaultValue(origParameters.attackerStrength)
                .tooltip("Indicates the portion of energy through which a successfully attacked cell is weakened. However, this energy portion can be "
                         "influenced by other factors adjustable within the attacker's simulation parameters."),
            parameters.attackerStrength);
        AlienImGui::SliderFloat(
            AlienImGui::SliderFloatParameters()
                .name("Attack radius")
                .textWidth(RightColumnWidth)
                .colorDependence(true)
                .min(0)
                .max(3.0f)
                .defaultValue(origParameters.attackerRadius)
                .tooltip("The maximum distance over which an attacker cell can attack another cell."),
            parameters.attackerRadius);
        AlienImGui::InputFloatColorMatrix(
            AlienImGui::InputFloatColorMatrixParameters()
                .name("Complex creature protection")
                .textWidth(RightColumnWidth)
                .min(0)
                .max(20.0f)
                .defaultValue(toVector<MAX_COLORS, MAX_COLORS>(origParameters.baseValues.cellTypeAttackerGenomeComplexityBonus))
                .tooltip("The larger this parameter is, the less energy can be gained by attacking creatures with more complex genomes."),
            parameters.baseValues.cellTypeAttackerGenomeComplexityBonus);
        AlienImGui::SliderFloat(
            AlienImGui::SliderFloatParameters()
                .name("Energy cost")
                .textWidth(RightColumnWidth)
                .colorDependence(true)
                .min(0)
                .max(1.0f)
                .format("%.5f")
                .logarithmic(true)
                .defaultValue(origParameters.baseValues.cellTypeAttackerEnergyCost)
                .tooltip("Amount of energy lost by an attempted attack of a cell in form of emitted energy particles."),
            parameters.baseValues.cellTypeAttackerEnergyCost);
        AlienImGui::Checkbox(
            AlienImGui::CheckboxParameters()
                .name("Destroy cells")
                .textWidth(RightColumnWidth)
                .defaultValue(origParameters.attackerDestroyCells)
                .tooltip("If activated, the attacker cell is able to destroy other cells. If deactivated, it only damages them."),
            parameters.attackerDestroyCells);
    }
    AlienImGui::EndTreeNode();
    ImGui::PopID();

    /**
     * Constructor
     */
    if (AlienImGui::BeginTreeNode(AlienImGui::TreeNodeParameters().name("Cell type: Constructor"))) {
        AlienImGui::SliderFloat(
            AlienImGui::SliderFloatParameters()
                .name("Connection distance")
                .textWidth(RightColumnWidth)
                .colorDependence(true)
                .min(0.1f)
                .max(3.0f)
                .defaultValue(origParameters.constructorConnectingCellDistance)
                .tooltip("The constructor can automatically connect constructed cells to other cells in the vicinity within this distance."),
            parameters.constructorConnectingCellDistance);
        AlienImGui::Checkbox(
            AlienImGui::CheckboxParameters()
                .name("Completeness check")
                .textWidth(RightColumnWidth)
                .defaultValue(origParameters.constructorCompletenessCheck)
                .tooltip("If activated, a self-replication process can only start when all other non-self-replicating constructors in the cell network are "
                         "finished."),
            parameters.constructorCompletenessCheck);
    }
    AlienImGui::EndTreeNode();

    /**
     * Defender
     */
    if (AlienImGui::BeginTreeNode(AlienImGui::TreeNodeParameters().name("Cell type: Defender"))) {
        AlienImGui::SliderFloat(
            AlienImGui::SliderFloatParameters()
                .name("Anti-attacker strength")
                .textWidth(RightColumnWidth)
                .colorDependence(true)
                .min(0.0f)
                .max(5.0f)
                .defaultValue(origParameters.defenderAntiAttackerStrength)
                .tooltip("If an attacked cell is connected to defender cells or itself a defender cell the attack strength is reduced by this factor."),
            parameters.defenderAntiAttackerStrength);
        AlienImGui::SliderFloat(
            AlienImGui::SliderFloatParameters()
                .name("Anti-injector strength")
                .textWidth(RightColumnWidth)
                .colorDependence(true)
                .min(0.0f)
                .max(5.0f)
                .defaultValue(origParameters.defenderAntiInjectorStrength)
                .tooltip("If a constructor cell is attacked by an injector and connected to defender cells, the injection duration is increased by this "
                         "factor."),
            parameters.defenderAntiInjectorStrength);
    }
    AlienImGui::EndTreeNode();

    /**
     * Injector
     */
    if (AlienImGui::BeginTreeNode(AlienImGui::TreeNodeParameters().name("Cell type: Injector"))) {
        AlienImGui::SliderFloat(
            AlienImGui::SliderFloatParameters()
                .name("Injection radius")
                .textWidth(RightColumnWidth)
                .colorDependence(true)
                .min(0.1f)
                .max(4.0f)
                .defaultValue(origParameters.injectorInjectionRadius)
                .tooltip("The maximum distance over which an injector cell can infect another cell."),
            parameters.injectorInjectionRadius);
        AlienImGui::InputIntColorMatrix(
            AlienImGui::InputIntColorMatrixParameters()
                .name("Injection time")
                .logarithmic(true)
                .max(100000)
                .textWidth(RightColumnWidth)
                .defaultValue(toVector<MAX_COLORS, MAX_COLORS>(origParameters.injectorInjectionTime))
                .tooltip("The number of activations an injector cell requires to infect another cell. One activation usually takes 6 time steps. The row "
                         "number determines the color of the injector cell, while the column number corresponds to the color of the infected cell."),
            parameters.injectorInjectionTime);
    }
    AlienImGui::EndTreeNode();

    /**
     * Muscle
     */
    if (AlienImGui::BeginTreeNode(AlienImGui::TreeNodeParameters().name("Cell type: Muscle"))) {
        AlienImGui::SliderFloat(
            AlienImGui::SliderFloatParameters()
                .name("Energy cost")
                .textWidth(RightColumnWidth)
                .colorDependence(true)
                .min(0)
                .max(5.0f)
                .format("%.5f")
                .logarithmic(true)
                .defaultValue(origParameters.muscleEnergyCost)
                .tooltip("Amount of energy lost by a muscle action of a cell in form of emitted energy particles."),
            parameters.muscleEnergyCost);
        AlienImGui::SliderFloat(
            AlienImGui::SliderFloatParameters()
                .name("Movement acceleration")
                .textWidth(RightColumnWidth)
                .colorDependence(true)
                .min(0)
                .max(10.0f)
                .logarithmic(true)
                .defaultValue(origParameters.muscleMovementAcceleration)
                .tooltip("The maximum value by which a muscle cell can modify its velocity during activation. This parameter applies only to muscle cells "
                         "which are in movement mode."),
            parameters.muscleMovementAcceleration);
        AlienImGui::SliderFloat(
            AlienImGui::SliderFloatParameters()
                .name("Crawling acceleration")
                .textWidth(RightColumnWidth)
                .colorDependence(true)
                .min(0)
                .max(10.0f)
                .logarithmic(true)
                .defaultValue(origParameters.muscleCrawlingAcceleration)
                .tooltip("The maximum length that a muscle cell can shorten or lengthen a cell connection. This parameter applies only to muscle cells "
                         "which are in contraction/expansion mode."),
            parameters.muscleCrawlingAcceleration);
        AlienImGui::SliderFloat(
            AlienImGui::SliderFloatParameters()
                .name("Bending acceleration")
                .textWidth(RightColumnWidth)
                .colorDependence(true)
                .min(0)
                .max(10.0f)
                .logarithmic(true)
                .defaultValue(origParameters.muscleBendingAcceleration)
                .tooltip("The maximum value by which a muscle cell can modify its velocity during a bending action. This parameter applies "
                         "only to muscle cells which are in bending mode."),
            parameters.muscleBendingAcceleration);
    }
    AlienImGui::EndTreeNode();

    /**
     * Sensor
     */
    if (AlienImGui::BeginTreeNode(AlienImGui::TreeNodeParameters().name("Cell type: Sensor"))) {
        AlienImGui::SliderFloat(
            AlienImGui::SliderFloatParameters()
                .name("Radius")
                .textWidth(RightColumnWidth)
                .colorDependence(true)
                .min(10.0f)
                .max(800.0f)
                .defaultValue(origParameters.sensorRadius)
                .tooltip("The maximum radius in which a sensor cell can detect mass concentrations."),
            parameters.sensorRadius);
    }
    AlienImGui::EndTreeNode();

    /**
     * Transmitter
     */
    if (AlienImGui::BeginTreeNode(AlienImGui::TreeNodeParameters().name("Cell type: Transmitter"))) {
        AlienImGui::SliderFloat(
            AlienImGui::SliderFloatParameters()
                .name("Energy distribution radius")
                .textWidth(RightColumnWidth)
                .colorDependence(true)
                .min(0)
                .max(5.0f)
                .defaultValue(origParameters.transmitterEnergyDistributionRadius)
                .tooltip("The maximum distance over which a transmitter cell transfers its additional energy to nearby transmitter or constructor cells."),
            parameters.transmitterEnergyDistributionRadius);
        AlienImGui::SliderFloat(
            AlienImGui::SliderFloatParameters()
                .name("Energy distribution Value")
                .textWidth(RightColumnWidth)
                .colorDependence(true)
                .min(0)
                .max(20.0f)
                .defaultValue(origParameters.transmitterEnergyDistributionValue)
                .tooltip("The amount of energy which a transmitter cell can transfer to nearby transmitter or constructor cells or to connected cells."),
            parameters.transmitterEnergyDistributionValue);
        AlienImGui::Checkbox(
            AlienImGui::CheckboxParameters()
                .name("Same creature energy distribution")
                .textWidth(RightColumnWidth)
                .defaultValue(origParameters.transmitterEnergyDistributionSameCreature)
                .tooltip("If activated, the transmitter cells can only transfer energy to nearby cells belonging to the same creature."),
            parameters.transmitterEnergyDistributionSameCreature);
    }
    AlienImGui::EndTreeNode();

    /**
     * Reconnector
     */
    if (AlienImGui::BeginTreeNode(AlienImGui::TreeNodeParameters().name("Cell type: Reconnector"))) {
        AlienImGui::SliderFloat(
            AlienImGui::SliderFloatParameters()
                .name("Radius")
                .textWidth(RightColumnWidth)
                .colorDependence(true)
                .min(0.0f)
                .max(3.0f)
                .defaultValue(origParameters.reconnectorRadius)
                .tooltip("The maximum radius in which a reconnector cell can establish or destroy connections to other cells."),
            parameters.reconnectorRadius);
    }
    AlienImGui::EndTreeNode();

    /**
     * Detonator
     */
    if (AlienImGui::BeginTreeNode(AlienImGui::TreeNodeParameters().name("Cell type: Detonator"))) {
        AlienImGui::SliderFloat(
            AlienImGui::SliderFloatParameters()
                .name("Blast radius")
                .textWidth(RightColumnWidth)
                .colorDependence(true)
                .min(0.0f)
                .max(10.0f)
                .defaultValue(origParameters.detonatorRadius)
                .tooltip("The radius of the detonation."),
            parameters.detonatorRadius);
        AlienImGui::SliderFloat(
            AlienImGui::SliderFloatParameters()
                .name("Chain explosion probability")
                .textWidth(RightColumnWidth)
                .colorDependence(true)
                .min(0.0f)
                .max(1.0f)
                .defaultValue(origParameters.detonatorChainExplosionProbability)
                .tooltip("The probability that the explosion of one detonator will trigger the explosion of other detonators within the blast radius."),
            parameters.detonatorChainExplosionProbability);
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
                .name("Low genome complexity penalty")
                .textWidth(RightColumnWidth)
                .colorDependence(true)
                .min(0)
                .max(1.0f)
                .format("%.2f")
                .defaultValue(origParameters.baseValues.radiationAbsorptionLowGenomeComplexityPenalty)
                .tooltip(Const::ParameterRadiationAbsorptionLowGenomeComplexityPenaltyTooltip),
            parameters.baseValues.radiationAbsorptionLowGenomeComplexityPenalty);
        AlienImGui::SliderFloat(
            AlienImGui::SliderFloatParameters()
                .name("Low connection penalty")
                .textWidth(RightColumnWidth)
                .colorDependence(true)
                .min(0)
                .max(5.0f)
                .format("%.1f")
                .defaultValue(origParameters.radiationAbsorptionLowConnectionPenalty)
                .tooltip("When this parameter is increased, cells with fewer cell connections will absorb less energy from an incoming energy "
                         "particle."),
            parameters.radiationAbsorptionLowConnectionPenalty);
        AlienImGui::SliderFloat(
            AlienImGui::SliderFloatParameters()
                .name("High velocity penalty")
                .textWidth(RightColumnWidth)
                .colorDependence(true)
                .min(0)
                .max(30.0f)
                .logarithmic(true)
                .format("%.2f")
                .defaultValue(origParameters.radiationAbsorptionHighVelocityPenalty)
                .tooltip("When this parameter is increased, fast moving cells will absorb less energy from an incoming energy particle."),
            parameters.radiationAbsorptionHighVelocityPenalty);
        AlienImGui::SliderFloat(
            AlienImGui::SliderFloatParameters()
                .name("Low velocity penalty")
                .textWidth(RightColumnWidth)
                .colorDependence(true)
                .min(0)
                .max(1.0f)
                .format("%.2f")
                .defaultValue(origParameters.baseValues.radiationAbsorptionLowVelocityPenalty)
                .tooltip("When this parameter is increased, slowly moving cells will absorb less energy from an incoming energy particle."),
            parameters.baseValues.radiationAbsorptionLowVelocityPenalty);
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
                .name("Same mutant protection")
                .textWidth(RightColumnWidth)
                .min(0)
                .max(1.0f)
                .defaultValue(toVector<MAX_COLORS, MAX_COLORS>(origParameters.attackerSameMutantPenalty))
                .tooltip("The larger this parameter is, the less energy can be gained by attacking creatures with the same mutation id."),
            parameters.attackerSameMutantPenalty);
        AlienImGui::InputFloatColorMatrix(
            AlienImGui::InputFloatColorMatrixParameters()
                .name("New complex mutant protection")
                .textWidth(RightColumnWidth)
                .min(0)
                .max(1.0f)
                .defaultValue(toVector<MAX_COLORS, MAX_COLORS>(origParameters.baseValues.cellTypeAttackerNewComplexMutantPenalty))
                .tooltip("A high value protects new mutants with equal or greater genome complexity from being attacked."),
            parameters.baseValues.cellTypeAttackerNewComplexMutantPenalty);
        AlienImGui::SliderFloat(
            AlienImGui::SliderFloatParameters()
                .name("Sensor detection factor")
                .textWidth(RightColumnWidth)
                .colorDependence(true)
                .min(0)
                .max(1.0f)
                .defaultValue(origParameters.attackerSensorDetectionFactor)
                .tooltip("This parameter controls whether the target must be previously detected with sensors in order to be attacked. The larger this "
                         "value is, the less energy can be gained during the attack if the target has not already been detected. For this purpose, the "
                         "attacker "
                         "cell searches for connected (or connected-connected) sensor cells to see which cell networks they have detected last time and "
                         "compares them with the attacked target."),
            parameters.attackerSensorDetectionFactor);
        AlienImGui::SliderFloat(
            AlienImGui::SliderFloatParameters()
                .name("Geometry penalty")
                .textWidth(RightColumnWidth)
                .colorDependence(true)
                .min(0)
                .max(5.0f)
                .defaultValue(origParameters.baseValues.cellTypeAttackerGeometryDeviationExponent)
                .tooltip("The larger this value is, the less energy a cell can gain from an attack if the local "
                         "geometry of the attacked cell does not match the attacking cell."),
            parameters.baseValues.cellTypeAttackerGeometryDeviationExponent);
        AlienImGui::SliderFloat(
            AlienImGui::SliderFloatParameters()
                .name("Connections mismatch penalty")
                .textWidth(RightColumnWidth)
                .colorDependence(true)
                .min(0)
                .max(1.0f)
                .defaultValue(origParameters.baseValues.cellTypeAttackerConnectionsMismatchPenalty)
                .tooltip("The larger this parameter is, the more difficult it is to attack cells that contain more connections."),
            parameters.baseValues.cellTypeAttackerConnectionsMismatchPenalty);
    }
    AlienImGui::EndTreeNode();

    /**
     * Expert settings: Cell color transition rules
     */
    if (AlienImGui::BeginTreeNode(AlienImGui::TreeNodeParameters()
                                      .name("Expert settings: Cell color transition rules")
                                      .visible(parameters.features.cellColorTransitionRules)
                                      .blinkWhenActivated(true))) {
        for (int color = 0; color < MAX_COLORS; ++color) {
            ImGui::PushID(color);
            auto widgetParameters = AlienImGui::InputColorTransitionParameters()
                                        .textWidth(RightColumnWidth)
                                        .color(color)
                                        .defaultTargetColor(origParameters.baseValues.cellColorTransitionTargetColor[color])
                                        .defaultTransitionAge(origParameters.baseValues.cellColorTransitionDuration[color])
                                        .logarithmic(true)
                                        .infinity(true);
            if (0 == color) {
                widgetParameters.name("Target color and duration")
                    .tooltip("Rules can be defined that describe how the colors of cells will change over time. For this purpose, a subsequent "
                             "color can "
                             "be defined for each cell color. In addition, durations must be specified that define how many time steps the "
                             "corresponding "
                             "color are kept.");
            }
            AlienImGui::InputColorTransition(
                widgetParameters, color, parameters.baseValues.cellColorTransitionTargetColor[color], parameters.baseValues.cellColorTransitionDuration[color]);
            ImGui::PopID();
        }
    }
    AlienImGui::EndTreeNode();

    /**
     * Expert settings: Cell age limiter
     */
    if (AlienImGui::BeginTreeNode(
            AlienImGui::TreeNodeParameters().name("Expert settings: Cell age limiter").visible(parameters.features.cellAgeLimiter).blinkWhenActivated(true))) {
        AlienImGui::SliderFloat(
            AlienImGui::SliderFloatParameters()
                .name("Maximum inactive cell age")
                .textWidth(RightColumnWidth)
                .colorDependence(true)
                .min(1.0f)
                .max(10000000.0f)
                .format("%.0f")
                .logarithmic(true)
                .infinity(true)
                .disabledValue(parameters.baseValues.maxAgeForInactiveCells)
                .defaultEnabledValue(&origParameters.maxAgeForInactiveCellsActivated)
                .defaultValue(origParameters.baseValues.maxAgeForInactiveCells)
                .tooltip("Here, you can set the maximum age for a cell whose function or those of its neighbors have not been triggered. Cells which "
                         "are in state 'Under construction' are not affected by this option."),
            parameters.baseValues.maxAgeForInactiveCells,
            &parameters.maxAgeForInactiveCellsActivated);
        AlienImGui::SliderInt(
            AlienImGui::SliderIntParameters()
                .name("Maximum free cell age")
                .textWidth(RightColumnWidth)
                .colorDependence(true)
                .min(1)
                .max(10000000)
                .logarithmic(true)
                .infinity(true)
                .disabledValue(parameters.freeCellMaxAge)
                .defaultEnabledValue(&origParameters.freeCellMaxAgeActivated)
                .defaultValue(origParameters.freeCellMaxAge)
                .tooltip("The maximal age of cells that arise from energy particles can be set here."),
            parameters.freeCellMaxAge,
            &parameters.freeCellMaxAgeActivated);
        AlienImGui::Checkbox(
            AlienImGui::CheckboxParameters()
                .name("Reset age after construction")
                .textWidth(RightColumnWidth)
                .defaultValue(origParameters.resetCellAgeAfterActivation)
                .tooltip("If this option is activated, the age of the cells is reset to 0 after the construction of their cell network is completed, "
                         "i.e. when the state of the cells changes from 'Under construction' to 'Ready'. This option is particularly useful if a low "
                         "'Maximum "
                         "inactive cell age' is set, as cell networks that are under construction are inactive and could die immediately after "
                         "completion if their construction takes a long time."),
            parameters.resetCellAgeAfterActivation);
        AlienImGui::SliderInt(
            AlienImGui::SliderIntParameters()
                .name("Maximum age balancing")
                .textWidth(RightColumnWidth)
                .logarithmic(true)
                .min(1000)
                .max(1000000)
                .disabledValue(&parameters.maxCellAgeBalancerInterval)
                .defaultEnabledValue(&origParameters.maxCellAgeBalancerActivated)
                .defaultValue(&origParameters.maxCellAgeBalancerInterval)
                .tooltip("Adjusts the maximum age at regular intervals. It increases the maximum age for the cell color where the fewest "
                         "replicators exist. "
                         "Conversely, the maximum age is decreased for the cell color with the most replicators."),
            &parameters.maxCellAgeBalancerInterval,
            &parameters.maxCellAgeBalancerActivated);
    }
    AlienImGui::EndTreeNode();

    /**
     * Expert settings: Cell glow
     */
    if (AlienImGui::BeginTreeNode(
            AlienImGui::TreeNodeParameters().name("Expert settings: Cell glow").visible(parameters.features.cellGlow).blinkWhenActivated(true))) {
        AlienImGui::Switcher(
            AlienImGui::SwitcherParameters()
                .name("Coloring")
                .textWidth(RightColumnWidth)
                .defaultValue(origParameters.cellGlowColoring)
                .values(
                    {"Energy",
                     "Standard cell colors",
                     "Mutants",
                     "Mutants and cell functions",
                     "Cell states",
                     "Genome complexities",
                     "Single cell function",
                     "All cell functions"})
                .tooltip(Const::ColoringParameterTooltip),
            parameters.cellGlowColoring);
        AlienImGui::SliderFloat(
            AlienImGui::SliderFloatParameters()
                .name("Radius")
                .textWidth(RightColumnWidth)
                .min(1.0f)
                .max(8.0f)
                .defaultValue(&origParameters.cellGlowRadius)
                .tooltip("The radius of the glow. Please note that a large radius affects the performance."),
            &parameters.cellGlowRadius);
        AlienImGui::SliderFloat(
            AlienImGui::SliderFloatParameters()
                .name("Strength")
                .textWidth(RightColumnWidth)
                .min(0)
                .max(1.0f)
                .defaultValue(&origParameters.cellGlowStrength)
                .tooltip("The strength of the glow."),
            &parameters.cellGlowStrength);
    }
    AlienImGui::EndTreeNode();

    /**
     * Expert settings: Customize deletion mutations
     */
    if (AlienImGui::BeginTreeNode(AlienImGui::TreeNodeParameters()
                                      .name("Expert settings: Customize deletion mutations")
                                      .visible(parameters.features.customizeDeletionMutations)
                                      .blinkWhenActivated(true))) {
        AlienImGui::SliderInt(
            AlienImGui::SliderIntParameters()
                .name("Minimum size")
                .textWidth(RightColumnWidth)
                .min(0)
                .max(1000)
                .logarithmic(true)
                .defaultValue(&origParameters.cellCopyMutationDeletionMinSize)
                .tooltip("The minimum size of genomes (on the basis of the coded cells) is determined here that can result from delete mutations. The default "
                         "is 0."),
            &parameters.cellCopyMutationDeletionMinSize);
    }
    AlienImGui::EndTreeNode();

    /**
     * Expert settings: Customize neuron mutations
     */
    if (AlienImGui::BeginTreeNode(AlienImGui::TreeNodeParameters()
                                      .name("Expert settings: Customize neuron mutations")
                                      .visible(parameters.features.customizeNeuronMutations)
                                      .blinkWhenActivated(true))) {
        AlienImGui::SliderFloat(
            AlienImGui::SliderFloatParameters()
                .name("Affected weights")
                .textWidth(RightColumnWidth)
                .min(0.0f)
                .max(1.0f)
                .format("%.3f")
                .defaultValue(&origParameters.cellCopyMutationNeuronDataWeight)
                .tooltip("The proportion of weights in the neuronal network of a cell that are changed within a neuron mutation. The default is 0.2."),
            &parameters.cellCopyMutationNeuronDataWeight);
        AlienImGui::SliderFloat(
            AlienImGui::SliderFloatParameters()
                .name("Affected biases")
                .textWidth(RightColumnWidth)
                .min(0.0f)
                .max(1.0f)
                .format("%.3f")
                .defaultValue(&origParameters.cellCopyMutationNeuronDataBias)
                .tooltip("The proportion of biases in the neuronal network of a cell that are changed within a neuron mutation. The default is 0.2."),
            &parameters.cellCopyMutationNeuronDataBias);
        AlienImGui::SliderFloat(
            AlienImGui::SliderFloatParameters()
                .name("Affected activation functions")
                .textWidth(RightColumnWidth)
                .min(0.0f)
                .max(1.0f)
                .format("%.3f")
                .defaultValue(&origParameters.cellCopyMutationNeuronDataActivationFunction)
                .tooltip("The proportion of activation functions in the neuronal network of a cell that are changed within a neuron mutation. The default is 0.05."),
            &parameters.cellCopyMutationNeuronDataActivationFunction);
        AlienImGui::SliderFloat(
            AlienImGui::SliderFloatParameters()
                .name("Reinforcement factor")
                .textWidth(RightColumnWidth)
                .min(1.0f)
                .max(1.2f)
                .format("%.3f")
                .defaultValue(&origParameters.cellCopyMutationNeuronDataReinforcement)
                .tooltip("If a weight or bias of the neural network is adjusted by a mutation, it can either be reinforced, weakened or shifted by an offset. "
                         "The factor that is used for reinforcement is defined here. The default is 1.05."),
            &parameters.cellCopyMutationNeuronDataReinforcement);
        AlienImGui::SliderFloat(
            AlienImGui::SliderFloatParameters()
                .name("Damping factor")
                .textWidth(RightColumnWidth)
                .min(1.0f)
                .max(1.2f)
                .format("%.3f")
                .defaultValue(&origParameters.cellCopyMutationNeuronDataDamping)
                .tooltip("If a weight or bias of the neural network is adjusted by a mutation, it can either be reinforced, weakened or shifted by an offset. "
                         "The factor that is used for weakening is defined here. The default is 1.05."),
            &parameters.cellCopyMutationNeuronDataDamping);
        AlienImGui::SliderFloat(
            AlienImGui::SliderFloatParameters()
                .name("Offset")
                .textWidth(RightColumnWidth)
                .min(0.0f)
                .max(0.2f)
                .format("%.3f")
                .defaultValue(&origParameters.cellCopyMutationNeuronDataOffset)
                .tooltip("If a weight or bias of the neural network is adjusted by a mutation, it can either be reinforced, weakened or shifted by an offset. "
                         "The value that is used for the offset is defined here. The default is 0.05."),
            &parameters.cellCopyMutationNeuronDataOffset);
    }
    AlienImGui::EndTreeNode();

    /**
     * Expert settings: External energy control
     */
    if (AlienImGui::BeginTreeNode(AlienImGui::TreeNodeParameters()
                                      .name("Expert settings: External energy control")
                                      .visible(parameters.features.externalEnergyControl)
                                      .blinkWhenActivated(true))) {
        AlienImGui::SliderFloat(
            AlienImGui::SliderFloatParameters()
                .name("External energy amount")
                .textWidth(RightColumnWidth)
                .min(0.0f)
                .max(100000000.0f)
                .format("%.0f")
                .logarithmic(true)
                .infinity(true)
                .defaultValue(&origParameters.externalEnergy)
                .tooltip("This parameter can be used to set the amount of energy of an external energy pool. This type of energy can then be "
                         "transferred to all constructor cells at a certain rate (see inflow settings).\n\nTip: You can explicitly enter a "
                         "numerical value by clicking on the slider while holding CTRL.\n\nWarning: Too much external energy can result in a "
                         "massive production of cells and slow down or even crash the simulation."),
            &parameters.externalEnergy);
        AlienImGui::SliderFloat(
            AlienImGui::SliderFloatParameters()
                .name("Inflow")
                .textWidth(RightColumnWidth)
                .colorDependence(true)
                .min(0.0f)
                .max(1.0f)
                .format("%.5f")
                .logarithmic(true)
                .defaultValue(origParameters.externalEnergyInflowFactor)
                .tooltip("Here one can specify the fraction of energy transferred to constructor cells.\n\nFor example, a value of 0.05 means that "
                         "each time "
                         "a constructor cell tries to build a new cell, 5% of the required energy is transferred for free from the external energy "
                         "source."),
            parameters.externalEnergyInflowFactor);
        AlienImGui::SliderFloat(
            AlienImGui::SliderFloatParameters()
                .name("Conditional inflow")
                .textWidth(RightColumnWidth)
                .colorDependence(true)
                .min(0.00f)
                .max(1.0f)
                .format("%.5f")
                .defaultValue(origParameters.externalEnergyConditionalInflowFactor)
                .tooltip("Here one can specify the fraction of energy transferred to constructor cells if they can provide the remaining energy for the "
                         "construction process.\n\nFor example, a value of 0.6 means that a constructor cell receives 60% of the energy required to "
                         "build the new cell for free from the external energy source. However, it must provide 40% of the energy required by itself. "
                         "Otherwise, no energy will be transferred."),
            parameters.externalEnergyConditionalInflowFactor);
        AlienImGui::Checkbox(
            AlienImGui::CheckboxParameters()
                .name("Inflow only for non-replicators")
                .textWidth(RightColumnWidth)
                .defaultValue(origParameters.externalEnergyInflowOnlyForNonSelfReplicators)
                .tooltip("If activated, external energy can only be transferred to constructor cells that are not self-replicators. "
                         "This option can be used to foster the evolution of additional body parts."),
            parameters.externalEnergyInflowOnlyForNonSelfReplicators);
        AlienImGui::SliderFloat(
            AlienImGui::SliderFloatParameters()
                .name("Backflow")
                .textWidth(RightColumnWidth)
                .colorDependence(true)
                .min(0.0f)
                .max(1.0f)
                .defaultValue(origParameters.externalEnergyBackflowFactor)
                .tooltip("The proportion of energy that flows back from the simulation to the external energy pool. Each time a cell loses energy "
                         "or dies a fraction of its energy will be taken. The remaining "
                         "fraction of the energy stays in the simulation and will be used to create a new energy particle."),
            parameters.externalEnergyBackflowFactor);
        AlienImGui::SliderFloat(
            AlienImGui::SliderFloatParameters()
                .name("Backflow limit")
                .textWidth(RightColumnWidth)
                .min(0.0f)
                .max(100000000.0f)
                .format("%.0f")
                .logarithmic(true)
                .infinity(true)
                .defaultValue(&origParameters.externalEnergyBackflowLimit)
                .tooltip("Energy from the simulation can only flow back into the external energy pool as long as the amount of external energy is "
                         "below this value."),
            &parameters.externalEnergyBackflowLimit);
    }
    AlienImGui::EndTreeNode();

    /**
     * Expert settings: Genome complexity measurement
     */
    if (AlienImGui::BeginTreeNode(AlienImGui::TreeNodeParameters()
                                      .name("Expert settings: Genome complexity measurement")
                                      .visible(parameters.features.genomeComplexityMeasurement)
                                      .blinkWhenActivated(true))) {
        AlienImGui::SliderFloat(
            AlienImGui::SliderFloatParameters()
                .name("Size factor")
                .textWidth(RightColumnWidth)
                .colorDependence(true)
                .min(0.0f)
                .max(1.0f)
                .format("%.2f")
                .defaultValue(origParameters.genomeComplexitySizeFactor)
                .tooltip("This parameter controls how the number of encoded cells in the genome influences the calculation of its complexity."),
            parameters.genomeComplexitySizeFactor);
        AlienImGui::SliderFloat(
            AlienImGui::SliderFloatParameters()
                .name("Ramification factor")
                .textWidth(RightColumnWidth)
                .colorDependence(true)
                .min(0.0f)
                .max(20.0f)
                .format("%.2f")
                .defaultValue(origParameters.genomeComplexityRamificationFactor)
                .tooltip("With this parameter, the number of ramifications of the cell structure to the genome is taken into account for the "
                         "calculation of the genome complexity. For instance, genomes that contain many sub-genomes or many construction branches will "
                         "then have a high complexity value."),
            parameters.genomeComplexityRamificationFactor);
        AlienImGui::SliderInt(
            AlienImGui::SliderIntParameters()
                .name("Depth level")
                .textWidth(RightColumnWidth)
                .colorDependence(true)
                .min(1)
                .max(20)
                .infinity(true)
                .defaultValue(origParameters.genomeComplexityDepthLevel)
                .tooltip("This allows to specify up to which level of the sub-genomes the complexity calculation should be carried out. For example, a "
                         "value of 2 means that the sub- and sub-sub-genomes are taken into account in addition to the main genome."),
            parameters.genomeComplexityDepthLevel);
    }
    AlienImGui::EndTreeNode();

    SimulationParametersValidationService::get().validateAndCorrect(parameters);

    if (parameters != lastParameters) {
        _simulationFacade->setSimulationParameters(parameters, SimulationParametersUpdateConfig::AllExceptChangingPositions);
    }
}

std::string _SimulationParametersBaseWidgets::getLocationName()
{
    return "Simulation parameters for 'Base'";
}

int _SimulationParametersBaseWidgets::getLocationIndex() const
{
    return 0;
}

void _SimulationParametersBaseWidgets::setLocationIndex(int locationIndex)
{
    // do nothing
}

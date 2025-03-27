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
     * Expert settings: Cell age limiter
     */
    if (AlienImGui::BeginTreeNode(
            AlienImGui::TreeNodeParameters().name("Expert settings: Cell age limiter").visible(parameters.expertSettingsToggles.cellAgeLimiter).blinkWhenActivated(true))) {
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
            AlienImGui::TreeNodeParameters().name("Expert settings: Cell glow").visible(parameters.expertSettingsToggles.cellGlow).blinkWhenActivated(true))) {
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
                                      .visible(parameters.expertSettingsToggles.customizeDeletionMutations)
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
                                      .visible(parameters.expertSettingsToggles.customizeNeuronMutations)
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
                                      .visible(parameters.expertSettingsToggles.externalEnergyControl)
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
                                      .visible(parameters.expertSettingsToggles.genomeComplexityMeasurement)
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

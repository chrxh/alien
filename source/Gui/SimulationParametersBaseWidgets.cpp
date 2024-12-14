#include "SimulationParametersBaseWidgets.h"

#include <imgui.h>

#include "EngineInterface/SimulationFacade.h"
#include "EngineInterface/SimulationParametersTypes.h"
#include "EngineInterface/SimulationParametersUpdateConfig.h"

#include "AlienImGui.h"
#include "CellFunctionStrings.h"
#include "HelpStrings.h"
#include "EngineInterface/SimulationParametersEditService.h"
#include "EngineInterface/SimulationParametersValidationService.h"

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
    for (int i = 0; i < CellFunction_Count; ++i) {
        _cellFunctionStrings.emplace_back(Const::CellFunctionToStringMap.at(i));
    }
}

void _SimulationParametersBaseWidgets::process()
{
    auto parameters = _simulationFacade->getSimulationParameters();
    auto origParameters = _simulationFacade->getOriginalSimulationParameters();
    auto lastParameters = parameters;

    /**
     * General
     */
    if (AlienImGui::BeginTreeNode(AlienImGui::TreeNodeParameters().text("General"))) {
        AlienImGui::InputText(
            AlienImGui::InputTextParameters().name("Project name").textWidth(RightColumnWidth).defaultValue(origParameters.projectName),
            parameters.projectName,
            sizeof(Char64) / sizeof(char));

        AlienImGui::EndTreeNode();
    }
    /**
     * Rendering
     */
    if (AlienImGui::BeginTreeNode(AlienImGui::TreeNodeParameters().text("Visualization"))) {
        AlienImGui::ColorButtonWithPicker(
            AlienImGui::ColorButtonWithPickerParameters().name("Background color").textWidth(RightColumnWidth).defaultValue(origParameters.backgroundColor),
            parameters.backgroundColor,
            _backupColor,
            _zoneColorPalette.getReference());
        AlienImGui::Switcher(
            AlienImGui::SwitcherParameters()
                .name("Primary cell coloring")
                .textWidth(RightColumnWidth)
                .defaultValue(origParameters.cellColoring)
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
            parameters.cellColoring);
        if (parameters.cellColoring == CellColoring_CellFunction) {
            AlienImGui::Switcher(
                AlienImGui::SwitcherParameters()
                    .name("Highlighted cell function")
                    .textWidth(RightColumnWidth)
                    .defaultValue(origParameters.highlightedCellFunction)
                    .values(_cellFunctionStrings)
                    .tooltip("The specific cell function type to be highlighted can be selected here."),
                parameters.highlightedCellFunction);
        }
        AlienImGui::SliderFloat(
            AlienImGui::SliderFloatParameters()
                .name("Cell radius")
                .textWidth(RightColumnWidth)
                .min(0)
                .max(0.5f)
                .defaultValue(&origParameters.cellRadius)
                .tooltip("Specifies the radius of the drawn cells in unit length."),
            &parameters.cellRadius);
        AlienImGui::SliderFloat(
            AlienImGui::SliderFloatParameters()
                .name("Zoom level for cell activity")
                .textWidth(RightColumnWidth)
                .min(0)
                .max(32.0f)
                .infinity(true)
                .defaultValue(&origParameters.zoomLevelNeuronalActivity)
                .tooltip("The zoom level from which the neuronal activities become visible."),
            &parameters.zoomLevelNeuronalActivity);
        AlienImGui::Checkbox(
            AlienImGui::CheckboxParameters()
                .name("Attack visualization")
                .textWidth(RightColumnWidth)
                .defaultValue(origParameters.attackVisualization)
                .tooltip("If activated, successful attacks of attacker cells are visualized."),
            parameters.attackVisualization);
        AlienImGui::Checkbox(
            AlienImGui::CheckboxParameters()
                .name("Muscle movement visualization")
                .textWidth(RightColumnWidth)
                .defaultValue(origParameters.muscleMovementVisualization)
                .tooltip("If activated, the direction in which muscle cells are moving are visualized."),
            parameters.muscleMovementVisualization);
        AlienImGui::Checkbox(
            AlienImGui::CheckboxParameters()
                .name("Borderless rendering")
                .textWidth(RightColumnWidth)
                .defaultValue(origParameters.borderlessRendering)
                .tooltip("If activated, the simulation is rendered periodically in the view port."),
            parameters.borderlessRendering);
        AlienImGui::Checkbox(
            AlienImGui::CheckboxParameters()
                .name("Adaptive space grid")
                .textWidth(RightColumnWidth)
                .defaultValue(origParameters.gridLines)
                .tooltip("Draws a suitable grid in the background depending on the zoom level."),
            parameters.gridLines);
        AlienImGui::Checkbox(
            AlienImGui::CheckboxParameters()
                .name("Mark reference domain")
                .textWidth(RightColumnWidth)
                .defaultValue(origParameters.markReferenceDomain)
                .tooltip("Draws borders along the world before it repeats itself."),
            parameters.markReferenceDomain);
        AlienImGui::Checkbox(
            AlienImGui::CheckboxParameters()
                .name("Show radiation sources")
                .textWidth(RightColumnWidth)
                .defaultValue(origParameters.showRadiationSources)
                .tooltip("Draws red crosses in the center of radiation sources."),
            parameters.showRadiationSources);
        AlienImGui::EndTreeNode();
    }

    /**
     * Numerics
     */
    if (AlienImGui::BeginTreeNode(AlienImGui::TreeNodeParameters().text("Numerics"))) {
        AlienImGui::SliderFloat(
            AlienImGui::SliderFloatParameters()
                .name("Time step size")
                .textWidth(RightColumnWidth)
                .min(0)
                .max(1.0f)
                .defaultValue(&origParameters.timestepSize)
                .tooltip(std::string("The time duration calculated in a single simulation step. Smaller values increase the accuracy of the simulation "
                                     "while larger values can lead to numerical instabilities.")),
            &parameters.timestepSize);
        AlienImGui::EndTreeNode();
    }

    /**
     * Physics: Motion
     */
    if (AlienImGui::BeginTreeNode(AlienImGui::TreeNodeParameters().text("Physics: Motion"))) {
        if (AlienImGui::Switcher(
                AlienImGui::SwitcherParameters()
                    .name("Motion type")
                    .textWidth(RightColumnWidth)
                    .defaultValue(origParameters.motionType)
                    .values({"Fluid dynamics", "Collision-based"})
                    .tooltip(std::string(
                        "The algorithm for the particle motions is defined here. If 'Fluid dynamics' is selected, an SPH fluid solver is used for the "
                        "calculation of the forces. The particles then behave like (compressible) liquids or gases. The other option 'Collision-based' "
                        "calculates the forces based on particle collisions and should be preferred for mechanical simulation with solids.")),
                parameters.motionType)) {
            if (parameters.motionType == MotionType_Fluid) {
                parameters.motionData.fluidMotion = FluidMotion();
            } else {
                parameters.motionData.collisionMotion = CollisionMotion();
            }
        }
        if (parameters.motionType == MotionType_Fluid) {
            AlienImGui::SliderFloat(
                AlienImGui::SliderFloatParameters()
                    .name("Smoothing length")
                    .textWidth(RightColumnWidth)
                    .min(0)
                    .max(3.0f)
                    .defaultValue(&origParameters.motionData.fluidMotion.smoothingLength)
                    .tooltip(std::string("The smoothing length determines the region of influence of the neighboring particles for the calculation of "
                                         "density, pressure and viscosity. Values that are too small lead to numerical instabilities, while values that "
                                         "are too large cause the particles to drift apart.")),
                &parameters.motionData.fluidMotion.smoothingLength);
            AlienImGui::SliderFloat(
                AlienImGui::SliderFloatParameters()
                    .name("Pressure")
                    .textWidth(RightColumnWidth)
                    .min(0)
                    .max(0.3f)
                    .defaultValue(&origParameters.motionData.fluidMotion.pressureStrength)
                    .tooltip(std::string("This parameter allows to control the strength of the pressure.")),
                &parameters.motionData.fluidMotion.pressureStrength);
            AlienImGui::SliderFloat(
                AlienImGui::SliderFloatParameters()
                    .name("Viscosity")
                    .textWidth(RightColumnWidth)
                    .min(0)
                    .max(0.3f)
                    .defaultValue(&origParameters.motionData.fluidMotion.viscosityStrength)
                    .tooltip(std::string("This parameter be used to control the strength of the viscosity. Larger values lead to a smoother movement.")),
                &parameters.motionData.fluidMotion.viscosityStrength);
        } else {
            AlienImGui::SliderFloat(
                AlienImGui::SliderFloatParameters()
                    .name("Repulsion strength")
                    .textWidth(RightColumnWidth)
                    .min(0)
                    .max(0.3f)
                    .defaultValue(&origParameters.motionData.collisionMotion.cellRepulsionStrength)
                    .tooltip(std::string("The strength of the repulsive forces, between two cells that are not connected.")),
                &parameters.motionData.collisionMotion.cellRepulsionStrength);
            AlienImGui::SliderFloat(
                AlienImGui::SliderFloatParameters()
                    .name("Maximum collision distance")
                    .textWidth(RightColumnWidth)
                    .min(0)
                    .max(3.0f)
                    .defaultValue(&origParameters.motionData.collisionMotion.cellMaxCollisionDistance)
                    .tooltip(std::string("Maximum distance up to which a collision of two cells is possible.")),
                &parameters.motionData.collisionMotion.cellMaxCollisionDistance);
        }
        AlienImGui::SliderFloat(
            AlienImGui::SliderFloatParameters()
                .name("Friction")
                .textWidth(RightColumnWidth)
                .min(0)
                .max(1.0f)
                .logarithmic(true)
                .format("%.4f")
                .defaultValue(&origParameters.baseValues.friction)
                .tooltip(std::string("This specifies the fraction of the velocity that is slowed down per time step.")),
            &parameters.baseValues.friction);
        AlienImGui::SliderFloat(
            AlienImGui::SliderFloatParameters()
                .name("Rigidity")
                .textWidth(RightColumnWidth)
                .min(0)
                .max(1.0f)
                .format("%.2f")
                .defaultValue(&origParameters.baseValues.rigidity)
                .tooltip(std::string(
                    "Controls the rigidity of connected cells.\nA higher value will cause connected cells to move more uniformly as a rigid body.")),
            &parameters.baseValues.rigidity);
        AlienImGui::EndTreeNode();
    }

    /**
     * Physics: Thresholds
     */
    if (AlienImGui::BeginTreeNode(AlienImGui::TreeNodeParameters().text("Physics: Thresholds"))) {
        AlienImGui::SliderFloat(
            AlienImGui::SliderFloatParameters()
                .name("Maximum velocity")
                .textWidth(RightColumnWidth)
                .min(0)
                .max(6.0f)
                .defaultValue(&origParameters.cellMaxVelocity)
                .tooltip(std::string("Maximum velocity that a cell can reach.")),
            &parameters.cellMaxVelocity);
        AlienImGui::SliderFloat(
            AlienImGui::SliderFloatParameters()
                .name("Maximum force")
                .textWidth(RightColumnWidth)
                .min(0)
                .max(3.0f)
                .colorDependence(true)
                .defaultValue(origParameters.baseValues.cellMaxForce)
                .tooltip(std::string("Maximum force that can be applied to a cell without causing it to disintegrate.")),
            parameters.baseValues.cellMaxForce);
        AlienImGui::SliderFloat(
            AlienImGui::SliderFloatParameters()
                .name("Minimum distance")
                .textWidth(RightColumnWidth)
                .min(0)
                .max(1.0f)
                .defaultValue(&origParameters.cellMinDistance)
                .tooltip(std::string("Minimum distance between two cells.")),
            &parameters.cellMinDistance);
        AlienImGui::EndTreeNode();
    }

    /**
     * Physics: Binding
     */
    if (AlienImGui::BeginTreeNode(AlienImGui::TreeNodeParameters().text("Physics: Binding"))) {
        AlienImGui::SliderFloat(
            AlienImGui::SliderFloatParameters()
                .name("Maximum distance")
                .textWidth(RightColumnWidth)
                .min(0)
                .max(5.0f)
                .colorDependence(true)
                .defaultValue(origParameters.cellMaxBindingDistance)
                .tooltip(std::string("Maximum distance up to which a connection of two cells is possible.")),
            parameters.cellMaxBindingDistance);
        AlienImGui::SliderFloat(
            AlienImGui::SliderFloatParameters()
                .name("Fusion velocity")
                .textWidth(RightColumnWidth)
                .min(0)
                .max(2.0f)
                .defaultValue(&origParameters.baseValues.cellFusionVelocity)
                .tooltip(std::string("Minimum relative velocity of two colliding cells so that a connection can be established.")),
            &parameters.baseValues.cellFusionVelocity);
        AlienImGui::SliderFloat(
            AlienImGui::SliderFloatParameters()
                .name("Maximum energy")
                .textWidth(RightColumnWidth)
                .min(50.0f)
                .max(10000000.0f)
                .logarithmic(true)
                .infinity(true)
                .format("%.0f")
                .defaultValue(&origParameters.baseValues.cellMaxBindingEnergy)
                .tooltip(std::string("Maximum energy of a cell at which it can contain bonds to adjacent cells. If the energy of a cell exceeds this "
                                     "value, all bonds will be destroyed.")),
            &parameters.baseValues.cellMaxBindingEnergy);
        AlienImGui::EndTreeNode();
    }

    /**
     * Physics: Radiation
     */
    if (AlienImGui::BeginTreeNode(AlienImGui::TreeNodeParameters().text("Physics: Radiation"))) {
        auto& editService = SimulationParametersEditService::get();
        auto strength = editService.getRadiationStrengths(parameters);
        auto origStrengths = editService.getRadiationStrengths(origParameters);
        auto editedStrength = strength;
        if (AlienImGui::SliderFloat(
                AlienImGui::SliderFloatParameters()
                    .name("Relative strength")
                    .textWidth(RightColumnWidth)
                    .min(0.0f)
                    .max(1.0f)
                    .format("%.3f")
                    .defaultValue(&origStrengths.values.front())
                    .tooltip("Cells can emit energy particles over time. A portion of this energy can be released directly near the cell, while the rest is "
                             "utilized by one of the available radiation sources. This parameter determines the fraction of energy assigned to the emitted "
                             "energy particle in the vicinity of the cell. Values between 0 and 1 are permitted."),
                &editedStrength.values.front(),
                nullptr,
                &parameters.baseStrengthRatioPinned)) {
            editService.adaptRadiationStrengths(editedStrength, strength, 0);
            editService.applyRadiationStrengths(parameters, editedStrength);
        }

        AlienImGui::SliderFloat(
            AlienImGui::SliderFloatParameters()
                .name("Absorption factor")
                .textWidth(RightColumnWidth)
                .logarithmic(true)
                .colorDependence(true)
                .min(0)
                .max(1.0)
                .format("%.4f")
                .defaultValue(origParameters.baseValues.radiationAbsorption)
                .tooltip("The fraction of energy that a cell can absorb from an incoming energy particle can be specified here."),
            parameters.baseValues.radiationAbsorption);
        AlienImGui::SliderFloat(
            AlienImGui::SliderFloatParameters()
                .name("Radiation type I: Strength")
                .textWidth(RightColumnWidth)
                .colorDependence(true)
                .min(0)
                .max(0.01f)
                .logarithmic(true)
                .format("%.6f")
                .defaultValue(origParameters.baseValues.radiationCellAgeStrength)
                .tooltip("Indicates how energetic the emitted particles of aged cells are."),
            parameters.baseValues.radiationCellAgeStrength);
        AlienImGui::SliderInt(
            AlienImGui::SliderIntParameters()
                .name("Radiation type I: Minimum age")
                .textWidth(RightColumnWidth)
                .colorDependence(true)
                .infinity(true)
                .min(0)
                .max(10000000)
                .logarithmic(true)
                .defaultValue(origParameters.radiationMinCellAge)
                .tooltip("The minimum age of a cell can be defined here, from which it emits energy particles."),
            parameters.radiationMinCellAge);
        AlienImGui::SliderFloat(
            AlienImGui::SliderFloatParameters()
                .name("Radiation type II: Strength")
                .textWidth(RightColumnWidth)
                .colorDependence(true)
                .min(0)
                .max(0.01f)
                .logarithmic(true)
                .format("%.6f")
                .defaultValue(origParameters.highRadiationFactor)
                .tooltip("Indicates how energetic the emitted particles of high energy cells are."),
            parameters.highRadiationFactor);
        AlienImGui::SliderFloat(
            AlienImGui::SliderFloatParameters()
                .name("Radiation type II: Energy threshold")
                .textWidth(RightColumnWidth)
                .colorDependence(true)
                .infinity(true)
                .min(0)
                .max(100000.0f)
                .logarithmic(true)
                .format("%.1f")
                .defaultValue(origParameters.highRadiationMinCellEnergy)
                .tooltip("The minimum energy of a cell can be defined here, from which it emits energy particles."),
            parameters.highRadiationMinCellEnergy);
        AlienImGui::SliderFloat(
            AlienImGui::SliderFloatParameters()
                .name("Minimum split energy")
                .textWidth(RightColumnWidth)
                .colorDependence(true)
                .infinity(true)
                .min(1.0f)
                .max(10000.0f)
                .logarithmic(true)
                .format("%.0f")
                .defaultValue(origParameters.particleSplitEnergy)
                .tooltip("The minimum energy of an energy particle after which it can split into two particles, whereby it receives a small momentum. The "
                         "splitting does not occur immediately, but only after a certain time."),
            parameters.particleSplitEnergy);
        AlienImGui::Checkbox(
            AlienImGui::CheckboxParameters()
                .name("Energy to cell transformation")
                .textWidth(RightColumnWidth)
                .defaultValue(origParameters.particleTransformationAllowed)
                .tooltip("If activated, an energy particle will transform into a cell if the energy of the particle exceeds the normal energy value."),
            parameters.particleTransformationAllowed);

        AlienImGui::EndTreeNode();
    }

    /**
     * Cell life cycle
     */
    ImGui::PushID("Transformation");
    if (AlienImGui::BeginTreeNode(AlienImGui::TreeNodeParameters().text("Cell life cycle"))) {
        AlienImGui::SliderInt(
            AlienImGui::SliderIntParameters()
                .name("Maximum age")
                .textWidth(RightColumnWidth)
                .colorDependence(true)
                .logarithmic(true)
                .infinity(true)
                .min(1)
                .max(10000000)
                .defaultValue(origParameters.cellMaxAge)
                .tooltip("Defines the maximum age of a cell. If a cell exceeds this age it will be transformed to an energy particle."),
            parameters.cellMaxAge);
        AlienImGui::SliderFloat(
            AlienImGui::SliderFloatParameters()
                .name("Minimum energy")
                .textWidth(RightColumnWidth)
                .colorDependence(true)
                .min(10.0f)
                .max(200.0f)
                .defaultValue(origParameters.baseValues.cellMinEnergy)
                .tooltip("Minimum energy a cell needs to exist."),
            parameters.baseValues.cellMinEnergy);
        AlienImGui::SliderFloat(
            AlienImGui::SliderFloatParameters()
                .name("Normal energy")
                .textWidth(RightColumnWidth)
                .colorDependence(true)
                .min(10.0f)
                .max(200.0f)
                .defaultValue(origParameters.cellNormalEnergy)
                .tooltip(
                    "The normal energy value of a cell is defined here. This is used as a reference value in various contexts: "
                    "\n\n" ICON_FA_CHEVRON_RIGHT
                    " Attacker and Transmitter cells: When the energy of these cells is above the normal value, some of their energy is distributed to "
                    "surrounding cells.\n\n" ICON_FA_CHEVRON_RIGHT
                    " Constructor cells: Creating new cells costs energy. The creation of new cells is executed only when the "
                    "residual energy of the constructor cell does not fall below the normal value.\n\n" ICON_FA_CHEVRON_RIGHT
                    " If the transformation of energy particles to "
                    "cells is activated, an energy particle will transform into a cell if the energy of the particle exceeds the normal value."),
            parameters.cellNormalEnergy);
        AlienImGui::SliderFloat(
            AlienImGui::SliderFloatParameters()
                .name("Decay rate of dying cells")
                .colorDependence(true)
                .textWidth(RightColumnWidth)
                .min(1e-6f)
                .max(0.1f)
                .format("%.6f")
                .logarithmic(true)
                .defaultValue(origParameters.baseValues.cellDeathProbability)
                .tooltip("The probability per time step with which a cell will disintegrate (i.e. transform into an energy particle) when it is in the "
                         "state 'Dying'. This can occur when one of the following conditions is satisfied:\n\n" ICON_FA_CHEVRON_RIGHT
                         " The cell has too low energy.\n\n" ICON_FA_CHEVRON_RIGHT " The cell has exceeded its maximum age."),
            parameters.baseValues.cellDeathProbability);
        AlienImGui::Switcher(
            AlienImGui::SwitcherParameters()
                .name("Cell death consequences")
                .textWidth(RightColumnWidth)
                .defaultValue(origParameters.cellDeathConsequences)
                .values({"None", "Entire creature dies", "Detached creature parts die"})
                .tooltip("Here one can define what happens to the organism when one of its cells is in the 'Dying' state.\n\n" ICON_FA_CHEVRON_RIGHT
                         " None: Only the cell dies.\n\n" ICON_FA_CHEVRON_RIGHT
                         " Entire creature dies: All the cells of the organism will also die.\n\n" ICON_FA_CHEVRON_RIGHT
                         " Detached creature parts die: Only the parts of the organism that are no longer connected to a "
                         "constructor cell for self-replication die."),
            parameters.cellDeathConsequences);
        AlienImGui::EndTreeNode();
    }
    ImGui::PopID();

    /**
     * Mutations
     */
    if (AlienImGui::BeginTreeNode(AlienImGui::TreeNodeParameters().text("Genome copy mutations"))) {
        AlienImGui::SliderFloat(
            AlienImGui::SliderFloatParameters()
                .name("Neural net")
                .textWidth(RightColumnWidth)
                .min(0.0f)
                .max(1.0f)
                .format("%.7f")
                .logarithmic(true)
                .colorDependence(true)
                .defaultValue(origParameters.baseValues.cellCopyMutationNeuronData)
                .tooltip("This type of mutation can change the weights, biases and activation functions of neural networks of each neuron cell encoded in the genome."),
            parameters.baseValues.cellCopyMutationNeuronData);
        AlienImGui::SliderFloat(
            AlienImGui::SliderFloatParameters()
                .name("Cell properties")
                .textWidth(RightColumnWidth)
                .min(0.0f)
                .max(1.0f)
                .format("%.7f")
                .logarithmic(true)
                .colorDependence(true)
                .defaultValue(origParameters.baseValues.cellCopyMutationCellProperties)
                .tooltip("This type of mutation changes a random property (e.g. (input) execution order number, required energy, block output and "
                         "function-specific properties such as minimum density for sensors, neural net weights etc.). The spatial structure, color, cell "
                         "function type and self-replication capabilities are not changed. This mutation is applied to each encoded cell in the genome."),
            parameters.baseValues.cellCopyMutationCellProperties);
        AlienImGui::SliderFloat(
            AlienImGui::SliderFloatParameters()
                .name("Geometry")
                .textWidth(RightColumnWidth)
                .min(0.0f)
                .max(1.0f)
                .format("%.7f")
                .logarithmic(true)
                .colorDependence(true)
                .defaultValue(origParameters.baseValues.cellCopyMutationGeometry)
                .tooltip("This type of mutation changes the geometry type, connection distance, stiffness and single construction flag. The probability of "
                         "a change is given by the specified value times the number of coded cells in the genome."),
            parameters.baseValues.cellCopyMutationGeometry);
        AlienImGui::SliderFloat(
            AlienImGui::SliderFloatParameters()
                .name("Custom geometry")
                .textWidth(RightColumnWidth)
                .min(0.0f)
                .max(1.0f)
                .format("%.7f")
                .logarithmic(true)
                .colorDependence(true)
                .defaultValue(origParameters.baseValues.cellCopyMutationCustomGeometry)
                .tooltip("This type of mutation only changes angles and required connections of custom geometries. The probability of a change is given by "
                         "the specified value times the number of coded cells in the genome."),
            parameters.baseValues.cellCopyMutationCustomGeometry);
        AlienImGui::SliderFloat(
            AlienImGui::SliderFloatParameters()
                .name("Cell function type")
                .textWidth(RightColumnWidth)
                .min(0.0f)
                .max(1.0f)
                .format("%.7f")
                .logarithmic(true)
                .colorDependence(true)
                .defaultValue(origParameters.baseValues.cellCopyMutationCellFunction)
                .tooltip("This type of mutation changes the type of cell function. The changed cell function will have random properties. The probability "
                         "of a change is given by the specified value times the number of coded cells in the genome. If the flag 'Preserve "
                         "self-replication' is disabled it can also alter self-replication capabilities by changing a constructor to "
                         "something else or vice versa."),
            parameters.baseValues.cellCopyMutationCellFunction);
        AlienImGui::SliderFloat(
            AlienImGui::SliderFloatParameters()
                .name("Insertion")
                .textWidth(RightColumnWidth)
                .min(0.0f)
                .max(1.0f)
                .format("%.7f")
                .logarithmic(true)
                .colorDependence(true)
                .defaultValue(origParameters.baseValues.cellCopyMutationInsertion)
                .tooltip("This type of mutation inserts a new cell description to the genome at a random position. The probability of a change is given by "
                         "the specified value times the number of coded cells in the genome."),
            parameters.baseValues.cellCopyMutationInsertion);
        AlienImGui::SliderFloat(
            AlienImGui::SliderFloatParameters()
                .name("Deletion")
                .textWidth(RightColumnWidth)
                .min(0.0f)
                .max(1.0f)
                .format("%.7f")
                .logarithmic(true)
                .colorDependence(true)
                .defaultValue(origParameters.baseValues.cellCopyMutationDeletion)
                .tooltip("This type of mutation deletes a cell description from the genome at a random position. The probability of a change is given by "
                         "the specified value times the number of coded cells in the genome."),
            parameters.baseValues.cellCopyMutationDeletion);
        AlienImGui::SliderFloat(
            AlienImGui::SliderFloatParameters()
                .name("Translation")
                .textWidth(RightColumnWidth)
                .min(0.0f)
                .max(1.0f)
                .format("%.7f")
                .logarithmic(true)
                .colorDependence(true)
                .defaultValue(origParameters.baseValues.cellCopyMutationTranslation)
                .tooltip("This type of mutation moves a block of cell descriptions from the genome at a random position to a new random position."),
            parameters.baseValues.cellCopyMutationTranslation);
        AlienImGui::SliderFloat(
            AlienImGui::SliderFloatParameters()
                .name("Duplication")
                .textWidth(RightColumnWidth)
                .min(0.0f)
                .max(1.0f)
                .format("%.7f")
                .logarithmic(true)
                .colorDependence(true)
                .defaultValue(origParameters.baseValues.cellCopyMutationDuplication)
                .tooltip("This type of mutation copies a block of cell descriptions from the genome at a random position to a new random position."),
            parameters.baseValues.cellCopyMutationDuplication);
        AlienImGui::SliderFloat(
            AlienImGui::SliderFloatParameters()
                .name("Individual cell color")
                .textWidth(RightColumnWidth)
                .min(0.0f)
                .max(1.0f)
                .format("%.7f")
                .logarithmic(true)
                .colorDependence(true)
                .defaultValue(origParameters.baseValues.cellCopyMutationCellColor)
                .tooltip("This type of mutation alters the color of a single cell descriptions in a genome by using the specified color transitions. The "
                         "probability of a change is given by the specified value times the number of coded cells in the genome."),
            parameters.baseValues.cellCopyMutationCellColor);
        AlienImGui::SliderFloat(
            AlienImGui::SliderFloatParameters()
                .name("Sub-genome color")
                .textWidth(RightColumnWidth)
                .min(0.0f)
                .max(1.0f)
                .format("%.7f")
                .logarithmic(true)
                .colorDependence(true)
                .defaultValue(origParameters.baseValues.cellCopyMutationSubgenomeColor)
                .tooltip("This type of mutation alters the color of all cell descriptions in a sub-genome by using the specified color transitions."),
            parameters.baseValues.cellCopyMutationSubgenomeColor);
        AlienImGui::SliderFloat(
            AlienImGui::SliderFloatParameters()
                .name("Genome color")
                .textWidth(RightColumnWidth)
                .min(0.0f)
                .max(1.0f)
                .format("%.7f")
                .logarithmic(true)
                .colorDependence(true)
                .defaultValue(origParameters.baseValues.cellCopyMutationGenomeColor)
                .tooltip("This type of mutation alters the color of all cell descriptions in a genome by using the specified color transitions."),
            parameters.baseValues.cellCopyMutationGenomeColor);
        AlienImGui::CheckboxColorMatrix(
            AlienImGui::CheckboxColorMatrixParameters()
                .name("Color transitions")
                .textWidth(RightColumnWidth)
                .defaultValue(toVector<MAX_COLORS, MAX_COLORS>(origParameters.cellCopyMutationColorTransitions))
                .tooltip("The color transitions are used for color mutations. The row index indicates the source color and the column index the target "
                         "color."),
            parameters.cellCopyMutationColorTransitions);
        AlienImGui::Checkbox(
            AlienImGui::CheckboxParameters()
                .name("Prevent genome depth increase")
                .textWidth(RightColumnWidth)
                .defaultValue(origParameters.cellCopyMutationPreventDepthIncrease)
                .tooltip(std::string("A genome has a tree-like structure because it can contain sub-genomes. If this flag is activated, the mutations will "
                                     "not increase the depth of the genome structure.")),
            parameters.cellCopyMutationPreventDepthIncrease);
        auto preserveSelfReplication = !parameters.cellCopyMutationSelfReplication;
        AlienImGui::Checkbox(
            AlienImGui::CheckboxParameters()
                .name("Preserve self-replication")
                .textWidth(RightColumnWidth)
                .defaultValue(!origParameters.cellCopyMutationSelfReplication)
                .tooltip("If deactivated, a mutation can also alter self-replication capabilities in the genome by changing a constructor cell to "
                         "something else or vice versa."),
            preserveSelfReplication);
        parameters.cellCopyMutationSelfReplication = !preserveSelfReplication;
        AlienImGui::EndTreeNode();
    }

    /**
     * Attacker
     */
    ImGui::PushID("Attacker");
    if (AlienImGui::BeginTreeNode(AlienImGui::TreeNodeParameters().text("Cell function: Attacker"))) {
        AlienImGui::InputFloatColorMatrix(
            AlienImGui::InputFloatColorMatrixParameters()
                .name("Food chain color matrix")
                .max(1)
                .textWidth(RightColumnWidth)
                .tooltip(
                    "This matrix can be used to determine how well one cell can attack another cell. The color of the attacking cell correspond to the "
                    "row "
                    "number and the color of the attacked cell to the column number. A value of 0 means that the attacked cell cannot be digested, "
                    "i.e. no energy can be obtained. A value of 1 means that the maximum energy can be obtained in the digestion process.\n\nExample: "
                    "If a "
                    "zero is entered in row 2 (red) and column 3 (green), it means that red cells cannot eat green cells.")
                .defaultValue(toVector<MAX_COLORS, MAX_COLORS>(origParameters.baseValues.cellFunctionAttackerFoodChainColorMatrix)),
            parameters.baseValues.cellFunctionAttackerFoodChainColorMatrix);
        AlienImGui::SliderFloat(
            AlienImGui::SliderFloatParameters()
                .name("Attack strength")
                .textWidth(RightColumnWidth)
                .colorDependence(true)
                .logarithmic(true)
                .min(0)
                .max(0.5f)
                .defaultValue(origParameters.cellFunctionAttackerStrength)
                .tooltip("Indicates the portion of energy through which a successfully attacked cell is weakened. However, this energy portion can be "
                         "influenced by other factors adjustable within the attacker's simulation parameters."),
            parameters.cellFunctionAttackerStrength);
        AlienImGui::SliderFloat(
            AlienImGui::SliderFloatParameters()
                .name("Attack radius")
                .textWidth(RightColumnWidth)
                .colorDependence(true)
                .min(0)
                .max(3.0f)
                .defaultValue(origParameters.cellFunctionAttackerRadius)
                .tooltip("The maximum distance over which an attacker cell can attack another cell."),
            parameters.cellFunctionAttackerRadius);
        AlienImGui::InputFloatColorMatrix(
            AlienImGui::InputFloatColorMatrixParameters()
                .name("Complex creature protection")
                .textWidth(RightColumnWidth)
                .min(0)
                .max(20.0f)
                .defaultValue(toVector<MAX_COLORS, MAX_COLORS>(origParameters.baseValues.cellFunctionAttackerGenomeComplexityBonus))
                .tooltip("The larger this parameter is, the less energy can be gained by attacking creatures with more complex genomes."),
            parameters.baseValues.cellFunctionAttackerGenomeComplexityBonus);
        AlienImGui::SliderFloat(
            AlienImGui::SliderFloatParameters()
                .name("Energy cost")
                .textWidth(RightColumnWidth)
                .colorDependence(true)
                .min(0)
                .max(1.0f)
                .format("%.5f")
                .logarithmic(true)
                .defaultValue(origParameters.baseValues.cellFunctionAttackerEnergyCost)
                .tooltip("Amount of energy lost by an attempted attack of a cell in form of emitted energy particles."),
            parameters.baseValues.cellFunctionAttackerEnergyCost);
        AlienImGui::Checkbox(
            AlienImGui::CheckboxParameters()
                .name("Destroy cells")
                .textWidth(RightColumnWidth)
                .defaultValue(origParameters.cellFunctionAttackerDestroyCells)
                .tooltip("If activated, the attacker cell is able to destroy other cells. If deactivated, it only damages them."),
            parameters.cellFunctionAttackerDestroyCells);
        AlienImGui::EndTreeNode();
    }
    ImGui::PopID();

    /**
     * Constructor
     */
    if (AlienImGui::BeginTreeNode(AlienImGui::TreeNodeParameters().text("Cell function: Constructor"))) {
        AlienImGui::SliderFloat(
            AlienImGui::SliderFloatParameters()
                .name("Connection distance")
                .textWidth(RightColumnWidth)
                .colorDependence(true)
                .min(0.1f)
                .max(3.0f)
                .defaultValue(origParameters.cellFunctionConstructorConnectingCellMaxDistance)
                .tooltip("The constructor can automatically connect constructed cells to other cells in the vicinity within this distance."),
            parameters.cellFunctionConstructorConnectingCellMaxDistance);
        AlienImGui::Checkbox(
            AlienImGui::CheckboxParameters()
                .name("Completeness check")
                .textWidth(RightColumnWidth)
                .defaultValue(origParameters.cellFunctionConstructorCheckCompletenessForSelfReplication)
                .tooltip("If activated, a self-replication process can only start when all other non-self-replicating constructors in the cell network are "
                         "finished."),
            parameters.cellFunctionConstructorCheckCompletenessForSelfReplication);
        AlienImGui::EndTreeNode();
    }

    /**
     * Defender
     */
    if (AlienImGui::BeginTreeNode(AlienImGui::TreeNodeParameters().text("Cell function: Defender"))) {
        AlienImGui::SliderFloat(
            AlienImGui::SliderFloatParameters()
                .name("Anti-attacker strength")
                .textWidth(RightColumnWidth)
                .colorDependence(true)
                .min(1.0f)
                .max(5.0f)
                .defaultValue(origParameters.cellFunctionDefenderAgainstAttackerStrength)
                .tooltip("If an attacked cell is connected to defender cells or itself a defender cell the attack strength is reduced by this factor."),
            parameters.cellFunctionDefenderAgainstAttackerStrength);
        AlienImGui::SliderFloat(
            AlienImGui::SliderFloatParameters()
                .name("Anti-injector strength")
                .textWidth(RightColumnWidth)
                .colorDependence(true)
                .min(1.0f)
                .max(5.0f)
                .defaultValue(origParameters.cellFunctionDefenderAgainstInjectorStrength)
                .tooltip("If a constructor cell is attacked by an injector and connected to defender cells, the injection duration is increased by this "
                         "factor."),
            parameters.cellFunctionDefenderAgainstInjectorStrength);
        AlienImGui::EndTreeNode();
    }

    /**
     * Injector
     */
    if (AlienImGui::BeginTreeNode(AlienImGui::TreeNodeParameters().text("Cell function: Injector"))) {
        AlienImGui::SliderFloat(
            AlienImGui::SliderFloatParameters()
                .name("Injection radius")
                .textWidth(RightColumnWidth)
                .colorDependence(true)
                .min(0.1f)
                .max(4.0f)
                .defaultValue(origParameters.cellFunctionInjectorRadius)
                .tooltip("The maximum distance over which an injector cell can infect another cell."),
            parameters.cellFunctionInjectorRadius);
        AlienImGui::InputIntColorMatrix(
            AlienImGui::InputIntColorMatrixParameters()
                .name("Injection time")
                .logarithmic(true)
                .max(100000)
                .textWidth(RightColumnWidth)
                .defaultValue(toVector<MAX_COLORS, MAX_COLORS>(origParameters.cellFunctionInjectorDurationColorMatrix))
                .tooltip("The number of activations an injector cell requires to infect another cell. One activation usually takes 6 time steps. The row "
                         "number determines the color of the injector cell, while the column number corresponds to the color of the infected cell."),
            parameters.cellFunctionInjectorDurationColorMatrix);
        AlienImGui::EndTreeNode();
    }

    /**
     * Muscle
     */
    if (AlienImGui::BeginTreeNode(AlienImGui::TreeNodeParameters().text("Cell function: Muscle"))) {
        AlienImGui::Checkbox(
            AlienImGui::CheckboxParameters()
                .name("Movement toward target")
                .textWidth(RightColumnWidth)
                .defaultValue(origParameters.cellFunctionMuscleMovementTowardTargetedObject)
                .tooltip("If activated, a muscle cell in movement mode will only move if the triggering signal originates from a sensor cell that has "
                         "targeted an object. The specified angle in the input is interpreted relative to the target."),
            parameters.cellFunctionMuscleMovementTowardTargetedObject);
        AlienImGui::SliderFloat(
            AlienImGui::SliderFloatParameters()
                .name("Movement acceleration")
                .textWidth(RightColumnWidth)
                .colorDependence(true)
                .min(0)
                .max(0.4f)
                .logarithmic(true)
                .defaultValue(origParameters.cellFunctionMuscleMovementAcceleration)
                .tooltip("The maximum value by which a muscle cell can modify its velocity during activation. This parameter applies only to muscle cells "
                         "which are in movement mode."),
            parameters.cellFunctionMuscleMovementAcceleration);
        AlienImGui::SliderFloat(
            AlienImGui::SliderFloatParameters()
                .name("Contraction and expansion delta")
                .textWidth(RightColumnWidth)
                .colorDependence(true)
                .min(0)
                .max(0.1f)
                .defaultValue(origParameters.cellFunctionMuscleContractionExpansionDelta)
                .tooltip("The maximum length that a muscle cell can shorten or lengthen a cell connection. This parameter applies only to muscle cells "
                         "which are in contraction/expansion mode."),
            parameters.cellFunctionMuscleContractionExpansionDelta);
        AlienImGui::SliderFloat(
            AlienImGui::SliderFloatParameters()
                .name("Bending angle")
                .textWidth(RightColumnWidth)
                .colorDependence(true)
                .min(0)
                .max(10.0f)
                .defaultValue(origParameters.cellFunctionMuscleBendingAngle)
                .tooltip("The maximum value by which a muscle cell can increase/decrease the angle between two cell connections. This parameter applies "
                         "only to muscle cells which are in bending mode."),
            parameters.cellFunctionMuscleBendingAngle);
        AlienImGui::SliderFloat(
            AlienImGui::SliderFloatParameters()
                .name("Bending acceleration")
                .textWidth(RightColumnWidth)
                .colorDependence(true)
                .min(0)
                .max(0.5f)
                .defaultValue(origParameters.cellFunctionMuscleBendingAcceleration)
                .tooltip("The maximum value by which a muscle cell can modify its velocity during a bending action. This parameter applies "
                         "only to muscle cells which are in bending mode."),
            parameters.cellFunctionMuscleBendingAcceleration);
        AlienImGui::EndTreeNode();
    }

    /**
     * Sensor
     */
    if (AlienImGui::BeginTreeNode(AlienImGui::TreeNodeParameters().text("Cell function: Sensor"))) {
        AlienImGui::SliderFloat(
            AlienImGui::SliderFloatParameters()
                .name("Radius")
                .textWidth(RightColumnWidth)
                .colorDependence(true)
                .min(10.0f)
                .max(800.0f)
                .defaultValue(origParameters.cellFunctionSensorRange)
                .tooltip("The maximum radius in which a sensor cell can detect mass concentrations."),
            parameters.cellFunctionSensorRange);
        AlienImGui::EndTreeNode();
    }

    /**
     * Transmitter
     */
    if (AlienImGui::BeginTreeNode(AlienImGui::TreeNodeParameters().text("Cell function: Transmitter"))) {
        AlienImGui::SliderFloat(
            AlienImGui::SliderFloatParameters()
                .name("Energy distribution radius")
                .textWidth(RightColumnWidth)
                .colorDependence(true)
                .min(0)
                .max(5.0f)
                .defaultValue(origParameters.cellFunctionTransmitterEnergyDistributionRadius)
                .tooltip("The maximum distance over which a transmitter cell transfers its additional energy to nearby transmitter or constructor cells."),
            parameters.cellFunctionTransmitterEnergyDistributionRadius);
        AlienImGui::SliderFloat(
            AlienImGui::SliderFloatParameters()
                .name("Energy distribution Value")
                .textWidth(RightColumnWidth)
                .colorDependence(true)
                .min(0)
                .max(20.0f)
                .defaultValue(origParameters.cellFunctionTransmitterEnergyDistributionValue)
                .tooltip("The amount of energy which a transmitter cell can transfer to nearby transmitter or constructor cells or to connected cells."),
            parameters.cellFunctionTransmitterEnergyDistributionValue);
        AlienImGui::Checkbox(
            AlienImGui::CheckboxParameters()
                .name("Same creature energy distribution")
                .textWidth(RightColumnWidth)
                .defaultValue(origParameters.cellFunctionTransmitterEnergyDistributionSameCreature)
                .tooltip("If activated, the transmitter cells can only transfer energy to nearby cells belonging to the same creature."),
            parameters.cellFunctionTransmitterEnergyDistributionSameCreature);
        AlienImGui::EndTreeNode();
    }

    /**
     * Reconnector
     */
    if (AlienImGui::BeginTreeNode(AlienImGui::TreeNodeParameters().text("Cell function: Reconnector"))) {
        AlienImGui::SliderFloat(
            AlienImGui::SliderFloatParameters()
                .name("Radius")
                .textWidth(RightColumnWidth)
                .colorDependence(true)
                .min(0.0f)
                .max(3.0f)
                .defaultValue(origParameters.cellFunctionReconnectorRadius)
                .tooltip("The maximum radius in which a reconnector cell can establish or destroy connections to other cells."),
            parameters.cellFunctionReconnectorRadius);
        AlienImGui::EndTreeNode();
    }

    /**
     * Detonator
     */
    if (AlienImGui::BeginTreeNode(AlienImGui::TreeNodeParameters().text("Cell function: Detonator"))) {
        AlienImGui::SliderFloat(
            AlienImGui::SliderFloatParameters()
                .name("Blast radius")
                .textWidth(RightColumnWidth)
                .colorDependence(true)
                .min(0.0f)
                .max(10.0f)
                .defaultValue(origParameters.cellFunctionDetonatorRadius)
                .tooltip("The radius of the detonation."),
            parameters.cellFunctionDetonatorRadius);
        AlienImGui::SliderFloat(
            AlienImGui::SliderFloatParameters()
                .name("Chain explosion probability")
                .textWidth(RightColumnWidth)
                .colorDependence(true)
                .min(0.0f)
                .max(1.0f)
                .defaultValue(origParameters.cellFunctionDetonatorChainExplosionProbability)
                .tooltip("The probability that the explosion of one detonator will trigger the explosion of other detonators within the blast radius."),
            parameters.cellFunctionDetonatorChainExplosionProbability);
        AlienImGui::EndTreeNode();
    }

    /**
     * Expert settings: Advanced absorption control
     */
    if (parameters.features.advancedAbsorptionControl) {
        if (AlienImGui::BeginTreeNode(AlienImGui::TreeNodeParameters().text("Expert settings: Advanced energy absorption control"))) {
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
            AlienImGui::EndTreeNode();
        }
    }

    /**
     * Expert settings: Advanced attacker control
     */
    if (parameters.features.advancedAttackerControl) {
        if (AlienImGui::BeginTreeNode(AlienImGui::TreeNodeParameters().text("Expert settings: Advanced attacker control"))) {
            AlienImGui::InputFloatColorMatrix(
                AlienImGui::InputFloatColorMatrixParameters()
                    .name("Same mutant protection")
                    .textWidth(RightColumnWidth)
                    .min(0)
                    .max(1.0f)
                    .defaultValue(toVector<MAX_COLORS, MAX_COLORS>(origParameters.cellFunctionAttackerSameMutantPenalty))
                    .tooltip("The larger this parameter is, the less energy can be gained by attacking creatures with the same mutation id."),
                parameters.cellFunctionAttackerSameMutantPenalty);
            AlienImGui::InputFloatColorMatrix(
                AlienImGui::InputFloatColorMatrixParameters()
                    .name("New complex mutant protection")
                    .textWidth(RightColumnWidth)
                    .min(0)
                    .max(1.0f)
                    .defaultValue(toVector<MAX_COLORS, MAX_COLORS>(origParameters.baseValues.cellFunctionAttackerNewComplexMutantPenalty))
                    .tooltip("A high value protects new mutants with equal or greater genome complexity from being attacked."),
                parameters.baseValues.cellFunctionAttackerNewComplexMutantPenalty);
            AlienImGui::SliderFloat(
                AlienImGui::SliderFloatParameters()
                    .name("Sensor detection factor")
                    .textWidth(RightColumnWidth)
                    .colorDependence(true)
                    .min(0)
                    .max(1.0f)
                    .defaultValue(origParameters.cellFunctionAttackerSensorDetectionFactor)
                    .tooltip(
                        "This parameter controls whether the target must be previously detected with sensors in order to be attacked. The larger this "
                        "value is, the less energy can be gained during the attack if the target has not already been detected. For this purpose, the "
                        "attacker "
                        "cell searches for connected (or connected-connected) sensor cells to see which cell networks they have detected last time and "
                        "compares them with the attacked target."),
                parameters.cellFunctionAttackerSensorDetectionFactor);
            AlienImGui::SliderFloat(
                AlienImGui::SliderFloatParameters()
                    .name("Geometry penalty")
                    .textWidth(RightColumnWidth)
                    .colorDependence(true)
                    .min(0)
                    .max(5.0f)
                    .defaultValue(origParameters.baseValues.cellFunctionAttackerGeometryDeviationExponent)
                    .tooltip("The larger this value is, the less energy a cell can gain from an attack if the local "
                             "geometry of the attacked cell does not match the attacking cell."),
                parameters.baseValues.cellFunctionAttackerGeometryDeviationExponent);
            AlienImGui::SliderFloat(
                AlienImGui::SliderFloatParameters()
                    .name("Connections mismatch penalty")
                    .textWidth(RightColumnWidth)
                    .colorDependence(true)
                    .min(0)
                    .max(1.0f)
                    .defaultValue(origParameters.baseValues.cellFunctionAttackerConnectionsMismatchPenalty)
                    .tooltip("The larger this parameter is, the more difficult it is to attack cells that contain more connections."),
                parameters.baseValues.cellFunctionAttackerConnectionsMismatchPenalty);
            AlienImGui::SliderFloat(
                AlienImGui::SliderFloatParameters()
                    .name("Color inhomogeneity factor")
                    .textWidth(RightColumnWidth)
                    .colorDependence(true)
                    .min(0)
                    .max(2.0f)
                    .defaultValue(origParameters.cellFunctionAttackerColorInhomogeneityFactor)
                    .tooltip("If the attacked cell is connected to cells with different colors, this factor affects the energy of the captured energy."),
                parameters.cellFunctionAttackerColorInhomogeneityFactor);
            AlienImGui::SliderFloat(
                AlienImGui::SliderFloatParameters()
                    .name("Energy distribution radius")
                    .textWidth(RightColumnWidth)
                    .colorDependence(true)
                    .min(0)
                    .max(5.0f)
                    .defaultValue(origParameters.cellFunctionAttackerEnergyDistributionRadius)
                    .tooltip("The maximum distance over which an attacker cell transfers its energy captured during an attack to nearby transmitter or "
                             "constructor cells."),
                parameters.cellFunctionAttackerEnergyDistributionRadius);
            AlienImGui::SliderFloat(
                AlienImGui::SliderFloatParameters()
                    .name("Energy distribution Value")
                    .textWidth(RightColumnWidth)
                    .colorDependence(true)
                    .min(0)
                    .max(20.0f)
                    .defaultValue(origParameters.cellFunctionAttackerEnergyDistributionValue)
                    .tooltip("The amount of energy which a attacker cell can transfer to nearby transmitter or constructor cells or to connected cells."),
                parameters.cellFunctionAttackerEnergyDistributionValue);
            AlienImGui::EndTreeNode();
        }
    }

    /**
     * Expert settings: Cell color transition rules
     */
    if (parameters.features.cellColorTransitionRules) {
        if (AlienImGui::BeginTreeNode(AlienImGui::TreeNodeParameters().text("Expert settings: Cell color transition rules"))) {
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
                    widgetParameters,
                    color,
                    parameters.baseValues.cellColorTransitionTargetColor[color],
                    parameters.baseValues.cellColorTransitionDuration[color]);
                ImGui::PopID();
            }
            AlienImGui::EndTreeNode();
        }
    }

    /**
     * Expert settings: Cell age limiter
     */
    if (parameters.features.cellAgeLimiter) {
        if (AlienImGui::BeginTreeNode(AlienImGui::TreeNodeParameters().text("Expert settings: Cell age limiter"))) {
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
                    .disabledValue(parameters.baseValues.cellInactiveMaxAge)
                    .defaultEnabledValue(&origParameters.cellInactiveMaxAgeActivated)
                    .defaultValue(origParameters.baseValues.cellInactiveMaxAge)
                    .tooltip("Here, you can set the maximum age for a cell whose function or those of its neighbors have not been triggered. Cells which "
                             "are in state 'Under construction' are not affected by this option."),
                parameters.baseValues.cellInactiveMaxAge,
                &parameters.cellInactiveMaxAgeActivated);
            AlienImGui::SliderInt(
                AlienImGui::SliderIntParameters()
                    .name("Maximum emergent cell age")
                    .textWidth(RightColumnWidth)
                    .colorDependence(true)
                    .min(1)
                    .max(10000000)
                    .logarithmic(true)
                    .infinity(true)
                    .disabledValue(parameters.cellEmergentMaxAge)
                    .defaultEnabledValue(&origParameters.cellEmergentMaxAgeActivated)
                    .defaultValue(origParameters.cellEmergentMaxAge)
                    .tooltip("The maximal age of cells that arise from energy particles can be set here."),
                parameters.cellEmergentMaxAge,
                &parameters.cellEmergentMaxAgeActivated);
            AlienImGui::Checkbox(
                AlienImGui::CheckboxParameters()
                    .name("Reset age after construction")
                    .textWidth(RightColumnWidth)
                    .defaultValue(origParameters.cellResetAgeAfterActivation)
                    .tooltip("If this option is activated, the age of the cells is reset to 0 after the construction of their cell network is completed, "
                             "i.e. when the state of the cells changes from 'Under construction' to 'Ready'. This option is particularly useful if a low "
                             "'Maximum "
                             "inactive cell age' is set, as cell networks that are under construction are inactive and could die immediately after "
                             "completion if their construction takes a long time."),
                parameters.cellResetAgeAfterActivation);
            AlienImGui::SliderInt(
                AlienImGui::SliderIntParameters()
                    .name("Maximum age balancing")
                    .textWidth(RightColumnWidth)
                    .logarithmic(true)
                    .min(1000)
                    .max(1000000)
                    .disabledValue(&parameters.cellMaxAgeBalancerInterval)
                    .defaultEnabledValue(&origParameters.cellMaxAgeBalancer)
                    .defaultValue(&origParameters.cellMaxAgeBalancerInterval)
                    .tooltip("Adjusts the maximum age at regular intervals. It increases the maximum age for the cell color where the fewest "
                             "replicators exist. "
                             "Conversely, the maximum age is decreased for the cell color with the most replicators."),
                &parameters.cellMaxAgeBalancerInterval,
                &parameters.cellMaxAgeBalancer);
            AlienImGui::EndTreeNode();
        }
    }

    /**
     * Expert settings: Customize neuron mutations
     */
    if (parameters.features.customizeNeuronMutations) {
        if (AlienImGui::BeginTreeNode(AlienImGui::TreeNodeParameters().text("Expert settings: Customize neuron mutations"))) {
            AlienImGui::SliderFloat(
                AlienImGui::SliderFloatParameters()
                    .name("Affected weights")
                    .textWidth(RightColumnWidth)
                    .min(0.0f)
                    .max(1.0f)
                    .format("%.3f")
                    .defaultValue(&origParameters.cellCopyMutationNeuronDataWeight),
                &parameters.cellCopyMutationNeuronDataWeight);
            AlienImGui::SliderFloat(
                AlienImGui::SliderFloatParameters()
                    .name("Affected biases")
                    .textWidth(RightColumnWidth)
                    .min(0.0f)
                    .max(1.0f)
                    .format("%.3f")
                    .defaultValue(&origParameters.cellCopyMutationNeuronDataBias),
                &parameters.cellCopyMutationNeuronDataBias);
            AlienImGui::SliderFloat(
                AlienImGui::SliderFloatParameters()
                    .name("Affected activation functions")
                    .textWidth(RightColumnWidth)
                    .min(0.0f)
                    .max(1.0f)
                    .format("%.3f")
                    .defaultValue(&origParameters.cellCopyMutationNeuronDataActivationFunction),
                &parameters.cellCopyMutationNeuronDataActivationFunction);
            AlienImGui::SliderFloat(
                AlienImGui::SliderFloatParameters()
                    .name("Reinforcement factor")
                    .textWidth(RightColumnWidth)
                    .min(1.0f)
                    .max(2.0f)
                    .format("%.3f")
                    .defaultValue(&origParameters.cellCopyMutationNeuronDataReinforcement),
                &parameters.cellCopyMutationNeuronDataReinforcement);
            AlienImGui::SliderFloat(
                AlienImGui::SliderFloatParameters()
                    .name("Damping factor")
                    .textWidth(RightColumnWidth)
                    .min(1.0f)
                    .max(2.0f)
                    .format("%.3f")
                    .defaultValue(&origParameters.cellCopyMutationNeuronDataDamping),
                &parameters.cellCopyMutationNeuronDataDamping);
            AlienImGui::SliderFloat(
                AlienImGui::SliderFloatParameters()
                    .name("Offset")
                    .textWidth(RightColumnWidth)
                    .min(0.0f)
                    .max(0.2f)
                    .format("%.3f")
                    .defaultValue(&origParameters.cellCopyMutationNeuronDataOffset),
                &parameters.cellCopyMutationNeuronDataOffset);
            AlienImGui::EndTreeNode();
        }
    }


    /**
     * Expert settings: Cell glow
     */
    if (parameters.features.cellGlow) {
        if (AlienImGui::BeginTreeNode(AlienImGui::TreeNodeParameters().text("Expert settings: Cell glow"))) {
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
            AlienImGui::EndTreeNode();
        }
    }

    /**
     * Expert settings: External energy control
     */
    if (parameters.features.externalEnergyControl) {
        if (AlienImGui::BeginTreeNode(AlienImGui::TreeNodeParameters().text("Expert settings: External energy control"))) {
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
                    .tooltip(
                        "Here one can specify the fraction of energy transferred to constructor cells if they can provide the remaining energy for the "
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
            AlienImGui::EndTreeNode();
        }
    }

    /**
     * Expert settings: Genome complexity measurement
     */
    if (parameters.features.genomeComplexityMeasurement) {
        if (AlienImGui::BeginTreeNode(AlienImGui::TreeNodeParameters().text("Expert settings: Genome complexity measurement"))) {
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
            AlienImGui::SliderFloat(
                AlienImGui::SliderFloatParameters()
                    .name("Neuron factor")
                    .textWidth(RightColumnWidth)
                    .colorDependence(true)
                    .min(0.0f)
                    .max(4.0f)
                    .format("%.2f")
                    .defaultValue(origParameters.genomeComplexityNeuronFactor)
                    .tooltip("This parameter takes into account the number of encoded neurons in the genome for the complexity value."),
                parameters.genomeComplexityNeuronFactor);
            AlienImGui::EndTreeNode();
        }
    }

    /**
     * Expert settings: Legacy behavior
     */
    if (parameters.features.legacyModes) {
        if (AlienImGui::BeginTreeNode(AlienImGui::TreeNodeParameters().text("Expert settings: Legacy behavior"))) {
            AlienImGui::Checkbox(
                AlienImGui::CheckboxParameters()
                    .name("Fetch angle from adjacent sensor")
                    .textWidth(RightColumnWidth)
                    .defaultValue(origParameters.legacyCellFunctionMuscleMovementAngleFromSensor)
                    .tooltip("This parameter changes the behavior of the parameter 'Movement toward target'. If activated, the muscle cell fetches the "
                             "movement angle directly from a connected (or connected-connected) sensor cell that has previously detected a target "
                             "(legacy behavior). If deactivated, the input signal must only originate from a sensor cell and must not be adjacent (new "
                             "behavior)."),
                parameters.legacyCellFunctionMuscleMovementAngleFromSensor);
            AlienImGui::EndTreeNode();
        }
    }

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

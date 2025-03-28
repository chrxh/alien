#include "SimulationParametersSpecificationService.h"

#include <Fonts/IconsFontAwesome5.h>

#include "CellTypeStrings.h"
#include "LocationHelper.h"
#include "SimulationParametersEditService.h"

#define BASE_VALUE_OFFSET(X) offsetof(SimulationParameters, X)
#define ZONE_VALUE_OFFSET(X) offsetof(SimulationParametersZoneValues, X)
#define ZONE_ENABLED_VALUE_OFFSET(X) offsetof(SimulationParametersZoneEnabledValues, X)
#define EXPERT_VALUE_OFFSET(X) offsetof(ExpertSettingsToggles, X)

ParametersSpec const& SimulationParametersSpecificationService::getSpec()
{
    if (!_parametersSpec.has_value()) {
        createSpec();
    }
    return _parametersSpec.value();
}

bool* SimulationParametersSpecificationService::getPinnedValueRef(ValueSpec const& spec, SimulationParameters& parameters, int locationIndex) const
{
    if (std::holds_alternative<BaseValueSpec>(spec)) {
        auto baseValueSpec = std::get<BaseValueSpec>(spec);
        if (baseValueSpec._pinnedAddress.has_value()) {
            return reinterpret_cast<bool*>(reinterpret_cast<char*>(&parameters) + baseValueSpec._pinnedAddress.value());
        }
    }
    return nullptr;
}

bool* SimulationParametersSpecificationService::getEnabledValueRef(ValueSpec const& spec, SimulationParameters& parameters, int locationIndex) const
{
    if (std::holds_alternative<BaseValueSpec>(spec)) {
        auto baseValueSpec = std::get<BaseValueSpec>(spec);
        if (baseValueSpec._enabledValueAddress.has_value()) {
            return reinterpret_cast<bool*>(reinterpret_cast<char*>(&parameters) + baseValueSpec._enabledValueAddress.value());
        }
    } else if (std::holds_alternative<BaseZoneValueSpec>(spec)) {
        auto baseZoneValueSpec = std::get<BaseZoneValueSpec>(spec);

        if (locationIndex == 0 && baseZoneValueSpec._enabledBaseValueAddress.has_value()) {
            return reinterpret_cast<bool*>(reinterpret_cast<char*>(&parameters) + baseZoneValueSpec._enabledBaseValueAddress.value());
        }
        for (int i = 0; i < parameters.numZones; ++i) {
            if (parameters.zone[i].locationIndex == locationIndex && baseZoneValueSpec._enabledZoneValueAddress.has_value()) {
                return reinterpret_cast<bool*>(
                    reinterpret_cast<char*>(&parameters.zone[i].enabledValues) + baseZoneValueSpec._enabledZoneValueAddress.value());
            }
        }
    }
    return nullptr;
}

bool* SimulationParametersSpecificationService::getExpertToggleValueRef(ParameterGroupSpec const& spec, SimulationParameters& parameters) const
{
    return reinterpret_cast<bool*>(reinterpret_cast<char*>(&parameters.expertSettingsToggles) + spec._expertToggleAddress.value());
}

void SimulationParametersSpecificationService::createSpec()
{
    std::vector<std::pair<std::string, std::vector<ParameterSpec>>> cellTypeStrings;
    for (int i = 0; i < CellType_Count; ++i) {
        cellTypeStrings.emplace_back(std::make_pair(Const::CellTypeToStringMap.at(i), std::vector<ParameterSpec>()));
    }

    auto radiationStrengthGetter = [](SimulationParameters const& parameters, int locationIndex) {
        auto strength = SimulationParametersEditService::get().getRadiationStrengths(parameters);
        if (locationIndex == 0) {
            return strength.values.at(0);
        } else {
            auto sourceIndex = LocationHelper::findLocationArrayIndex(parameters, locationIndex);
            return parameters.radiationSource[sourceIndex].strength;
        }
    };

    auto radiationStrengthSetter = [](float value, SimulationParameters& parameters, int locationIndex) {
        auto& editService = SimulationParametersEditService::get();
        auto strength = SimulationParametersEditService::get().getRadiationStrengths(parameters);
        auto editedStrength = strength;
        int strengthIndex = locationIndex == 0 ? 0 : LocationHelper::findLocationArrayIndex(parameters, locationIndex) + 1;
        editedStrength.values.at(strengthIndex) = value;
        editService.adaptRadiationStrengths(editedStrength, strength, strengthIndex);
        editService.applyRadiationStrengths(parameters, editedStrength);
    };

    std::string const coloringTooltip =
        "Here, one can set how the cells are to be colored during rendering. \n\n" ICON_FA_CHEVRON_RIGHT
        " Energy: The more energy a cell has, the brighter it is displayed. A grayscale is used.\n\n" ICON_FA_CHEVRON_RIGHT
        " Standard cell colors: Each cell is assigned one of 7 default colors, which is displayed with this option. \n\n" ICON_FA_CHEVRON_RIGHT
        " Mutants: Different mutants are represented by different colors (only larger structural mutations such as translations or "
        "duplications are taken into account).\n\n" ICON_FA_CHEVRON_RIGHT
        " Mutant and cell function: Combination of mutants and cell function coloring.\n\n" ICON_FA_CHEVRON_RIGHT
        " Cell state: blue = ready, green = under construction, white = activating, pink = detached, pale blue = reviving, red = "
        "dying\n\n" ICON_FA_CHEVRON_RIGHT
        " Genome complexity: This property can be utilized by attacker cells when the parameter 'Complex creature protection' is "
        "activated (see tooltip there). The coloring is as follows: blue = creature with low bonus (usually small or simple genome structure), "
        "red = large bonus\n\n" ICON_FA_CHEVRON_RIGHT " Specific cell function: A specific type of cell function can be highlighted, which is "
        "selected in the next parameter.\n\n" ICON_FA_CHEVRON_RIGHT " Every cell function: The cells are colored according to their cell function.";

    _parametersSpec =  ParametersSpec().groups({
        ParameterGroupSpec().name("General").parameters({
            ParameterSpec().name("Project name").value(BaseValueSpec().valueAddress(BASE_VALUE_OFFSET(projectName))).type(Char64Spec()),
        }),
        ParameterGroupSpec()
            .name("Visualization")
            .parameters({
                ParameterSpec().name("Background color").value(BaseZoneValueSpec().valueAddress(ZONE_VALUE_OFFSET(backgroundColor))).type(ColorPickerSpec()),
                ParameterSpec()
                    .name("Primary cell coloring")
                    .value(BaseValueSpec().valueAddress(BASE_VALUE_OFFSET(primaryCellColoring)))
                    .type(AlternativeSpec().alternatives({
                        {"Energy", {}},
                        {"Standard cell color", {}},
                        {"Mutant", {}},
                        {"Mutant and cell function", {}},
                        {"Cell state", {}},
                        {"Genome complexity", {}},
                        {"Specific cell function",
                         {ParameterSpec()
                              .name("Highlighted cell function")
                              .value(BaseValueSpec().valueAddress(BASE_VALUE_OFFSET(highlightedCellType)))
                              .tooltip("The specific cell function type to be highlighted can be selected here.")
                              .type(AlternativeSpec().alternatives(cellTypeStrings))}},
                        {"Every cell function", {}},
                    }))
                    .tooltip(coloringTooltip),
                ParameterSpec()
                    .name("Cell radius")
                    .value(BaseValueSpec().valueAddress(BASE_VALUE_OFFSET(cellRadius)))
                    .type(FloatSpec().min(0.0f).max(0.5f))
                    .tooltip("Specifies the radius of the drawn cells in unit length."),
                ParameterSpec()
                    .name("Zoom level for neural activity")
                    .value(BaseValueSpec().valueAddress(BASE_VALUE_OFFSET(zoomLevelForNeuronVisualization)))
                    .type(FloatSpec().min(0.0f).max(32.f).infinity(true))
                    .tooltip("The zoom level from which the neuronal activities become visible."),
                ParameterSpec()
                    .name("Attack visualization")
                    .value(BaseValueSpec().valueAddress(BASE_VALUE_OFFSET(attackVisualization)))
                    .type(BoolSpec())
                    .tooltip("If activated, successful attacks of attacker cells are visualized."),
                ParameterSpec()
                    .name("Muscle movement visualization")
                    .value(BaseValueSpec().valueAddress(BASE_VALUE_OFFSET(muscleMovementVisualization)))
                    .type(BoolSpec())
                    .tooltip("If activated, the direction in which muscle cells are moving are visualized."),
                ParameterSpec()
                    .name("Borderless rendering")
                    .value(BaseValueSpec().valueAddress(BASE_VALUE_OFFSET(borderlessRendering)))
                    .type(BoolSpec())
                    .tooltip("If activated, the simulation is rendered periodically in the view port."),
                ParameterSpec()
                    .name("Grid lines")
                    .value(BaseValueSpec().valueAddress(BASE_VALUE_OFFSET(gridLines)))
                    .type(BoolSpec())
                    .tooltip("This option draws a suitable grid in the background depending on the zoom level."),
                ParameterSpec()
                    .name("Mark reference domain")
                    .value(BaseValueSpec().valueAddress(BASE_VALUE_OFFSET(markReferenceDomain)))
                    .type(BoolSpec())
                    .tooltip("This option draws a suitable grid in the background depending on the zoom level."),
                ParameterSpec()
                    .name("Show radiation sources")
                    .value(BaseValueSpec().valueAddress(BASE_VALUE_OFFSET(showRadiationSources)))
                    .type(BoolSpec())
                    .tooltip("This option draws red crosses in the center of radiation sources."),
            }),
        ParameterGroupSpec()
            .name("Numerics")
            .parameters({
                ParameterSpec()
                    .name("Time step size")
                    .value(BaseValueSpec().valueAddress(BASE_VALUE_OFFSET(timestepSize)))
                    .type(FloatSpec().min(0.0f).max(1.0f))
                    .tooltip("The time duration calculated in a single simulation step. Smaller values increase the accuracy of the simulation while larger "
                             "values can lead to numerical instabilities."),
            }),
        ParameterGroupSpec()
            .name("Physics: Motion")
            .parameters({
                ParameterSpec()
                    .name("Motion type")
                    .value(BaseValueSpec().valueAddress(BASE_VALUE_OFFSET(motionType)))
                    .type(AlternativeSpec().alternatives(
                        {{"Fluid solver",
                          {
                              ParameterSpec()
                                  .name("Smoothing length")
                                  .value(BaseValueSpec().valueAddress(BASE_VALUE_OFFSET(smoothingLength)))
                                  .tooltip("The smoothing length determines the region of influence of the neighboring particles for the calculation of "
                                           "density, pressure and viscosity. Values that are too small lead to numerical instabilities, while values that "
                                           "are too large cause the particles to drift apart.")
                                  .type(FloatSpec().min(0).max(3.0f)),
                              ParameterSpec()
                                  .name("Pressure")
                                  .value(BaseValueSpec().valueAddress(BASE_VALUE_OFFSET(pressureStrength)))
                                  .tooltip("This parameter allows to control the strength of the pressure.")
                                  .type(FloatSpec().min(0).max(0.3f)),
                              ParameterSpec()
                                  .name("Viscosity")
                                  .value(BaseValueSpec().valueAddress(BASE_VALUE_OFFSET(viscosityStrength)))
                                  .tooltip("This parameter be used to control the strength of the viscosity. Larger values lead to a smoother movement.")
                                  .type(FloatSpec().min(0).max(0.3f)),
                          }},
                         {"Collision-based solver",
                          {
                              ParameterSpec()
                                  .name("Repulsion strength")
                                  .value(BaseValueSpec().valueAddress(BASE_VALUE_OFFSET(repulsionStrength)))
                                  .tooltip("The strength of the repulsive forces, between two cells that are not connected.")
                                  .type(FloatSpec().min(0).max(0.3f)),
                              ParameterSpec()
                                  .name("Maximum collision distance")
                                  .value(BaseValueSpec().valueAddress(BASE_VALUE_OFFSET(maxCollisionDistance)))
                                  .tooltip("Maximum distance up to which a collision of two cells is possible.")
                                  .type(FloatSpec().min(0).max(3.0f)),
                          }}}))
                    .tooltip(std::string(
                        "The algorithm for the particle motions is defined here. If 'Fluid dynamics' is selected, an SPH fluid solver is used for the "
                        "calculation of the forces. The particles then behave like (compressible) liquids or gases. The other option 'Collision-based' "
                        "calculates the forces based on particle collisions and should be preferred for mechanical simulation with solids.")),
                ParameterSpec()
                    .name("Friction")
                    .value(BaseZoneValueSpec().valueAddress(ZONE_VALUE_OFFSET(friction)).enabledZoneValueAddress(ZONE_ENABLED_VALUE_OFFSET(friction)))
                    .type(FloatSpec().min(0).max(1.0f).logarithmic(true).format("%.4f"))
                    .tooltip("This specifies the fraction of the velocity that is slowed down per time step."),
                ParameterSpec()
                    .name("Rigidity")
                    .value(BaseZoneValueSpec().valueAddress(ZONE_VALUE_OFFSET(rigidity)).enabledZoneValueAddress(ZONE_ENABLED_VALUE_OFFSET(rigidity)))
                    .type(FloatSpec().min(0).max(3.0f))
                    .tooltip("Controls the rigidity of connected cells. A higher value will cause connected cells to move more uniformly as a rigid body."),
            }),
        ParameterGroupSpec()
            .name("Physics: Thresholds")
            .parameters({
                ParameterSpec()
                    .name("Maximum velocity")
                    .value(BaseValueSpec().valueAddress(BASE_VALUE_OFFSET(maxVelocity)))
                    .type(FloatSpec().min(0.0f).max(6.0f))
                    .tooltip("Maximum velocity that a cell can reach."),
                ParameterSpec()
                    .name("Maximum force")
                    .value(BaseZoneValueSpec().valueAddress(ZONE_VALUE_OFFSET(cellMaxForce)).enabledZoneValueAddress(ZONE_ENABLED_VALUE_OFFSET(cellMaxForce)))
                    .type(FloatSpec().min(0.0f).max(3.0f))
                    .colorDependence(ColorDependence::Vector)
                    .tooltip("Maximum force that can be applied to a cell without causing it to disintegrate."),
                ParameterSpec()
                    .name("Minimum distance")
                    .value(BaseValueSpec().valueAddress(BASE_VALUE_OFFSET(minCellDistance)))
                    .type(FloatSpec().min(0.0f).max(1.0f))
                    .tooltip("Minimum distance between two cells."),
            }),
        ParameterGroupSpec()
            .name("Physics: Binding")
            .parameters({
                ParameterSpec()
                    .name("Maximum distance")
                    .value(BaseValueSpec().valueAddress(BASE_VALUE_OFFSET(maxBindingDistance)))
                    .type(FloatSpec().min(0.0f).max(5.0f))
                    .colorDependence(ColorDependence::Vector)
                    .tooltip("Maximum distance up to which a connection of two cells is possible."),
                ParameterSpec()
                    .name("Fusion velocity")
                    .value(BaseZoneValueSpec()
                               .valueAddress(ZONE_VALUE_OFFSET(cellFusionVelocity))
                               .enabledZoneValueAddress(ZONE_ENABLED_VALUE_OFFSET(cellFusionVelocity)))
                    .type(FloatSpec().min(0.0f).max(2.0f))
                    .tooltip("Maximum force that can be applied to a cell without causing it to disintegrate."),
                ParameterSpec()
                    .name("Maximum energy")
                    .value(BaseZoneValueSpec().valueAddress(ZONE_VALUE_OFFSET(cellMaxBindingEnergy)).enabledZoneValueAddress(ZONE_ENABLED_VALUE_OFFSET(cellMaxBindingEnergy)))
                    .type(FloatSpec().min(50.0f).max(10000000.0f).logarithmic(true).infinity(true).format("%.0f"))
                    .tooltip("Maximum energy of a cell at which it can contain bonds to adjacent cells. If the energy of a cell exceeds this "
                             "value, all bonds will be destroyed."),
            }),
        ParameterGroupSpec()
            .name("Physics: Radiation")
            .parameters({
                ParameterSpec()
                    .name("Relative strength")
                    .value(BaseValueSpec()
                               .pinnedAddress(BASE_VALUE_OFFSET(baseStrengthRatioPinned))
                               .valueGetter(radiationStrengthGetter)
                               .valueSetter(radiationStrengthSetter))
                    .type(FloatSpec().min(0.0f).max(1.0f))
                    .tooltip("Cells can emit energy particles over time. A portion of this energy can be released directly near the cell, while the rest is "
                             "utilized by one of the available radiation sources. This parameter determines the fraction of energy assigned to the emitted "
                             "energy particle in the vicinity of the cell. Values between 0 and 1 are permitted."),
                ParameterSpec()
                    .name("Absorption factor")
                    .value(BaseZoneValueSpec().valueAddress(ZONE_VALUE_OFFSET(radiationAbsorption)).enabledZoneValueAddress(ZONE_ENABLED_VALUE_OFFSET(radiationAbsorption)))
                    .type(FloatSpec().min(0.0f).max(1.0f).logarithmic(true).format("%.4f"))
                    .colorDependence(ColorDependence::Vector)
                    .tooltip("The fraction of energy that a cell can absorb from an incoming energy particle can be specified here."),
                ParameterSpec()
                    .name("Radiation type I: Strength")
                    .value(BaseZoneValueSpec().valueAddress(ZONE_VALUE_OFFSET(radiationType1_strength)).enabledZoneValueAddress(ZONE_ENABLED_VALUE_OFFSET(radiationType1_strength)))
                    .type(FloatSpec().min(0.0f).max(0.01f).logarithmic(true).format("%.6f"))
                    .colorDependence(ColorDependence::Vector)
                    .tooltip("Indicates how energetic the emitted particles of aged cells are."),
                ParameterSpec()
                    .name("Radiation type I: Minimum age")
                    .value(BaseValueSpec().valueAddress(BASE_VALUE_OFFSET(radiationType1_minimumAge)))
                    .type(IntSpec().min(0).max(10000000).logarithmic(true).infinity(true))
                    .colorDependence(ColorDependence::Vector)
                    .tooltip("The minimum age of a cell can be defined here, from which it emits energy particles."),
                ParameterSpec()
                    .name("Radiation type II: Strength")
                    .value(BaseValueSpec().valueAddress(BASE_VALUE_OFFSET(radiationType2_strength)))
                    .type(FloatSpec().min(0.0f).max(0.01f).logarithmic(true).format("%.6f"))
                    .colorDependence(ColorDependence::Vector)
                    .tooltip("Indicates how energetic the emitted particles of high energy cells are."),
                ParameterSpec()
                    .name("Radiation type II: Energy threshold")
                    .value(BaseValueSpec().valueAddress(BASE_VALUE_OFFSET(radiationType2_energyThreshold)))
                    .type(FloatSpec().min(0.0f).max(100000.0f).logarithmic(true).infinity(true).format("%.1f"))
                    .colorDependence(ColorDependence::Vector)
                    .tooltip("The minimum energy of a cell can be defined here, from which it emits energy particles."),
                ParameterSpec()
                    .name("Minimum split energy")
                    .value(BaseValueSpec().valueAddress(BASE_VALUE_OFFSET(particleSplitEnergy)))
                    .type(FloatSpec().min(1.0f).max(10000.0f).logarithmic(true).infinity(true).format("%.0f"))
                    .colorDependence(ColorDependence::Vector)
                    .tooltip("The minimum energy of an energy particle after which it can split into two particles, whereby it receives a small momentum. The "
                             "splitting does not occur immediately, but only after a certain time."),
                ParameterSpec()
                    .name("Energy to cell transformation")
                    .value(BaseValueSpec().valueAddress(BASE_VALUE_OFFSET(particleTransformationAllowed)))
                    .type(BoolSpec())
                    .tooltip("If activated, an energy particle will transform into a cell if the energy of the particle exceeds the normal energy value."),
            }),
        ParameterGroupSpec()
            .name("Cell life cycle")
            .parameters({
                ParameterSpec()
                    .name("Maximum age")
                    .value(BaseValueSpec().valueAddress(BASE_VALUE_OFFSET(maxCellAge)))
                    .type(IntSpec().min(1).max(1e7).logarithmic(true).infinity(true))
                    .colorDependence(ColorDependence::Vector)
                    .tooltip("Defines the maximum age of a cell. If a cell exceeds this age it will be transformed to an energy particle."),
                ParameterSpec()
                    .name("Minimum energy")
                    .value(BaseZoneValueSpec().valueAddress(ZONE_VALUE_OFFSET(minCellEnergy)).enabledZoneValueAddress(ZONE_ENABLED_VALUE_OFFSET(minCellEnergy)))
                    .type(FloatSpec().min(10.0f).max(200.0f))
                    .colorDependence(ColorDependence::Vector)
                    .tooltip("Minimum energy a cell needs to exist."),
                ParameterSpec()
                    .name("Normal energy")
                    .value(BaseValueSpec().valueAddress(BASE_VALUE_OFFSET(normalCellEnergy)))
                    .type(FloatSpec().min(10.0f).max(200.0f))
                    .colorDependence(ColorDependence::Vector)
                    .tooltip(
                        "The normal energy value of a cell is defined here. This is used as a reference value in various contexts: "
                        "\n\n" ICON_FA_CHEVRON_RIGHT
                        " Attacker and Transmitter cells: When the energy of these cells is above the normal value, some of their energy is distributed to "
                        "surrounding cells.\n\n" ICON_FA_CHEVRON_RIGHT
                        " Constructor cells: Creating new cells costs energy. The creation of new cells is executed only when the "
                        "residual energy of the constructor cell does not fall below the normal value.\n\n" ICON_FA_CHEVRON_RIGHT
                        " If the transformation of energy particles to "
                        "cells is activated, an energy particle will transform into a cell if the energy of the particle exceeds the normal value."),
                ParameterSpec()
                    .name("Decay rate of dying cells")
                    .value(BaseZoneValueSpec().valueAddress(ZONE_VALUE_OFFSET(cellDeathProbability)).enabledZoneValueAddress(ZONE_ENABLED_VALUE_OFFSET(cellDeathProbability)))
                    .type(FloatSpec().min(1e-6f).max(1e-1f).format("%.6f").logarithmic(true))
                    .colorDependence(ColorDependence::Vector)
                    .tooltip("The probability per time step with which a cell will disintegrate (i.e. transform into an energy particle) when it is in the "
                             "state 'Dying'. This can occur when one of the following conditions is satisfied:\n\n" ICON_FA_CHEVRON_RIGHT
                             " The cell has too low energy.\n\n" ICON_FA_CHEVRON_RIGHT " The cell has exceeded its maximum age."),
                ParameterSpec()
                    .name("Cell death consequences")
                    .value(BaseValueSpec().valueAddress(BASE_VALUE_OFFSET(cellDeathConsequences)))
                    .type(AlternativeSpec().alternatives({{"None", {}}, {"Entire creature dies", {}}, {"Detached creature parts die", {}}}))
                    .tooltip("Here one can define what happens to the organism when one of its cells is in the 'Dying' state.\n\n" ICON_FA_CHEVRON_RIGHT
                             " None: Only the cell dies.\n\n" ICON_FA_CHEVRON_RIGHT
                             " Entire creature dies: All the cells of the organism will also die.\n\n" ICON_FA_CHEVRON_RIGHT
                             " Detached creature parts die: Only the parts of the organism that are no longer connected to a "
                             "constructor cell for self-replication die."),
            }),
        ParameterGroupSpec()
            .name("Genome copy mutations")
            .parameters({
                ParameterSpec()
                    .name("Neural nets")
                    .value(BaseZoneValueSpec().valueAddress(ZONE_VALUE_OFFSET(copyMutationNeuronData)).enabledZoneValueAddress(ZONE_ENABLED_VALUE_OFFSET(copyMutationNeuronData)))
                    .type(FloatSpec().min(0.0f).max(1.0f).format("%.7f").logarithmic(true))
                    .colorDependence(ColorDependence::Vector)
                    .tooltip(
                        "This type of mutation can change the weights, biases and activation functions of neural networks of each neuron cell encoded in the "
                        "genome."),
                ParameterSpec()
                    .name("Cell properties")
                    .value(BaseZoneValueSpec().valueAddress(ZONE_VALUE_OFFSET(copyMutationCellProperties)).enabledZoneValueAddress(ZONE_ENABLED_VALUE_OFFSET(copyMutationCellProperties)))
                    .type(FloatSpec().min(0.0f).max(1.0f).format("%.7f").logarithmic(true))
                    .colorDependence(ColorDependence::Vector)
                    .tooltip("This type of mutation changes a random property (e.g. (input) execution order number, required energy, block output and "
                             "function-specific properties such as minimum density for sensors, neural net weights etc.). The spatial structure, color, cell "
                             "function type and self-replication capabilities are not changed. This mutation is applied to each encoded cell in the genome."),
                ParameterSpec()
                    .name("Geometry")
                    .value(BaseZoneValueSpec().valueAddress(ZONE_VALUE_OFFSET(copyMutationGeometry)).enabledZoneValueAddress(ZONE_ENABLED_VALUE_OFFSET(copyMutationGeometry)))
                    .type(FloatSpec().min(0.0f).max(1.0f).format("%.7f").logarithmic(true))
                    .colorDependence(ColorDependence::Vector)
                    .tooltip("This type of mutation changes the geometry type, connection distance, stiffness and single construction flag. The probability of "
                             "a change is given by the specified value times the number of coded cells in the genome."),
                ParameterSpec()
                    .name("Custom geometry")
                    .value(BaseZoneValueSpec().valueAddress(ZONE_VALUE_OFFSET(copyMutationCustomGeometry)).enabledZoneValueAddress(ZONE_ENABLED_VALUE_OFFSET(copyMutationCustomGeometry)))
                    .type(FloatSpec().min(0.0f).max(1.0f).format("%.7f").logarithmic(true))
                    .colorDependence(ColorDependence::Vector)
                    .tooltip("This type of mutation only changes angles and required connections of custom geometries. The probability of a change is given by "
                             "the specified value times the number of coded cells in the genome."),
                ParameterSpec()
                    .name("Cell function type")
                    .value(BaseZoneValueSpec().valueAddress(ZONE_VALUE_OFFSET(copyMutationCellType)).enabledZoneValueAddress(ZONE_ENABLED_VALUE_OFFSET(copyMutationCellType)))
                    .type(FloatSpec().min(0.0f).max(1.0f).format("%.7f").logarithmic(true))
                    .colorDependence(ColorDependence::Vector)
                    .tooltip("This type of mutation changes the type of cell function. The changed cell function will have random properties. The probability "
                             "of a change is given by the specified value times the number of coded cells in the genome. If the flag 'Preserve "
                             "self-replication' is disabled it can also alter self-replication capabilities by changing a constructor to "
                             "something else or vice versa."),
                ParameterSpec()
                    .name("Insertion")
                    .value(BaseZoneValueSpec().valueAddress(ZONE_VALUE_OFFSET(copyMutationInsertion)).enabledZoneValueAddress(ZONE_ENABLED_VALUE_OFFSET(copyMutationInsertion)))
                    .type(FloatSpec().min(0.0f).max(1.0f).format("%.7f").logarithmic(true))
                    .colorDependence(ColorDependence::Vector)
                    .tooltip("This type of mutation inserts a new cell description to the genome at a random position. The probability of a change is given by "
                             "the specified value times the number of coded cells in the genome."),
                ParameterSpec()
                    .name("Deletion")
                    .value(BaseZoneValueSpec().valueAddress(ZONE_VALUE_OFFSET(copyMutationDeletion)).enabledZoneValueAddress(ZONE_ENABLED_VALUE_OFFSET(copyMutationDeletion)))
                    .type(FloatSpec().min(0.0f).max(1.0f).format("%.7f").logarithmic(true))
                    .colorDependence(ColorDependence::Vector)
                    .tooltip("This type of mutation deletes a cell description from the genome at a random position. The probability of a change is given by "
                             "the specified value times the number of coded cells in the genome."),
                ParameterSpec()
                    .name("Translation")
                    .value(BaseZoneValueSpec().valueAddress(ZONE_VALUE_OFFSET(copyMutationTranslation)).enabledZoneValueAddress(ZONE_ENABLED_VALUE_OFFSET(copyMutationTranslation)))
                    .type(FloatSpec().min(0.0f).max(1.0f).format("%.7f").logarithmic(true))
                    .colorDependence(ColorDependence::Vector)
                    .tooltip("This type of mutation moves a block of cell descriptions from the genome at a random position to a new random position."),
                ParameterSpec()
                    .name("Duplication")
                    .value(BaseZoneValueSpec().valueAddress(ZONE_VALUE_OFFSET(copyMutationDuplication)).enabledZoneValueAddress(ZONE_ENABLED_VALUE_OFFSET(copyMutationDuplication)))
                    .type(FloatSpec().min(0.0f).max(1.0f).format("%.7f").logarithmic(true))
                    .colorDependence(ColorDependence::Vector)
                    .tooltip("This type of mutation copies a block of cell descriptions from the genome at a random position to a new random position."),
                ParameterSpec()
                    .name("Individual cell color")
                    .value(BaseZoneValueSpec().valueAddress(ZONE_VALUE_OFFSET(copyMutationCellColor)).enabledZoneValueAddress(ZONE_ENABLED_VALUE_OFFSET(copyMutationCellColor)))
                    .type(FloatSpec().min(0.0f).max(1.0f).format("%.7f").logarithmic(true))
                    .colorDependence(ColorDependence::Vector)
                    .tooltip("This type of mutation alters the color of a single cell descriptions in a genome by using the specified color transitions. The "
                             "probability of a change is given by the specified value times the number of coded cells in the genome."),
                ParameterSpec()
                    .name("Sub-genome color")
                    .value(BaseZoneValueSpec().valueAddress(ZONE_VALUE_OFFSET(copyMutationSubgenomeColor)).enabledZoneValueAddress(ZONE_ENABLED_VALUE_OFFSET(copyMutationSubgenomeColor)))
                    .type(FloatSpec().min(0.0f).max(1.0f).format("%.7f").logarithmic(true))
                    .colorDependence(ColorDependence::Vector)
                    .tooltip("This type of mutation alters the color of all cell descriptions in a sub-genome by using the specified color transitions."),
                ParameterSpec()
                    .name("Genome color")
                    .value(BaseZoneValueSpec().valueAddress(ZONE_VALUE_OFFSET(copyMutationGenomeColor)).enabledZoneValueAddress(ZONE_ENABLED_VALUE_OFFSET(copyMutationGenomeColor)))
                    .type(FloatSpec().min(0.0f).max(1.0f).format("%.7f").logarithmic(true))
                    .colorDependence(ColorDependence::Vector)
                    .tooltip("This type of mutation alters the color of all cell descriptions in a genome by using the specified color transitions."),
                ParameterSpec()
                    .name("Color transitions")
                    .value(BaseValueSpec().valueAddress(BASE_VALUE_OFFSET(copyMutationColorTransitions)))
                    .type(BoolSpec())
                    .colorDependence(ColorDependence::Matrix)
                    .tooltip("The color transitions are used for color mutations. The row index indicates the source color and the column index the target "
                             "color."),
                ParameterSpec()
                    .name("Prevent genome depth increase")
                    .value(BaseValueSpec().valueAddress(BASE_VALUE_OFFSET(copyMutationPreventDepthIncrease)))
                    .type(BoolSpec())
                    .tooltip("A genome has a tree-like structure because it can contain sub-genomes. If this flag is activated, the mutations will "
                             "not increase the depth of the genome structure."),
                ParameterSpec()
                    .name("Mutate self-replication")
                    .value(BaseValueSpec().valueAddress(BASE_VALUE_OFFSET(copyMutationSelfReplication)))
                    .type(BoolSpec())
                    .tooltip("If activated, a mutation can also alter self-replication capabilities in the genome by changing a constructor cell to "
                             "something else or vice versa."),
            }),
        ParameterGroupSpec()
            .name("Cell type: Attacker")
            .parameters({
                ParameterSpec()
                    .name("Energy cost")
                    .value(BaseZoneValueSpec().valueAddress(ZONE_VALUE_OFFSET(attackerEnergyCost)).enabledZoneValueAddress(ZONE_ENABLED_VALUE_OFFSET(attackerEnergyCost)))
                    .type(FloatSpec().min(0).max(1.0f).logarithmic(true).format("%.5f"))
                    .colorDependence(ColorDependence::Vector)
                    .tooltip("Amount of energy lost by an attempted attack of a cell in form of emitted energy particles."),
                ParameterSpec()
                    .name("Food chain color matrix")
                    .value(BaseZoneValueSpec().valueAddress(ZONE_VALUE_OFFSET(attackerFoodChainColorMatrix)).enabledZoneValueAddress(ZONE_ENABLED_VALUE_OFFSET(attackerFoodChainColorMatrix)))
                    .type(FloatSpec().min(0.0f).max(1.0f).format("%.2f"))
                    .colorDependence(ColorDependence::Matrix)
                    .tooltip(
                        "This matrix can be used to determine how well one cell can attack another cell. The color of the attacking cell correspond to the "
                        "row number and the color of the attacked cell to the column number. A value of 0 means that the attacked cell cannot be digested, "
                        "i.e. no energy can be obtained. A value of 1 means that the maximum energy can be obtained in the digestion process.\n\nExample: If "
                        "a zero is entered in row 2 (red) and column 3 (green), it means that red cells cannot eat green cells."),
                ParameterSpec()
                    .name("Attack strength")
                    .value(BaseValueSpec().valueAddress(BASE_VALUE_OFFSET(attackerStrength)))
                    .type(FloatSpec().min(0).max(0.5f).logarithmic(true))
                    .colorDependence(ColorDependence::Vector)
                    .tooltip("Indicates the portion of energy through which a successfully attacked cell is weakened. However, this energy portion can be "
                             "influenced by other factors adjustable within the attacker's simulation parameters."),
                ParameterSpec()
                    .name("Attack radius")
                    .value(BaseValueSpec().valueAddress(BASE_VALUE_OFFSET(attackerRadius)))
                    .type(FloatSpec().min(0).max(3.0f))
                    .colorDependence(ColorDependence::Vector)
                    .tooltip("The maximum distance over which an attacker cell can attack another cell."),
                ParameterSpec()
                    .name("Complex creature protection")
                    .value(BaseZoneValueSpec()
                               .valueAddress(ZONE_VALUE_OFFSET(attackerComplexCreatureProtection))
                               .enabledZoneValueAddress(ZONE_ENABLED_VALUE_OFFSET(attackerComplexCreatureProtection)))
                    .type(FloatSpec().min(0).max(20.0f).format("%.2f"))
                    .colorDependence(ColorDependence::Matrix)
                    .tooltip("The larger this parameter is, the less energy can be gained by attacking creatures with more complex genomes."),
                ParameterSpec()
                    .name("Destroy cells")
                    .value(BaseValueSpec().valueAddress(BASE_VALUE_OFFSET(attackerDestroyCells)))
                    .type(BoolSpec())
                    .tooltip("If activated, the attacker cell is able to destroy other cells. If deactivated, it only damages them."),
            }),
        ParameterGroupSpec()
            .name("Cell type: Constructor")
            .parameters({
                ParameterSpec()
                    .name("Connection distance")
                    .value(BaseValueSpec().valueAddress(BASE_VALUE_OFFSET(constructorConnectingCellDistance)))
                    .type(FloatSpec().min(0.1f).max(3.0f))
                    .colorDependence(ColorDependence::Vector)
                    .tooltip("The constructor can automatically connect constructed cells to other cells in the vicinity within this distance."),
                ParameterSpec()
                    .name("Completeness check")
                    .value(BaseValueSpec().valueAddress(BASE_VALUE_OFFSET(constructorCompletenessCheck)))
                    .type(BoolSpec())
                    .tooltip("If activated, a self-replication process can only start when all other non-self-replicating constructors in the cell network are "
                             "finished."),
            }),
        ParameterGroupSpec()
            .name("Cell type: Defender")
            .parameters({
                ParameterSpec()
                    .name("Anti-attacker strength")
                    .value(BaseValueSpec().valueAddress(BASE_VALUE_OFFSET(defenderAntiAttackerStrength)))
                    .type(FloatSpec().min(0.0f).max(5.0f))
                    .colorDependence(ColorDependence::Vector)
                    .tooltip("If an attacked cell is connected to defender cells or itself a defender cell the attack strength is reduced by this factor."),
                ParameterSpec()
                    .name("Anti-injector strength")
                    .value(BaseValueSpec().valueAddress(BASE_VALUE_OFFSET(defenderAntiInjectorStrength)))
                    .type(FloatSpec().min(0.0f).max(5.0f))
                    .colorDependence(ColorDependence::Vector)
                    .tooltip("If a constructor cell is attacked by an injector and connected to defender cells, the injection duration is increased by this "
                             "factor."),
            }),
        ParameterGroupSpec()
            .name("Cell type: Injector")
            .parameters({
                ParameterSpec()
                    .name("Injection radius")
                    .value(BaseValueSpec().valueAddress(BASE_VALUE_OFFSET(injectorInjectionRadius)))
                    .type(FloatSpec().min(0.1f).max(4.0f))
                    .colorDependence(ColorDependence::Vector)
                    .tooltip("The maximum distance over which an injector cell can infect another cell."),
                ParameterSpec()
                    .name("Injection time")
                    .value(BaseValueSpec().valueAddress(BASE_VALUE_OFFSET(injectorInjectionTime)))
                    .type(IntSpec().min(0).max(100000).logarithmic(true))
                    .colorDependence(ColorDependence::Matrix)
                    .tooltip("The number of activations an injector cell requires to infect another cell. One activation usually takes 6 time steps. The row "
                             "number determines the color of the injector cell, while the column number corresponds to the color of the infected cell."),
            }),
        ParameterGroupSpec()
            .name("Cell type: Muscle")
            .parameters({
                ParameterSpec()
                    .name("Energy cost")
                    .value(BaseValueSpec().valueAddress(BASE_VALUE_OFFSET(muscleEnergyCost)))
                    .type(FloatSpec().min(0).max(5.0f).format("%.5f").logarithmic(true))
                    .colorDependence(ColorDependence::Vector)
                    .tooltip("Amount of energy lost by a muscle action of a cell in form of emitted energy particles."),
                ParameterSpec()
                    .name("Movement acceleration")
                    .value(BaseValueSpec().valueAddress(BASE_VALUE_OFFSET(muscleMovementAcceleration)))
                    .type(FloatSpec().min(0).max(10.0f).logarithmic(true))
                    .colorDependence(ColorDependence::Vector)
                    .tooltip("The maximum value by which a muscle cell can modify its velocity during activation. This parameter applies only to muscle cells "
                             "which are in movement mode."),
                ParameterSpec()
                    .name("Crawling acceleration")
                    .value(BaseValueSpec().valueAddress(BASE_VALUE_OFFSET(muscleCrawlingAcceleration)))
                    .type(FloatSpec().min(0).max(10.0f).logarithmic(true))
                    .colorDependence(ColorDependence::Vector)
                    .tooltip("Amount of energy lost by a muscle action of a cell in form of emitted energy particles."),
                ParameterSpec()
                    .name("Bending acceleration")
                    .value(BaseValueSpec().valueAddress(BASE_VALUE_OFFSET(muscleBendingAcceleration)))
                    .type(FloatSpec().min(0).max(10.0f).logarithmic(true))
                    .colorDependence(ColorDependence::Vector)
                    .tooltip("The maximum value by which a muscle cell can modify its velocity during a bending action. This parameter applies "
                             "only to muscle cells which are in bending mode."),
            }),
        ParameterGroupSpec()
            .name("Cell type: Sensor")
            .parameters({
                ParameterSpec()
                    .name("Radius")
                    .value(BaseValueSpec().valueAddress(BASE_VALUE_OFFSET(sensorRadius)))
                    .type(FloatSpec().min(10.0f).max(800.0f))
                    .colorDependence(ColorDependence::Vector)
                    .tooltip("The maximum radius in which a sensor cell can detect mass concentrations."),
            }),
        ParameterGroupSpec()
            .name("Cell type: Transmitter")
            .parameters({
                ParameterSpec()
                    .name("Energy distribution radius")
                    .value(BaseValueSpec().valueAddress(BASE_VALUE_OFFSET(transmitterEnergyDistributionRadius)))
                    .type(FloatSpec().min(0).max(5.0f))
                    .colorDependence(ColorDependence::Vector)
                    .tooltip("The maximum distance over which a transmitter cell transfers its additional energy to nearby transmitter or constructor cells."),
                ParameterSpec()
                    .name("Energy distribution Value")
                    .value(BaseValueSpec().valueAddress(BASE_VALUE_OFFSET(transmitterEnergyDistributionValue)))
                    .type(FloatSpec().min(0).max(20.0f))
                    .colorDependence(ColorDependence::Vector)
                    .tooltip("The amount of energy which a transmitter cell can transfer to nearby transmitter or constructor cells or to connected cells."),
                ParameterSpec()
                    .name("Same creature energy distribution")
                    .value(BaseValueSpec().valueAddress(BASE_VALUE_OFFSET(transmitterEnergyDistributionSameCreature)))
                    .type(BoolSpec())
                    .tooltip("If activated, the transmitter cells can only transfer energy to nearby cells belonging to the same creature."),
            }),
        ParameterGroupSpec()
            .name("Cell type: Reconnector")
            .parameters({
                ParameterSpec()
                    .name("Radius")
                    .value(BaseValueSpec().valueAddress(BASE_VALUE_OFFSET(reconnectorRadius)))
                    .type(FloatSpec().min(0).max(3.0f))
                    .colorDependence(ColorDependence::Vector)
                    .tooltip("The maximum radius in which a reconnector cell can establish or destroy connections to other cells."),
            }),
        ParameterGroupSpec()
            .name("Cell type: Detonator")
            .parameters({
                ParameterSpec()
                    .name("Blast radius")
                    .value(BaseValueSpec().valueAddress(BASE_VALUE_OFFSET(detonatorRadius)))
                    .type(FloatSpec().min(0).max(10.0f))
                    .colorDependence(ColorDependence::Vector)
                    .tooltip("The radius of the detonation."),
                ParameterSpec()
                    .name("Chain explosion probability")
                    .value(BaseValueSpec().valueAddress(BASE_VALUE_OFFSET(detonatorChainExplosionProbability)))
                    .type(FloatSpec().min(0).max(1.0f))
                    .colorDependence(ColorDependence::Vector)
                    .tooltip("The probability that the explosion of one detonator will trigger the explosion of other detonators within the blast radius."),
            }),
        ParameterGroupSpec()
            .name("Advanced energy absorption control")
            .expertToggleAddress(EXPERT_VALUE_OFFSET(advancedAbsorptionControl))
            .parameters({
                ParameterSpec()
                    .name("Low genome complexity penalty")
                    .value(BaseZoneValueSpec()
                               .valueAddress(ZONE_VALUE_OFFSET(radiationAbsorptionLowGenomeComplexityPenalty))
                               .enabledZoneValueAddress(ZONE_ENABLED_VALUE_OFFSET(radiationAbsorptionLowGenomeComplexityPenalty)))
                    .type(FloatSpec().min(0).max(1.0f).format("%.2f"))
                    .colorDependence(ColorDependence::Vector)
                    .tooltip("When this parameter is increased, cells with fewer genome complexity will absorb less energy from an incoming energy particle."),
                ParameterSpec()
                    .name("Low connection penalty")
                    .value(BaseValueSpec().valueAddress(BASE_VALUE_OFFSET(radiationAbsorptionLowConnectionPenalty)))
                    .type(FloatSpec().min(0).max(5.0f).format("%.1f"))
                    .colorDependence(ColorDependence::Vector)
                    .tooltip("When this parameter is increased, cells with fewer cell connections will absorb less energy from an incoming energy particle."),
                ParameterSpec()
                    .name("High velocity penalty")
                    .value(BaseValueSpec().valueAddress(BASE_VALUE_OFFSET(radiationAbsorptionHighVelocityPenalty)))
                    .type(FloatSpec().min(0).max(30.0f).logarithmic(true).format("%.2f"))
                    .colorDependence(ColorDependence::Vector)
                    .tooltip("When this parameter is increased, fast moving cells will absorb less energy from an incoming energy particle."),
                ParameterSpec()
                    .name("Low velocity penalty")
                    .value(BaseZoneValueSpec()
                               .valueAddress(ZONE_VALUE_OFFSET(radiationAbsorptionLowVelocityPenalty))
                               .enabledZoneValueAddress(ZONE_ENABLED_VALUE_OFFSET(radiationAbsorptionLowVelocityPenalty)))
                    .type(FloatSpec().min(0).max(1.0f).format("%.2f"))
                    .colorDependence(ColorDependence::Vector)
                    .tooltip("When this parameter is increased, slowly moving cells will absorb less energy from an incoming energy particle."),
            }),
        ParameterGroupSpec()
            .name("Advanced attacker control")
            .expertToggleAddress(EXPERT_VALUE_OFFSET(advancedAttackerControl))
            .parameters({
                ParameterSpec()
                    .name("Same mutant protection")
                    .value(BaseValueSpec().valueAddress(BASE_VALUE_OFFSET(attackerSameMutantProtection)))
                    .type(FloatSpec().min(0).max(1.0f).format("%.2f"))
                    .colorDependence(ColorDependence::Matrix)
                    .tooltip("The larger this parameter is, the less energy can be gained by attacking creatures with the same mutation id."),
                ParameterSpec()
                    .name("New complex mutant protection")
                    .value(BaseZoneValueSpec()
                               .valueAddress(ZONE_VALUE_OFFSET(attackerNewComplexMutantProtection))
                               .enabledZoneValueAddress(ZONE_ENABLED_VALUE_OFFSET(attackerNewComplexMutantProtection)))
                    .type(FloatSpec().min(0).max(1.0f))
                    .colorDependence(ColorDependence::Matrix)
                    .tooltip("A high value protects new mutants with equal or greater genome complexity from being attacked."),
                ParameterSpec()
                    .name("Sensor detection factor")
                    .value(BaseValueSpec().valueAddress(BASE_VALUE_OFFSET(attackerSensorDetectionFactor)))
                    .type(FloatSpec().min(0).max(1.0f))
                    .colorDependence(ColorDependence::Vector)
                    .tooltip("This parameter controls whether the target must be previously detected with sensors in order to be attacked. The larger this "
                             "value is, the less energy can be gained during the attack if the target has not already been detected. For this purpose, the "
                             "attacker cell searches for connected (or connected-connected) sensor cells to see which cell networks they have detected last "
                             "time and "
                             "compares them with the attacked target."),
                ParameterSpec()
                    .name("Geometry deviation protection")
                    .value(BaseZoneValueSpec()
                               .valueAddress(ZONE_VALUE_OFFSET(attackerGeometryDeviationProtection))
                               .enabledZoneValueAddress(ZONE_ENABLED_VALUE_OFFSET(attackerGeometryDeviationProtection)))
                    .type(FloatSpec().min(0).max(5.0f))
                    .colorDependence(ColorDependence::Vector)
                    .tooltip("The larger this value is, the less energy a cell can gain from an attack if the local geometry of the attacked cell does not "
                             "match the attacking cell."),
                ParameterSpec()
                    .name("Connections mismatch protection")
                    .value(BaseZoneValueSpec()
                               .valueAddress(ZONE_VALUE_OFFSET(attackerConnectionsMismatchProtection))
                               .enabledZoneValueAddress(ZONE_ENABLED_VALUE_OFFSET(attackerConnectionsMismatchProtection)))
                    .type(FloatSpec().min(0).max(1.0f))
                    .colorDependence(ColorDependence::Vector)
                    .tooltip("The larger this parameter is, the more difficult it is to attack cells that contain more connections."),
            }),
        ParameterGroupSpec()
            .name("Cell age limiter")
            .expertToggleAddress(EXPERT_VALUE_OFFSET(cellAgeLimiter))
            .parameters({
                ParameterSpec()
                    .name("Maximum inactive cell age")
                    .value(BaseZoneValueSpec()
                               .valueAddress(ZONE_VALUE_OFFSET(maxAgeForInactiveCells))
                               .enabledBaseValueAddress(BASE_VALUE_OFFSET(maxAgeForInactiveCellsEnabled))
                               .enabledZoneValueAddress(ZONE_ENABLED_VALUE_OFFSET(maxAgeForInactiveCellsEnabled)))
                    .type(FloatSpec().min(1.0f).max(1e7f).format("%.0f").logarithmic(true).infinity(true))
                    .colorDependence(ColorDependence::Vector)
                    .tooltip("Here, you can set the maximum age for a cell whose function or those of its neighbors have not been triggered. Cells which "
                             "are in state 'Under construction' are not affected by this option."),
                ParameterSpec()
                    .name("Maximum free cell age")
                    .value(BaseValueSpec().valueAddress(BASE_VALUE_OFFSET(freeCellMaxAge)).enabledValueAddress(BASE_VALUE_OFFSET(freeCellMaxAgeEnabled)))
                    .type(IntSpec().min(1).max(1e7).logarithmic(true).infinity(true))
                    .colorDependence(ColorDependence::Vector)
                    .tooltip("The maximal age of free cells (= cells that arise from energy particles) can be set here."),
                ParameterSpec()
                    .name("Reset age after construction")
                    .value(BaseValueSpec().valueAddress(BASE_VALUE_OFFSET(resetCellAgeAfterActivation)))
                    .type(BoolSpec())
                    .tooltip("If this option is activated, the age of the cells is reset to 0 after the construction of their cell network is completed, "
                             "i.e. when the state of the cells changes from 'Under construction' to 'Ready'. This option is particularly useful if a low "
                             "'Maximum inactive cell age' is set, as cell networks that are under construction are inactive and could die immediately after "
                             "completion if their construction takes a long time."),
                ParameterSpec()
                    .name("Maximum age balancing")
                    .value(BaseValueSpec()
                               .valueAddress(BASE_VALUE_OFFSET(maxCellAgeBalancerInterval))
                               .enabledValueAddress(BASE_VALUE_OFFSET(maxCellAgeBalancerEnabled)))
                    .type(IntSpec().min(1e3).max(1e6).logarithmic(true))
                    .tooltip("Adjusts the maximum age at regular intervals. It increases the maximum age for the cell color where the fewest "
                             "replicators exist. Conversely, the maximum age is decreased for the cell color with the most replicators."),
            }),
        ParameterGroupSpec()
            .name("Cell color transition rules")
            .expertToggleAddress(EXPERT_VALUE_OFFSET(cellColorTransitionRules))
            .parameters({
                ParameterSpec()
                    .name("Target color and duration")
                    .value(BaseZoneValueSpec()
                               .valueAddress(ZONE_VALUE_OFFSET(colorTransitionRules))
                               .enabledZoneValueAddress(ZONE_ENABLED_VALUE_OFFSET(colorTransitionRules)))
                    .type(ColorTransitionSpec())
                    .tooltip("Rules can be defined that describe how the colors of cells will change over time. For this purpose, a subsequent "
                             "color can be defined for each cell color. In addition, durations must be specified that define how many time steps the "
                             "corresponding color are kept."),
            }),
        ParameterGroupSpec()
            .name("Cell glow")
            .expertToggleAddress(EXPERT_VALUE_OFFSET(cellGlow))
            .parameters({
                ParameterSpec()
                    .name("Coloring")
                    .value(BaseValueSpec().valueAddress(BASE_VALUE_OFFSET(cellGlowColoring)))
                    .type(AlternativeSpec().alternatives(
                        {{"Energy", {}},
                         {"Standard cell colors", {}},
                         {"Mutants", {}},
                         {"Mutants and cell functions", {}},
                         {"Cell states", {}},
                         {"Genome complexities", {}},
                         {"Single cell function", {}},
                         {"All cell functions", {}}}))
                    .tooltip(coloringTooltip),
                ParameterSpec()
                    .name("Radius")
                    .value(BaseValueSpec().valueAddress(BASE_VALUE_OFFSET(cellGlowRadius)))
                    .type(FloatSpec().min(1.0f).max(8.0f))
                    .tooltip("The radius of the glow. Please note that a large radius affects the performance."),
                ParameterSpec()
                    .name("Strength")
                    .value(BaseValueSpec().valueAddress(BASE_VALUE_OFFSET(cellGlowStrength)))
                    .type(FloatSpec().min(0.0f).max(1.0f))
                    .colorDependence(ColorDependence::Vector)
                    .tooltip("The strength of the glow."),
            }),
        ParameterGroupSpec()
            .name("Customize deletion mutations")
            .expertToggleAddress(EXPERT_VALUE_OFFSET(customizeDeletionMutations))
            .parameters({
                ParameterSpec()
                    .name("Minimum size")
                    .value(BaseValueSpec().valueAddress(BASE_VALUE_OFFSET(cellCopyMutationDeletionMinSize)))
                    .type(IntSpec().min(0).max(1000).logarithmic(true))
                    .tooltip("The minimum size of genomes (on the basis of the coded cells) is determined here that can result from delete mutations. The "
                             "default is 0."),
            }),
        ParameterGroupSpec()
            .name("Customize neuron mutations")
            .expertToggleAddress(EXPERT_VALUE_OFFSET(customizeNeuronMutations))
            .parameters({
                ParameterSpec()
                    .name("Affected weights")
                    .value(BaseValueSpec().valueAddress(BASE_VALUE_OFFSET(cellCopyMutationNeuronDataWeight)))
                    .type(FloatSpec().min(0.0f).max(1.0f).format("%.3f"))
                    .tooltip("The proportion of weights in the neuronal network of a cell that are changed within a neuron mutation. The default is 0.2."),
                ParameterSpec()
                    .name("Affected biases")
                    .value(BaseValueSpec().valueAddress(BASE_VALUE_OFFSET(cellCopyMutationNeuronDataBias)))
                    .type(FloatSpec().min(0.0f).max(1.0f).format("%.3f"))
                    .tooltip("The proportion of biases in the neuronal network of a cell that are changed within a neuron mutation. The default is 0.2."),
                ParameterSpec()
                    .name("Affected activation functions")
                    .value(BaseValueSpec().valueAddress(BASE_VALUE_OFFSET(cellCopyMutationNeuronDataActivationFunction)))
                    .type(FloatSpec().min(0.0f).max(1.0f).format("%.3f"))
                    .tooltip("The proportion of activation functions in the neuronal network of a cell that are changed within a neuron mutation. The default "
                             "is 0.05."),
                ParameterSpec()
                    .name("Reinforcement factor")
                    .value(BaseValueSpec().valueAddress(BASE_VALUE_OFFSET(cellCopyMutationNeuronDataReinforcement)))
                    .type(FloatSpec().min(1.0f).max(1.2f).format("%.3f"))
                    .tooltip(
                        "If a weight or bias of the neural network is adjusted by a mutation, it can either be reinforced, weakened or shifted by an offset. "
                        "The factor that is used for reinforcement is defined here. The default is 1.05."),
                ParameterSpec()
                    .name("Damping factor")
                    .value(BaseValueSpec().valueAddress(BASE_VALUE_OFFSET(cellCopyMutationNeuronDataDamping)))
                    .type(FloatSpec().min(1.0f).max(1.2f).format("%.3f"))
                    .tooltip(
                        "If a weight or bias of the neural network is adjusted by a mutation, it can either be reinforced, weakened or shifted by an offset. "
                        "The factor that is used for weakening is defined here. The default is 1.05."),
                ParameterSpec()
                    .name("Offset")
                    .value(BaseValueSpec().valueAddress(BASE_VALUE_OFFSET(cellCopyMutationNeuronDataOffset)))
                    .type(FloatSpec().min(0.0f).max(0.2f).format("%.3f"))
                    .tooltip(
                        "If a weight or bias of the neural network is adjusted by a mutation, it can either be reinforced, weakened or shifted by an offset. "
                        "The value that is used for the offset is defined here. The default is 0.05."),
            }),
        ParameterGroupSpec()
            .name("External energy control")
            .expertToggleAddress(EXPERT_VALUE_OFFSET(externalEnergyControl))
            .parameters({
                ParameterSpec()
                    .name("External energy amount")
                    .value(BaseValueSpec().valueAddress(BASE_VALUE_OFFSET(externalEnergy)))
                    .type(FloatSpec().min(0.0f).max(100000000.0f).format("%.0f").logarithmic(true).infinity(true))
                    .tooltip(
                        "This parameter can be used to set the amount of energy of an external energy pool. This type of energy can then be "
                        "transferred to all constructor cells at a certain rate (see inflow settings).\n\nWarning: Too much external energy can result in a "
                        "massive production of cells and slow down or even crash the simulation."),
                ParameterSpec()
                    .name("Inflow")
                    .value(BaseValueSpec().valueAddress(BASE_VALUE_OFFSET(externalEnergyInflowFactor)))
                    .type(FloatSpec().min(0.0f).max(1.0f).format("%.5f").logarithmic(true))
                    .colorDependence(ColorDependence::Vector)
                    .tooltip(
                        "Here one can specify the fraction of energy transferred to constructor cells.\n\nFor example, a value of 0.05 means that "
                        "each time a constructor cell tries to build a new cell, 5% of the required energy is transferred for free from the external energy "
                        "source."),
                ParameterSpec()
                    .name("Conditional inflow")
                    .value(BaseValueSpec().valueAddress(BASE_VALUE_OFFSET(externalEnergyConditionalInflowFactor)))
                    .type(FloatSpec().min(0.00f).max(1.0f).format("%.5f").logarithmic(true))
                    .colorDependence(ColorDependence::Vector)
                    .tooltip("Here one can specify the fraction of energy transferred to constructor cells if they can provide the remaining energy for the "
                             "construction process.\n\nFor example, a value of 0.6 means that a constructor cell receives 60% of the energy required to "
                             "build the new cell for free from the external energy source. However, it must provide 40% of the energy required by itself. "
                             "Otherwise, no energy will be transferred."),
                ParameterSpec()
                    .name("Inflow only for non-replicators")
                    .value(BaseValueSpec().valueAddress(BASE_VALUE_OFFSET(externalEnergyInflowOnlyForNonSelfReplicators)))
                    .type(BoolSpec())
                    .tooltip("If activated, external energy can only be transferred to constructor cells that are not self-replicators. "
                             "This option can be used to foster the evolution of additional body parts."),
                ParameterSpec()
                    .name("Backflow")
                    .value(BaseValueSpec().valueAddress(BASE_VALUE_OFFSET(externalEnergyBackflowFactor)))
                    .type(FloatSpec().min(0.0f).max(1.0f))
                    .colorDependence(ColorDependence::Vector)
                    .tooltip("The proportion of energy that flows back from the simulation to the external energy pool. Each time a cell loses energy "
                             "or dies a fraction of its energy will be taken. The remaining "
                             "fraction of the energy stays in the simulation and will be used to create a new energy particle."),
                ParameterSpec()
                    .name("Backflow limit")
                    .value(BaseValueSpec().valueAddress(BASE_VALUE_OFFSET(externalEnergyBackflowLimit)))
                    .type(FloatSpec().min(0.0f).max(1e8f).format("%.0f").logarithmic(true).infinity(true))
                    .tooltip("Energy from the simulation can only flow back into the external energy pool as long as the amount of external energy is "
                             "below this value."),
            }),
        ParameterGroupSpec()
            .name("Genome complexity measurement")
            .expertToggleAddress(EXPERT_VALUE_OFFSET(genomeComplexityMeasurement))
            .parameters({
                ParameterSpec()
                    .name("Size factor")
                    .value(BaseValueSpec().valueAddress(BASE_VALUE_OFFSET(genomeComplexitySizeFactor)))
                    .type(FloatSpec().min(0.0f).max(1.0f).format("%.2f"))
                    .colorDependence(ColorDependence::Vector)
                    .tooltip("This parameter controls how the number of encoded cells in the genome influences the calculation of its complexity."),
                ParameterSpec()
                    .name("Ramification factor")
                    .value(BaseValueSpec().valueAddress(BASE_VALUE_OFFSET(genomeComplexityRamificationFactor)))
                    .type(FloatSpec().min(0.0f).max(20.0f).format("%.2f"))
                    .colorDependence(ColorDependence::Vector)
                    .tooltip("With this parameter, the number of ramifications of the cell structure to the genome is taken into account for the "
                             "calculation of the genome complexity. For instance, genomes that contain many sub-genomes or many construction branches will "
                             "then have a high complexity value."),
                ParameterSpec()
                    .name("Depth level")
                    .value(BaseValueSpec().valueAddress(BASE_VALUE_OFFSET(genomeComplexityDepthLevel)))
                    .type(IntSpec().min(1).max(20).infinity(true))
                    .colorDependence(ColorDependence::Vector)
                    .tooltip("This allows to specify up to which level of the sub-genomes the complexity calculation should be carried out. For example, a "
                             "value of 2 means that the sub- and sub-sub-genomes are taken into account in addition to the main genome."),
            }),
    });
}

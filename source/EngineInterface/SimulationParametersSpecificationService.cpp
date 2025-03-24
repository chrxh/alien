#include "SimulationParametersSpecificationService.h"

#include <Fonts/IconsFontAwesome5.h>

#include "CellTypeStrings.h"
#include "LocationHelper.h"
#include "SimulationParametersEditService.h"

#define BASE_VALUE_OFFSET(X) offsetof(SimulationParameters, X)
#define ZONE_VALUE_OFFSET(X) offsetof(SimulationParametersZoneValues, X)

ParametersSpec SimulationParametersSpecificationService::createParametersSpec() const
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

    auto coloringTooltip = std::string(
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
        "selected in the next parameter.\n\n" ICON_FA_CHEVRON_RIGHT " Every cell function: The cells are colored according to their cell function.");

    return ParametersSpec().groups({
        ParameterGroupSpec().name("General").parameters({
            ParameterSpec().name("Project name").valueAddress(BASE_VALUE_OFFSET(projectName)).type(Char64Spec()),
        }),
        ParameterGroupSpec()
            .name("Visualization")
            .parameters({
                ParameterSpec().name("Background color").visibleInZone(true).valueAddress(ZONE_VALUE_OFFSET(backgroundColor)).type(ColorSpec()),
                ParameterSpec()
                    .name("Primary cell coloring")
                    .valueAddress(BASE_VALUE_OFFSET(primaryCellColoring))
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
                              .valueAddress(BASE_VALUE_OFFSET(highlightedCellType))
                              .tooltip("The specific cell function type to be highlighted can be selected here.")
                              .type(AlternativeSpec().alternatives(cellTypeStrings))}},
                        {"Every cell function", {}},
                    }))
                    .tooltip(coloringTooltip),
                ParameterSpec()
                    .name("Cell radius")
                    .valueAddress(BASE_VALUE_OFFSET(cellRadius))
                    .type(FloatSpec().min(0.0f).max(0.5f))
                    .tooltip("Specifies the radius of the drawn cells in unit length."),
                ParameterSpec()
                    .name("Zoom level for neural activity")
                    .valueAddress(BASE_VALUE_OFFSET(zoomLevelForNeuronVisualization))
                    .type(FloatSpec().min(0.0f).max(32.f).infinity(true))
                    .tooltip("The zoom level from which the neuronal activities become visible."),
                ParameterSpec()
                    .name("Attack visualization")
                    .valueAddress(BASE_VALUE_OFFSET(attackVisualization))
                    .type(BoolSpec())
                    .tooltip("If activated, successful attacks of attacker cells are visualized."),
                ParameterSpec()
                    .name("Muscle movement visualization")
                    .valueAddress(BASE_VALUE_OFFSET(muscleMovementVisualization))
                    .type(BoolSpec())
                    .tooltip("If activated, the direction in which muscle cells are moving are visualized."),
                ParameterSpec()
                    .name("Borderless rendering")
                    .valueAddress(BASE_VALUE_OFFSET(borderlessRendering))
                    .type(BoolSpec())
                    .tooltip("If activated, the simulation is rendered periodically in the view port."),
                ParameterSpec()
                    .name("Grid lines")
                    .valueAddress(BASE_VALUE_OFFSET(gridLines))
                    .type(BoolSpec())
                    .tooltip("This option draws a suitable grid in the background depending on the zoom level."),
                ParameterSpec()
                    .name("Mark reference domain")
                    .valueAddress(BASE_VALUE_OFFSET(markReferenceDomain))
                    .type(BoolSpec())
                    .tooltip("This option draws a suitable grid in the background depending on the zoom level."),
                ParameterSpec()
                    .name("Show radiation sources")
                    .valueAddress(BASE_VALUE_OFFSET(showRadiationSources))
                    .type(BoolSpec())
                    .tooltip("This option draws red crosses in the center of radiation sources."),
            }),
        ParameterGroupSpec()
            .name("Numerics")
            .parameters({
                ParameterSpec()
                    .name("Time step size")
                    .valueAddress(BASE_VALUE_OFFSET(timestepSize))
                    .type(FloatSpec().min(0.0f).max(1.0f))
                    .tooltip("The time duration calculated in a single simulation step. Smaller values increase the accuracy of the simulation while larger "
                             "values can lead to numerical instabilities."),
            }),
        ParameterGroupSpec()
            .name("Physics: Motion")
            .parameters({
                ParameterSpec()
                    .name("Motion type")
                    .valueAddress(BASE_VALUE_OFFSET(motionType))
                    .type(AlternativeSpec().alternatives(
                        {{"Fluid solver",
                          {
                              ParameterSpec()
                                  .name("Smoothing length")
                                  .valueAddress(BASE_VALUE_OFFSET(smoothingLength))
                                  .tooltip("The smoothing length determines the region of influence of the neighboring particles for the calculation of "
                                           "density, pressure and viscosity. Values that are too small lead to numerical instabilities, while values that "
                                           "are too large cause the particles to drift apart.")
                                  .type(FloatSpec().min(0).max(3.0f)),
                              ParameterSpec()
                                  .name("Pressure")
                                  .valueAddress(BASE_VALUE_OFFSET(pressureStrength))
                                  .tooltip("This parameter allows to control the strength of the pressure.")
                                  .type(FloatSpec().min(0).max(0.3f)),
                              ParameterSpec()
                                  .name("Viscosity")
                                  .valueAddress(BASE_VALUE_OFFSET(viscosityStrength))
                                  .tooltip("This parameter be used to control the strength of the viscosity. Larger values lead to a smoother movement.")
                                  .type(FloatSpec().min(0).max(0.3f)),
                          }},
                         {"Collision-based solver",
                          {
                              ParameterSpec()
                                  .name("Repulsion strength")
                                  .valueAddress(BASE_VALUE_OFFSET(repulsionStrength))
                                  .tooltip("The strength of the repulsive forces, between two cells that are not connected.")
                                  .type(FloatSpec().min(0).max(0.3f)),
                              ParameterSpec()
                                  .name("Maximum collision distance")
                                  .valueAddress(BASE_VALUE_OFFSET(maxCollisionDistance))
                                  .tooltip("Maximum distance up to which a collision of two cells is possible.")
                                  .type(FloatSpec().min(0).max(3.0f)),
                          }}}))
                    .tooltip(std::string(
                        "The algorithm for the particle motions is defined here. If 'Fluid dynamics' is selected, an SPH fluid solver is used for the "
                        "calculation of the forces. The particles then behave like (compressible) liquids or gases. The other option 'Collision-based' "
                        "calculates the forces based on particle collisions and should be preferred for mechanical simulation with solids.")),
                ParameterSpec()
                    .name("Friction")
                    .visibleInZone(true)
                    .valueAddress(ZONE_VALUE_OFFSET(friction))
                    .type(FloatSpec().min(0).max(1.0f).logarithmic(true).format("%.4f"))
                    .tooltip("This specifies the fraction of the velocity that is slowed down per time step."),
                ParameterSpec()
                    .name("Rigidity")
                    .visibleInZone(true)
                    .valueAddress(ZONE_VALUE_OFFSET(rigidity))
                    .type(FloatSpec().min(0).max(3.0f))
                    .tooltip("Controls the rigidity of connected cells. A higher value will cause connected cells to move more uniformly as a rigid body."),
            }),
        ParameterGroupSpec()
            .name("Physics: Thresholds")
            .parameters({
                ParameterSpec()
                    .name("Maximum velocity")
                    .valueAddress(BASE_VALUE_OFFSET(maxVelocity))
                    .type(FloatSpec().min(0.0f).max(6.0f))
                    .tooltip("Maximum velocity that a cell can reach."),
                ParameterSpec()
                    .name("Maximum force")
                    .visibleInZone(true)
                    .valueAddress(ZONE_VALUE_OFFSET(cellMaxForce))
                    .colorDependence(true)
                    .type(FloatSpec().min(0.0f).max(3.0f))
                    .tooltip("Maximum force that can be applied to a cell without causing it to disintegrate."),
                ParameterSpec()
                    .name("Minimum distance")
                    .valueAddress(BASE_VALUE_OFFSET(minCellDistance))
                    .type(FloatSpec().min(0.0f).max(1.0f))
                    .tooltip("Minimum distance between two cells."),
            }),
        ParameterGroupSpec()
            .name("Physics: Binding")
            .parameters({
                ParameterSpec()
                    .name("Maximum distance")
                    .valueAddress(BASE_VALUE_OFFSET(maxBindingDistance))
                    .colorDependence(true)
                    .type(FloatSpec().min(0.0f).max(5.0f))
                    .tooltip("Maximum distance up to which a connection of two cells is possible."),
                ParameterSpec()
                    .name("Fusion velocity")
                    .visibleInZone(true)
                    .valueAddress(ZONE_VALUE_OFFSET(cellFusionVelocity))
                    .type(FloatSpec().min(0.0f).max(2.0f))
                    .tooltip("Maximum force that can be applied to a cell without causing it to disintegrate."),
                ParameterSpec()
                    .name("Maximum energy")
                    .visibleInZone(true)
                    .valueAddress(ZONE_VALUE_OFFSET(cellMaxBindingEnergy))
                    .type(FloatSpec().min(50.0f).max(10000000.0f).logarithmic(true).infinity(true).format("%.0f"))
                    .tooltip("Maximum energy of a cell at which it can contain bonds to adjacent cells. If the energy of a cell exceeds this "
                             "value, all bonds will be destroyed."),
            }),
        ParameterGroupSpec()
            .name("Physics: Radiation")
            .parameters({
                ParameterSpec()
                    .name("Relative strength")
                    .type(FloatSpec()
                              .min(0.0f)
                              .max(1.0f)
                              .pinnedAddress(BASE_VALUE_OFFSET(baseStrengthRatioPinned))
                              .valueGetter(radiationStrengthGetter)
                              .valueSetter(radiationStrengthSetter))
                    .tooltip("Cells can emit energy particles over time. A portion of this energy can be released directly near the cell, while the rest is "
                             "utilized by one of the available radiation sources. This parameter determines the fraction of energy assigned to the emitted "
                             "energy particle in the vicinity of the cell. Values between 0 and 1 are permitted."),
                ParameterSpec()
                    .name("Absorption factor")
                    .visibleInZone(true)
                    .valueAddress(ZONE_VALUE_OFFSET(radiationAbsorption))
                    .colorDependence(true)
                    .type(FloatSpec().min(0.0f).max(1.0f).logarithmic(true).format("%.4f"))
                    .tooltip("The fraction of energy that a cell can absorb from an incoming energy particle can be specified here."),
                ParameterSpec()
                    .name("Radiation type I: Strength")
                    .visibleInZone(true)
                    .valueAddress(ZONE_VALUE_OFFSET(radiationType1_strength))
                    .colorDependence(true)
                    .type(FloatSpec().min(0.0f).max(0.01f).logarithmic(true).format("%.6f"))
                    .tooltip("Indicates how energetic the emitted particles of aged cells are."),
                ParameterSpec()
                    .name("Radiation type I: Minimum age")
                    .valueAddress(BASE_VALUE_OFFSET(radiationType1_minimumAge))
                    .colorDependence(true)
                    .type(IntSpec().min(0).max(10000000).logarithmic(true).infinity(true))
                    .tooltip("The minimum age of a cell can be defined here, from which it emits energy particles."),
                ParameterSpec()
                    .name("Radiation type II: Strength")
                    .valueAddress(BASE_VALUE_OFFSET(radiationType2_strength))
                    .colorDependence(true)
                    .type(FloatSpec().min(0.0f).max(0.01f).logarithmic(true).format("%.6f"))
                    .tooltip("Indicates how energetic the emitted particles of high energy cells are."),
                ParameterSpec()
                    .name("Radiation type II: Energy threshold")
                    .valueAddress(BASE_VALUE_OFFSET(radiationType2_energyThreshold))
                    .colorDependence(true)
                    .type(FloatSpec().min(0.0f).max(100000.0f).logarithmic(true).infinity(true).format("%.1f"))
                    .tooltip("The minimum energy of a cell can be defined here, from which it emits energy particles."),
                ParameterSpec()
                    .name("Minimum split energy")
                    .valueAddress(BASE_VALUE_OFFSET(particleSplitEnergy))
                    .colorDependence(true)
                    .type(FloatSpec().min(1.0f).max(10000.0f).logarithmic(true).infinity(true).format("%.0f"))
                    .tooltip("The minimum energy of an energy particle after which it can split into two particles, whereby it receives a small momentum. The "
                             "splitting does not occur immediately, but only after a certain time."),
                ParameterSpec()
                    .name("Energy to cell transformation")
                    .valueAddress(BASE_VALUE_OFFSET(particleTransformationAllowed))
                    .type(BoolSpec())
                    .tooltip("If activated, an energy particle will transform into a cell if the energy of the particle exceeds the normal energy value."),
            }),
        ParameterGroupSpec()
            .name("Cell life cycle")
            .parameters({
                ParameterSpec()
                    .name("Maximum age")
                    .valueAddress(BASE_VALUE_OFFSET(maxCellAge))
                    .colorDependence(true)
                    .type(IntSpec().min(1).max(1e7).logarithmic(true).infinity(true))
                    .tooltip("Defines the maximum age of a cell. If a cell exceeds this age it will be transformed to an energy particle."),
                ParameterSpec()
                    .name("Minimum energy")
                    .visibleInZone(true)
                    .valueAddress(ZONE_VALUE_OFFSET(minCellEnergy))
                    .colorDependence(true)
                    .type(FloatSpec().min(10.0f).max(200.0f))
                    .tooltip("Minimum energy a cell needs to exist."),
                ParameterSpec()
                    .name("Normal energy")
                    .valueAddress(BASE_VALUE_OFFSET(normalCellEnergy))
                    .colorDependence(true)
                    .type(FloatSpec().min(10.0f).max(200.0f))
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
                    .visibleInZone(true)
                    .valueAddress(ZONE_VALUE_OFFSET(cellDeathProbability))
                    .colorDependence(true)
                    .type(FloatSpec().min(1e-6f).max(1e-1f).format("%.6f").logarithmic(true))
                    .tooltip("The probability per time step with which a cell will disintegrate (i.e. transform into an energy particle) when it is in the "
                             "state 'Dying'. This can occur when one of the following conditions is satisfied:\n\n" ICON_FA_CHEVRON_RIGHT
                             " The cell has too low energy.\n\n" ICON_FA_CHEVRON_RIGHT " The cell has exceeded its maximum age."),
                ParameterSpec()
                    .name("Cell death consequences")
                    .valueAddress(BASE_VALUE_OFFSET(cellDeathConsequences))
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
                    .visibleInZone(true)
                    .valueAddress(ZONE_VALUE_OFFSET(copyMutationNeuronData))
                    .colorDependence(true)
                    .type(FloatSpec().min(0.0f).max(1.0f).format("%.7f").logarithmic(true))
                    .tooltip("This type of mutation can change the weights, biases and activation functions of neural networks of each neuron cell encoded in the "
                         "genome."),
                ParameterSpec()
                    .name("Cell properties")
                    .visibleInZone(true)
                    .valueAddress(ZONE_VALUE_OFFSET(copyMutationCellProperties))
                    .colorDependence(true)
                    .type(FloatSpec().min(0.0f).max(1.0f).format("%.7f").logarithmic(true))
                    .tooltip("This type of mutation changes a random property (e.g. (input) execution order number, required energy, block output and "
                         "function-specific properties such as minimum density for sensors, neural net weights etc.). The spatial structure, color, cell "
                         "function type and self-replication capabilities are not changed. This mutation is applied to each encoded cell in the genome."),
                ParameterSpec()
                    .name("Geometry")
                    .visibleInZone(true)
                    .valueAddress(ZONE_VALUE_OFFSET(copyMutationGeometry))
                    .colorDependence(true)
                    .type(FloatSpec().min(0.0f).max(1.0f).format("%.7f").logarithmic(true))
                    .tooltip("This type of mutation changes the geometry type, connection distance, stiffness and single construction flag. The probability of "
                         "a change is given by the specified value times the number of coded cells in the genome."),
                ParameterSpec()
                    .name("Custom geometry")
                    .visibleInZone(true)
                    .valueAddress(ZONE_VALUE_OFFSET(copyMutationCustomGeometry))
                    .colorDependence(true)
                    .type(FloatSpec().min(0.0f).max(1.0f).format("%.7f").logarithmic(true))
                    .tooltip("This type of mutation only changes angles and required connections of custom geometries. The probability of a change is given by "
                         "the specified value times the number of coded cells in the genome."),
                ParameterSpec()
                    .name("Cell function type")
                    .visibleInZone(true)
                    .valueAddress(ZONE_VALUE_OFFSET(copyMutationCellType))
                    .colorDependence(true)
                    .type(FloatSpec().min(0.0f).max(1.0f).format("%.7f").logarithmic(true))
                    .tooltip("This type of mutation changes the type of cell function. The changed cell function will have random properties. The probability "
                         "of a change is given by the specified value times the number of coded cells in the genome. If the flag 'Preserve "
                         "self-replication' is disabled it can also alter self-replication capabilities by changing a constructor to "
                         "something else or vice versa."),
                ParameterSpec()
                    .name("Insertion")
                    .visibleInZone(true)
                    .valueAddress(ZONE_VALUE_OFFSET(copyMutationInsertion))
                    .colorDependence(true)
                    .type(FloatSpec().min(0.0f).max(1.0f).format("%.7f").logarithmic(true))
                    .tooltip("This type of mutation inserts a new cell description to the genome at a random position. The probability of a change is given by "
                         "the specified value times the number of coded cells in the genome."),
                ParameterSpec()
                    .name("Deletion")
                    .visibleInZone(true)
                    .valueAddress(ZONE_VALUE_OFFSET(copyMutationDeletion))
                    .colorDependence(true)
                    .type(FloatSpec().min(0.0f).max(1.0f).format("%.7f").logarithmic(true))
                    .tooltip("This type of mutation deletes a cell description from the genome at a random position. The probability of a change is given by "
                         "the specified value times the number of coded cells in the genome."),
                ParameterSpec()
                    .name("Translation")
                    .visibleInZone(true)
                    .valueAddress(ZONE_VALUE_OFFSET(copyMutationTranslation))
                    .colorDependence(true)
                    .type(FloatSpec().min(0.0f).max(1.0f).format("%.7f").logarithmic(true))
                    .tooltip("This type of mutation moves a block of cell descriptions from the genome at a random position to a new random position."),
                ParameterSpec()
                    .name("Duplication")
                    .visibleInZone(true)
                    .valueAddress(ZONE_VALUE_OFFSET(copyMutationDuplication))
                    .colorDependence(true)
                    .type(FloatSpec().min(0.0f).max(1.0f).format("%.7f").logarithmic(true))
                    .tooltip("This type of mutation copies a block of cell descriptions from the genome at a random position to a new random position."),
                ParameterSpec()
                    .name("Individual cell color")
                    .visibleInZone(true)
                    .valueAddress(ZONE_VALUE_OFFSET(copyMutationCellColor))
                    .colorDependence(true)
                    .type(FloatSpec().min(0.0f).max(1.0f).format("%.7f").logarithmic(true))
                    .tooltip("This type of mutation alters the color of a single cell descriptions in a genome by using the specified color transitions. The "
                         "probability of a change is given by the specified value times the number of coded cells in the genome."),
                ParameterSpec()
                    .name("Sub-genome color")
                    .visibleInZone(true)
                    .valueAddress(ZONE_VALUE_OFFSET(copyMutationSubgenomeColor))
                    .colorDependence(true)
                    .type(FloatSpec().min(0.0f).max(1.0f).format("%.7f").logarithmic(true))
                    .tooltip("This type of mutation alters the color of all cell descriptions in a sub-genome by using the specified color transitions."),
                ParameterSpec()
                    .name("Genome color")
                    .visibleInZone(true)
                    .valueAddress(ZONE_VALUE_OFFSET(copyMutationGenomeColor))
                    .colorDependence(true)
                    .type(FloatSpec().min(0.0f).max(1.0f).format("%.7f").logarithmic(true))
                    .tooltip("This type of mutation alters the color of all cell descriptions in a genome by using the specified color transitions."),
            }),
    });
}

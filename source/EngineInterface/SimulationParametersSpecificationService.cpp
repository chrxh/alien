#include "SimulationParametersSpecificationService.h"

#include <Fonts/IconsFontAwesome5.h>

#include "CellTypeStrings.h"
#include "LocationHelper.h"
#include "SimulationParametersEditService.h"

#define BASE_VALUE_OFFSET(X) offsetof(SimulationParameters, X)
#define ZONE_VALUE_OFFSET(X) offsetof(SimulationParametersZoneValues, X)
#define EXPERT_VALUE_OFFSET(X) offsetof(ExpertSettingsToggles, X)

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
                ParameterSpec().name("Background color").visibleInZone(true).valueAddress(ZONE_VALUE_OFFSET(backgroundColor)).type(ColorPickerSpec()),
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
                    .colorDependence(ColorDependence::Vector)
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
                    .colorDependence(ColorDependence::Vector)
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
                    .colorDependence(ColorDependence::Vector)
                    .type(FloatSpec().min(0.0f).max(1.0f).logarithmic(true).format("%.4f"))
                    .tooltip("The fraction of energy that a cell can absorb from an incoming energy particle can be specified here."),
                ParameterSpec()
                    .name("Radiation type I: Strength")
                    .visibleInZone(true)
                    .valueAddress(ZONE_VALUE_OFFSET(radiationType1_strength))
                    .colorDependence(ColorDependence::Vector)
                    .type(FloatSpec().min(0.0f).max(0.01f).logarithmic(true).format("%.6f"))
                    .tooltip("Indicates how energetic the emitted particles of aged cells are."),
                ParameterSpec()
                    .name("Radiation type I: Minimum age")
                    .valueAddress(BASE_VALUE_OFFSET(radiationType1_minimumAge))
                    .colorDependence(ColorDependence::Vector)
                    .type(IntSpec().min(0).max(10000000).logarithmic(true).infinity(true))
                    .tooltip("The minimum age of a cell can be defined here, from which it emits energy particles."),
                ParameterSpec()
                    .name("Radiation type II: Strength")
                    .valueAddress(BASE_VALUE_OFFSET(radiationType2_strength))
                    .colorDependence(ColorDependence::Vector)
                    .type(FloatSpec().min(0.0f).max(0.01f).logarithmic(true).format("%.6f"))
                    .tooltip("Indicates how energetic the emitted particles of high energy cells are."),
                ParameterSpec()
                    .name("Radiation type II: Energy threshold")
                    .valueAddress(BASE_VALUE_OFFSET(radiationType2_energyThreshold))
                    .colorDependence(ColorDependence::Vector)
                    .type(FloatSpec().min(0.0f).max(100000.0f).logarithmic(true).infinity(true).format("%.1f"))
                    .tooltip("The minimum energy of a cell can be defined here, from which it emits energy particles."),
                ParameterSpec()
                    .name("Minimum split energy")
                    .valueAddress(BASE_VALUE_OFFSET(particleSplitEnergy))
                    .colorDependence(ColorDependence::Vector)
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
                    .colorDependence(ColorDependence::Vector)
                    .type(IntSpec().min(1).max(1e7).logarithmic(true).infinity(true))
                    .tooltip("Defines the maximum age of a cell. If a cell exceeds this age it will be transformed to an energy particle."),
                ParameterSpec()
                    .name("Minimum energy")
                    .visibleInZone(true)
                    .valueAddress(ZONE_VALUE_OFFSET(minCellEnergy))
                    .colorDependence(ColorDependence::Vector)
                    .type(FloatSpec().min(10.0f).max(200.0f))
                    .tooltip("Minimum energy a cell needs to exist."),
                ParameterSpec()
                    .name("Normal energy")
                    .valueAddress(BASE_VALUE_OFFSET(normalCellEnergy))
                    .colorDependence(ColorDependence::Vector)
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
                    .colorDependence(ColorDependence::Vector)
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
                    .colorDependence(ColorDependence::Vector)
                    .type(FloatSpec().min(0.0f).max(1.0f).format("%.7f").logarithmic(true))
                    .tooltip(
                        "This type of mutation can change the weights, biases and activation functions of neural networks of each neuron cell encoded in the "
                        "genome."),
                ParameterSpec()
                    .name("Cell properties")
                    .visibleInZone(true)
                    .valueAddress(ZONE_VALUE_OFFSET(copyMutationCellProperties))
                    .colorDependence(ColorDependence::Vector)
                    .type(FloatSpec().min(0.0f).max(1.0f).format("%.7f").logarithmic(true))
                    .tooltip("This type of mutation changes a random property (e.g. (input) execution order number, required energy, block output and "
                             "function-specific properties such as minimum density for sensors, neural net weights etc.). The spatial structure, color, cell "
                             "function type and self-replication capabilities are not changed. This mutation is applied to each encoded cell in the genome."),
                ParameterSpec()
                    .name("Geometry")
                    .visibleInZone(true)
                    .valueAddress(ZONE_VALUE_OFFSET(copyMutationGeometry))
                    .colorDependence(ColorDependence::Vector)
                    .type(FloatSpec().min(0.0f).max(1.0f).format("%.7f").logarithmic(true))
                    .tooltip("This type of mutation changes the geometry type, connection distance, stiffness and single construction flag. The probability of "
                             "a change is given by the specified value times the number of coded cells in the genome."),
                ParameterSpec()
                    .name("Custom geometry")
                    .visibleInZone(true)
                    .valueAddress(ZONE_VALUE_OFFSET(copyMutationCustomGeometry))
                    .colorDependence(ColorDependence::Vector)
                    .type(FloatSpec().min(0.0f).max(1.0f).format("%.7f").logarithmic(true))
                    .tooltip("This type of mutation only changes angles and required connections of custom geometries. The probability of a change is given by "
                             "the specified value times the number of coded cells in the genome."),
                ParameterSpec()
                    .name("Cell function type")
                    .visibleInZone(true)
                    .valueAddress(ZONE_VALUE_OFFSET(copyMutationCellType))
                    .colorDependence(ColorDependence::Vector)
                    .type(FloatSpec().min(0.0f).max(1.0f).format("%.7f").logarithmic(true))
                    .tooltip("This type of mutation changes the type of cell function. The changed cell function will have random properties. The probability "
                             "of a change is given by the specified value times the number of coded cells in the genome. If the flag 'Preserve "
                             "self-replication' is disabled it can also alter self-replication capabilities by changing a constructor to "
                             "something else or vice versa."),
                ParameterSpec()
                    .name("Insertion")
                    .visibleInZone(true)
                    .valueAddress(ZONE_VALUE_OFFSET(copyMutationInsertion))
                    .colorDependence(ColorDependence::Vector)
                    .type(FloatSpec().min(0.0f).max(1.0f).format("%.7f").logarithmic(true))
                    .tooltip("This type of mutation inserts a new cell description to the genome at a random position. The probability of a change is given by "
                             "the specified value times the number of coded cells in the genome."),
                ParameterSpec()
                    .name("Deletion")
                    .visibleInZone(true)
                    .valueAddress(ZONE_VALUE_OFFSET(copyMutationDeletion))
                    .colorDependence(ColorDependence::Vector)
                    .type(FloatSpec().min(0.0f).max(1.0f).format("%.7f").logarithmic(true))
                    .tooltip("This type of mutation deletes a cell description from the genome at a random position. The probability of a change is given by "
                             "the specified value times the number of coded cells in the genome."),
                ParameterSpec()
                    .name("Translation")
                    .visibleInZone(true)
                    .valueAddress(ZONE_VALUE_OFFSET(copyMutationTranslation))
                    .colorDependence(ColorDependence::Vector)
                    .type(FloatSpec().min(0.0f).max(1.0f).format("%.7f").logarithmic(true))
                    .tooltip("This type of mutation moves a block of cell descriptions from the genome at a random position to a new random position."),
                ParameterSpec()
                    .name("Duplication")
                    .visibleInZone(true)
                    .valueAddress(ZONE_VALUE_OFFSET(copyMutationDuplication))
                    .colorDependence(ColorDependence::Vector)
                    .type(FloatSpec().min(0.0f).max(1.0f).format("%.7f").logarithmic(true))
                    .tooltip("This type of mutation copies a block of cell descriptions from the genome at a random position to a new random position."),
                ParameterSpec()
                    .name("Individual cell color")
                    .visibleInZone(true)
                    .valueAddress(ZONE_VALUE_OFFSET(copyMutationCellColor))
                    .colorDependence(ColorDependence::Vector)
                    .type(FloatSpec().min(0.0f).max(1.0f).format("%.7f").logarithmic(true))
                    .tooltip("This type of mutation alters the color of a single cell descriptions in a genome by using the specified color transitions. The "
                             "probability of a change is given by the specified value times the number of coded cells in the genome."),
                ParameterSpec()
                    .name("Sub-genome color")
                    .visibleInZone(true)
                    .valueAddress(ZONE_VALUE_OFFSET(copyMutationSubgenomeColor))
                    .colorDependence(ColorDependence::Vector)
                    .type(FloatSpec().min(0.0f).max(1.0f).format("%.7f").logarithmic(true))
                    .tooltip("This type of mutation alters the color of all cell descriptions in a sub-genome by using the specified color transitions."),
                ParameterSpec()
                    .name("Genome color")
                    .visibleInZone(true)
                    .valueAddress(ZONE_VALUE_OFFSET(copyMutationGenomeColor))
                    .colorDependence(ColorDependence::Vector)
                    .type(FloatSpec().min(0.0f).max(1.0f).format("%.7f").logarithmic(true))
                    .tooltip("This type of mutation alters the color of all cell descriptions in a genome by using the specified color transitions."),
                ParameterSpec()
                    .name("Color transitions")
                    .valueAddress(BASE_VALUE_OFFSET(copyMutationColorTransitions))
                    .colorDependence(ColorDependence::Matrix)
                    .type(BoolSpec())
                    .tooltip("The color transitions are used for color mutations. The row index indicates the source color and the column index the target "
                             "color."),
                ParameterSpec()
                    .name("Prevent genome depth increase")
                    .valueAddress(BASE_VALUE_OFFSET(copyMutationPreventDepthIncrease))
                    .type(BoolSpec())
                    .tooltip("A genome has a tree-like structure because it can contain sub-genomes. If this flag is activated, the mutations will "
                             "not increase the depth of the genome structure."),
                ParameterSpec()
                    .name("Mutate self-replication")
                    .valueAddress(BASE_VALUE_OFFSET(copyMutationSelfReplication))
                    .type(BoolSpec())
                    .tooltip("If activated, a mutation can also alter self-replication capabilities in the genome by changing a constructor cell to "
                             "something else or vice versa."),
            }),
        ParameterGroupSpec()
            .name("Cell type: Attacker")
            .parameters({
                ParameterSpec()
                    .name("Energy cost")
                    .visibleInZone(true)
                    .valueAddress(ZONE_VALUE_OFFSET(attackerEnergyCost))
                    .colorDependence(ColorDependence::Vector)
                    .type(FloatSpec().min(0).max(1.0f).logarithmic(true).format("%.5f"))
                    .tooltip("Amount of energy lost by an attempted attack of a cell in form of emitted energy particles."),
                ParameterSpec()
                    .name("Food chain color matrix")
                    .visibleInZone(true)
                    .valueAddress(ZONE_VALUE_OFFSET(attackerFoodChainColorMatrix))
                    .colorDependence(ColorDependence::Matrix)
                    .type(FloatSpec().min(0.0f).max(1.0f).format("%.2f"))
                    .tooltip(
                        "This matrix can be used to determine how well one cell can attack another cell. The color of the attacking cell correspond to the "
                        "row number and the color of the attacked cell to the column number. A value of 0 means that the attacked cell cannot be digested, "
                        "i.e. no energy can be obtained. A value of 1 means that the maximum energy can be obtained in the digestion process.\n\nExample: If "
                        "a zero is entered in row 2 (red) and column 3 (green), it means that red cells cannot eat green cells."),
                ParameterSpec()
                    .name("Attack strength")
                    .valueAddress(BASE_VALUE_OFFSET(attackerStrength))
                    .colorDependence(ColorDependence::Vector)
                    .type(FloatSpec().min(0).max(0.5f).logarithmic(true))
                    .tooltip("Indicates the portion of energy through which a successfully attacked cell is weakened. However, this energy portion can be "
                             "influenced by other factors adjustable within the attacker's simulation parameters."),
                ParameterSpec()
                    .name("Attack radius")
                    .valueAddress(BASE_VALUE_OFFSET(attackerRadius))
                    .colorDependence(ColorDependence::Vector)
                    .type(FloatSpec().min(0).max(3.0f))
                    .tooltip("The maximum distance over which an attacker cell can attack another cell."),
                ParameterSpec()
                    .name("Complex creature protection")
                    .visibleInZone(true)
                    .valueAddress(ZONE_VALUE_OFFSET(attackerCreatureProtection))
                    .colorDependence(ColorDependence::Matrix)
                    .type(FloatSpec().min(0).max(20.0f).format("%.2f"))
                    .tooltip("The larger this parameter is, the less energy can be gained by attacking creatures with more complex genomes."),
                ParameterSpec()
                    .name("Destroy cells")
                    .valueAddress(BASE_VALUE_OFFSET(attackerDestroyCells))
                    .type(BoolSpec())
                    .tooltip("If activated, the attacker cell is able to destroy other cells. If deactivated, it only damages them."),
            }),
        ParameterGroupSpec()
            .name("Cell type: Constructor")
            .parameters({
                ParameterSpec()
                    .name("Connection distance")
                    .valueAddress(BASE_VALUE_OFFSET(constructorConnectingCellDistance))
                    .colorDependence(ColorDependence::Vector)
                    .type(FloatSpec().min(0.1f).max(3.0f))
                    .tooltip("The constructor can automatically connect constructed cells to other cells in the vicinity within this distance."),
                ParameterSpec()
                    .name("Completeness check")
                    .valueAddress(BASE_VALUE_OFFSET(constructorCompletenessCheck))
                    .type(BoolSpec())
                    .tooltip("If activated, a self-replication process can only start when all other non-self-replicating constructors in the cell network are "
                         "finished."),
            }),
        ParameterGroupSpec()
            .name("Cell type: Defender")
            .parameters({
                ParameterSpec()
                    .name("Anti-attacker strength")
                    .valueAddress(BASE_VALUE_OFFSET(defenderAntiAttackerStrength))
                    .colorDependence(ColorDependence::Vector)
                    .type(FloatSpec().min(0.0f).max(5.0f))
                    .tooltip("If an attacked cell is connected to defender cells or itself a defender cell the attack strength is reduced by this factor."),
                ParameterSpec()
                    .name("Anti-injector strength")
                    .valueAddress(BASE_VALUE_OFFSET(defenderAntiInjectorStrength))
                    .colorDependence(ColorDependence::Vector)
                    .type(FloatSpec().min(0.0f).max(5.0f))
                    .tooltip("If a constructor cell is attacked by an injector and connected to defender cells, the injection duration is increased by this "
                         "factor."),
            }),
        ParameterGroupSpec()
            .name("Cell type: Injector")
            .parameters({
                ParameterSpec()
                    .name("Injection radius")
                    .valueAddress(BASE_VALUE_OFFSET(injectorInjectionRadius))
                    .colorDependence(ColorDependence::Vector)
                    .type(FloatSpec().min(0.1f).max(4.0f))
                    .tooltip("The maximum distance over which an injector cell can infect another cell."),
                ParameterSpec()
                    .name("Injection time")
                    .valueAddress(BASE_VALUE_OFFSET(injectorInjectionTime))
                    .colorDependence(ColorDependence::Matrix)
                    .type(IntSpec().min(0).max(100000).logarithmic(true))
                    .tooltip("The number of activations an injector cell requires to infect another cell. One activation usually takes 6 time steps. The row "
                         "number determines the color of the injector cell, while the column number corresponds to the color of the infected cell."),
            }),
        ParameterGroupSpec()
            .name("Cell type: Muscle")
            .parameters({
                ParameterSpec()
                    .name("Energy cost")
                    .valueAddress(BASE_VALUE_OFFSET(muscleEnergyCost))
                    .colorDependence(ColorDependence::Vector)
                    .type(FloatSpec().min(0).max(5.0f).format("%.5f").logarithmic(true))
                    .tooltip("Amount of energy lost by a muscle action of a cell in form of emitted energy particles."),
                ParameterSpec()
                    .name("Movement acceleration")
                    .valueAddress(BASE_VALUE_OFFSET(muscleMovementAcceleration))
                    .colorDependence(ColorDependence::Vector)
                    .type(FloatSpec().min(0).max(10.0f).logarithmic(true))
                    .tooltip("The maximum value by which a muscle cell can modify its velocity during activation. This parameter applies only to muscle cells "
                         "which are in movement mode."),
                ParameterSpec()
                    .name("Crawling acceleration")
                    .valueAddress(BASE_VALUE_OFFSET(muscleCrawlingAcceleration))
                    .colorDependence(ColorDependence::Vector)
                    .type(FloatSpec().min(0).max(10.0f).logarithmic(true))
                    .tooltip("Amount of energy lost by a muscle action of a cell in form of emitted energy particles."),
                ParameterSpec()
                    .name("Bending acceleration")
                    .valueAddress(BASE_VALUE_OFFSET(muscleBendingAcceleration))
                    .colorDependence(ColorDependence::Vector)
                    .type(FloatSpec().min(0).max(10.0f).logarithmic(true))
                    .tooltip("The maximum value by which a muscle cell can modify its velocity during a bending action. This parameter applies "
                         "only to muscle cells which are in bending mode."),
            }),
        ParameterGroupSpec()
            .name("Cell type: Sensor")
            .parameters({
                ParameterSpec()
                    .name("Radius")
                    .valueAddress(BASE_VALUE_OFFSET(sensorRadius))
                    .colorDependence(ColorDependence::Vector)
                    .type(FloatSpec().min(10.0f).max(800.0f))
                    .tooltip("The maximum radius in which a sensor cell can detect mass concentrations."),
            }),
        ParameterGroupSpec()
            .name("Cell type: Transmitter")
            .parameters({
                ParameterSpec()
                    .name("Energy distribution radius")
                    .valueAddress(BASE_VALUE_OFFSET(transmitterEnergyDistributionRadius))
                    .colorDependence(ColorDependence::Vector)
                    .type(FloatSpec().min(0).max(5.0f))
                    .tooltip("The maximum distance over which a transmitter cell transfers its additional energy to nearby transmitter or constructor cells."),
                ParameterSpec()
                    .name("Energy distribution Value")
                    .valueAddress(BASE_VALUE_OFFSET(transmitterEnergyDistributionValue))
                    .colorDependence(ColorDependence::Vector)
                    .type(FloatSpec().min(0).max(20.0f))
                    .tooltip("The amount of energy which a transmitter cell can transfer to nearby transmitter or constructor cells or to connected cells."),
                ParameterSpec()
                    .name("Same creature energy distribution")
                    .valueAddress(BASE_VALUE_OFFSET(transmitterEnergyDistributionSameCreature))
                    .colorDependence(ColorDependence::Vector)
                    .type(BoolSpec())
                    .tooltip("If activated, the transmitter cells can only transfer energy to nearby cells belonging to the same creature."),
            }),
        ParameterGroupSpec()
            .name("Cell type: Reconnector")
            .parameters({
                ParameterSpec()
                    .name("Radius")
                    .valueAddress(BASE_VALUE_OFFSET(reconnectorRadius))
                    .colorDependence(ColorDependence::Vector)
                    .type(FloatSpec().min(0).max(3.0f))
                    .tooltip("The maximum radius in which a reconnector cell can establish or destroy connections to other cells."),
            }),
        ParameterGroupSpec()
            .name("Cell type: Detonator")
            .parameters({
                ParameterSpec()
                    .name("Blast radius")
                    .valueAddress(BASE_VALUE_OFFSET(detonatorRadius))
                    .colorDependence(ColorDependence::Vector)
                    .type(FloatSpec().min(0).max(10.0f))
                    .tooltip("The radius of the detonation."),
                ParameterSpec()
                    .name("Chain explosion probability")
                    .valueAddress(BASE_VALUE_OFFSET(detonatorChainExplosionProbability))
                    .colorDependence(ColorDependence::Vector)
                    .type(FloatSpec().min(0).max(1.0f))
                    .tooltip("The probability that the explosion of one detonator will trigger the explosion of other detonators within the blast radius."),
            }),
        ParameterGroupSpec()
            .name("Expert settings: Advanced energy absorption control")
            .expertSettingAddress(EXPERT_VALUE_OFFSET(advancedAbsorptionControl))
            .parameters({
                ParameterSpec()
                    .name("Low genome complexity penalty")
                    .visibleInZone(true)
                    .valueAddress(ZONE_VALUE_OFFSET(radiationAbsorptionLowGenomeComplexityPenalty))
                    .colorDependence(ColorDependence::Vector)
                    .type(FloatSpec().min(0).max(1.0f).format("%.2f"))
                    .tooltip("When this parameter is increased, cells with fewer genome complexity will absorb less energy from an incoming energy particle."),
                ParameterSpec()
                    .name("Low connection penalty")
                    .valueAddress(BASE_VALUE_OFFSET(radiationAbsorptionLowConnectionPenalty))
                    .colorDependence(ColorDependence::Vector)
                    .type(FloatSpec().min(0).max(5.0f).format("%.1f"))
                    .tooltip("When this parameter is increased, cells with fewer cell connections will absorb less energy from an incoming energy particle."),
                ParameterSpec()
                    .name("High velocity penalty")
                    .valueAddress(BASE_VALUE_OFFSET(radiationAbsorptionHighVelocityPenalty))
                    .colorDependence(ColorDependence::Vector)
                    .type(FloatSpec().min(0).max(30.0f).logarithmic(true).format("%.2f"))
                    .tooltip("When this parameter is increased, fast moving cells will absorb less energy from an incoming energy particle."),
                ParameterSpec()
                    .name("Low velocity penalty")
                    .visibleInZone(true)
                    .valueAddress(ZONE_VALUE_OFFSET(radiationAbsorptionLowVelocityPenalty))
                    .colorDependence(ColorDependence::Vector)
                    .type(FloatSpec().min(0).max(1.0f).format("%.2f"))
                    .tooltip("When this parameter is increased, slowly moving cells will absorb less energy from an incoming energy particle."),
            }),
        ParameterGroupSpec()
            .name("Expert settings: Advanced attacker control")
            .expertSettingAddress(EXPERT_VALUE_OFFSET(advancedAttackerControl))
            .parameters({
                ParameterSpec()
                    .name("Same mutant protection")
                    .valueAddress(BASE_VALUE_OFFSET(attackerSameMutantProtection))
                    .colorDependence(ColorDependence::Matrix)
                    .type(FloatSpec().min(0).max(1.0f).format("%.2f"))
                    .tooltip("The larger this parameter is, the less energy can be gained by attacking creatures with the same mutation id."),
                ParameterSpec()
                    .name("New complex mutant protection")
                    .visibleInZone(true)
                    .valueAddress(ZONE_VALUE_OFFSET(attackerNewComplexMutantProtection))
                    .colorDependence(ColorDependence::Matrix)
                    .type(FloatSpec().min(0).max(1.0f))
                    .tooltip("A high value protects new mutants with equal or greater genome complexity from being attacked."),
                ParameterSpec()
                    .name("Sensor detection factor")
                    .valueAddress(BASE_VALUE_OFFSET(attackerSensorDetectionFactor))
                    .colorDependence(ColorDependence::Vector)
                    .type(FloatSpec().min(0).max(1.0f))
                    .tooltip("This parameter controls whether the target must be previously detected with sensors in order to be attacked. The larger this "
                         "value is, the less energy can be gained during the attack if the target has not already been detected. For this purpose, the "
                         "attacker cell searches for connected (or connected-connected) sensor cells to see which cell networks they have detected last time and "
                         "compares them with the attacked target."),
                ParameterSpec()
                    .name("Geometry deviation protection")
                    .visibleInZone(true)
                    .valueAddress(ZONE_VALUE_OFFSET(attackerGeometryDeviationProtection))
                    .colorDependence(ColorDependence::Vector)
                    .type(FloatSpec().min(0).max(5.0f))
                    .tooltip("The larger this value is, the less energy a cell can gain from an attack if the local geometry of the attacked cell does not "
                             "match the attacking cell."),
                ParameterSpec()
                    .name("Connections mismatch protection")
                    .visibleInZone(true)
                    .valueAddress(ZONE_VALUE_OFFSET(attackerConnectionsMismatchProtection))
                    .colorDependence(ColorDependence::Vector)
                    .type(FloatSpec().min(0).max(1.0f))
                    .tooltip("The larger this parameter is, the more difficult it is to attack cells that contain more connections."),
            }),
        ParameterGroupSpec()
            .name("Expert settings: Cell age limiter")
            .expertSettingAddress(EXPERT_VALUE_OFFSET(cellAgeLimiter))
            .parameters({
                ParameterSpec()
                    .name("Maximum inactive cell age")
                    .visibleInZone(true)
                    .valueAddress(ZONE_VALUE_OFFSET(maxAgeForInactiveCells))
                    .enabledValueBaseAddress(BASE_VALUE_OFFSET(maxAgeForInactiveCellsActivated))
                    .colorDependence(ColorDependence::Vector)
                    .type(FloatSpec().min(1.0f).max(10000000.0f).format("%.0f").logarithmic(true).infinity(true))
                    .tooltip("Here, you can set the maximum age for a cell whose function or those of its neighbors have not been triggered. Cells which "
                         "are in state 'Under construction' are not affected by this option."),
            }),
        ParameterGroupSpec()
            .name("Expert settings: Cell color transition rules")
            .expertSettingAddress(EXPERT_VALUE_OFFSET(cellColorTransitionRules))
            .parameters({
                ParameterSpec()
                    .name("Target color and duration")
                    .visibleInZone(true)
                    .valueAddress(ZONE_VALUE_OFFSET(cellColorTransitionTargetColor))
                    .type(ColorTransitionSpec().transitionDurationAddress(ZONE_VALUE_OFFSET(cellColorTransitionDuration)))
                    .tooltip("Rules can be defined that describe how the colors of cells will change over time. For this purpose, a subsequent "
                             "color can be defined for each cell color. In addition, durations must be specified that define how many time steps the "
                             "corresponding color are kept."),
            }),
    });
}

bool& SimulationParametersSpecificationService::getExpertSettingsToggleRef(ParameterGroupSpec const& spec, SimulationParameters& parameters) const
{
    return *(reinterpret_cast<bool*>(reinterpret_cast<char*>(&parameters.expertSettingsToggles) + spec._expertSettingAddress.value()));
}

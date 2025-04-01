#include "SimulationParametersSpecificationService.h"

#include <Fonts/IconsFontAwesome5.h>
#include <boost/variant.hpp>

#include "CellTypeStrings.h"
#include "LocationHelper.h"
#include "SimulationParametersEditService.h"

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

bool* SimulationParametersSpecificationService::getBoolRef(BoolMemberSpec const& memberSpec, SimulationParameters& parameters, int locationIndex) const
{
    // Single value
    if (std::holds_alternative<BoolMember>(memberSpec)) {
        return &(parameters.**std::get<BoolMember>(memberSpec));
    } else if (std::holds_alternative<BoolZoneValuesMember>(memberSpec)) {
        switch (LocationHelper::getLocationType(locationIndex, parameters)) {
        case LocationType::Base:
            return &(parameters.baseValues.**std::get<BoolZoneValuesMember>(memberSpec));
        case LocationType::Zone: {
            auto index = LocationHelper::findLocationArrayIndex(parameters, locationIndex);
            return &(parameters.zone[index].values.**std::get<BoolZoneValuesMember>(memberSpec));
        }
        }
    }

    // Color matrix
    else if (std::holds_alternative<ColorMatrixBoolMember>(memberSpec)) {
        return reinterpret_cast<bool*>(parameters.**std::get<ColorMatrixBoolMember>(memberSpec));
    }

    return nullptr;
}

int* SimulationParametersSpecificationService::getIntRef(IntMemberSpec const& memberSpec, SimulationParameters& parameters, int locationIndex) const
{
    // Single value
    if (std::holds_alternative<IntMember>(memberSpec)) {
        return &(parameters.**std::get<IntMember>(memberSpec));
    }

    // Color vector
    else if (std::holds_alternative<ColorVectorIntMember>(memberSpec)) {
        return parameters.**std::get<ColorVectorIntMember>(memberSpec);
    }

    // Color matrix
    else if (std::holds_alternative<ColorMatrixIntMember>(memberSpec)) {
        return reinterpret_cast<int*>(parameters.**std::get<ColorMatrixIntMember>(memberSpec));
    }

    return nullptr;
}

float* SimulationParametersSpecificationService::getFloatRef(FloatMemberSpec const& memberSpec, SimulationParameters& parameters, int locationIndex) const
{
    // Single value
    if (std::holds_alternative<FloatMember>(memberSpec)) {
        return &(parameters.**std::get<FloatMember>(memberSpec));
    } else if (std::holds_alternative<FloatZoneValuesMember>(memberSpec)) {
        if (locationIndex == 0) {
            return &(parameters.baseValues.**std::get<FloatZoneValuesMember>(memberSpec));
        } else {
            auto index = LocationHelper::findLocationArrayIndex(parameters, locationIndex);
            return &(parameters.zone[index].values.**std::get<FloatZoneValuesMember>(memberSpec));
        }
    }

    // Color vector
    else if (std::holds_alternative<ColorVectorFloatMember>(memberSpec)) {
        return parameters.**std::get<ColorVectorFloatMember>(memberSpec);
    } else if (std::holds_alternative<ColorVectorFloatZoneValuesMember>(memberSpec)) {
        switch (LocationHelper::getLocationType(locationIndex, parameters)) {
        case LocationType::Base:
            return parameters.baseValues.**std::get<ColorVectorFloatZoneValuesMember>(memberSpec);
        case LocationType::Zone: {
            auto index = LocationHelper::findLocationArrayIndex(parameters, locationIndex);
            return parameters.zone[index].values.**std::get<ColorVectorFloatZoneValuesMember>(memberSpec);
        }
        }
    }

    // Color matrix
    else if (std::holds_alternative<ColorMatrixFloatMember>(memberSpec)) {
        return reinterpret_cast<float*>(parameters.**std::get<ColorMatrixFloatMember>(memberSpec));
    } else if (std::holds_alternative<ColorMatrixFloatZoneValuesMember>(memberSpec)) {
        switch (LocationHelper::getLocationType(locationIndex, parameters)) {
        case LocationType::Base:
            return reinterpret_cast<float*>(parameters.baseValues.**std::get<ColorMatrixFloatZoneValuesMember>(memberSpec));
        case LocationType::Zone: {
            auto index = LocationHelper::findLocationArrayIndex(parameters, locationIndex);
            return reinterpret_cast<float*>(parameters.zone[index].values.**std::get<ColorMatrixFloatZoneValuesMember>(memberSpec));
        }
        }
    }

    return nullptr;
}

char* SimulationParametersSpecificationService::getChar64Ref(Char64MemberSpec const& memberSpec, SimulationParameters& parameters, int locationIndex) const
{
    if (std::holds_alternative<Char64Member>(memberSpec)) {
        return parameters.**std::get<Char64Member>(memberSpec);
    }

    return nullptr;
}

int* SimulationParametersSpecificationService::getAlternativeRef(AlternativeMemberSpec const& memberSpec, SimulationParameters& parameters, int locationIndex)
    const
{
    // Single value
    if (std::holds_alternative<IntMember>(memberSpec)) {
        return &(parameters.**std::get<IntMember>(memberSpec));
    }

    return nullptr;
}

FloatColorRGB* SimulationParametersSpecificationService::getFloatColorRGBRef(
    ColorPickerMemberSpec const& memberSpec,
    SimulationParameters& parameters,
    int locationIndex) const
{
    if (std::holds_alternative<FloatColorRGBZoneMember>(memberSpec)) {
        switch (LocationHelper::getLocationType(locationIndex, parameters)) {
        case LocationType::Base:
            return &(parameters.baseValues.**std::get<FloatColorRGBZoneMember>(memberSpec));
        case LocationType::Zone: {
            auto index = LocationHelper::findLocationArrayIndex(parameters, locationIndex);
            return &(parameters.zone[index].values.**std::get<FloatColorRGBZoneMember>(memberSpec));
        }
        }
    }

    return nullptr;
}

ColorTransitionRules* SimulationParametersSpecificationService::getColorTransitionRulesRef(
    ColorTransitionRulesMemberSpec const& memberSpec,
    SimulationParameters& parameters,
    int locationIndex) const
{
    if (std::holds_alternative<ColorTransitionRulesZoneMember>(memberSpec)) {
        switch (LocationHelper::getLocationType(locationIndex, parameters)) {
        case LocationType::Base:
            return &(parameters.baseValues.**std::get<ColorTransitionRulesZoneMember>(memberSpec));
        case LocationType::Zone: {
            auto index = LocationHelper::findLocationArrayIndex(parameters, locationIndex);
            return &(parameters.zone[index].values.**std::get<ColorTransitionRulesZoneMember>(memberSpec));
        }
        }
    }

    return nullptr;
}

bool* SimulationParametersSpecificationService::getEnabledRef(EnabledSpec const& spec, SimulationParameters& parameters, int locationIndex) const
{
    auto locationType = LocationHelper::getLocationType(locationIndex, parameters);
    if (spec._base && locationType == LocationType::Base) {
        return &(parameters.**spec._base.get());
    }
    if (spec._zone && locationType == LocationType::Zone) {
        auto index = LocationHelper::findLocationArrayIndex(parameters, locationIndex);
        return &(parameters.zone[index].enabledValues.**spec._zone.get());
    }
    return nullptr;
}

//bool* SimulationParametersSpecificationService::getPinnedValueRef(ValueSpec const& spec, SimulationParameters& parameters, int locationIndex) const
//{
//    if (std::get<BaseValueSpec>(&spec)) {
//        auto baseValueSpec = std::get<BaseValueSpec>(spec);
//        if (baseValueSpec._pinnedAddress.has_value()) {
//            return reinterpret_cast<bool*>(reinterpret_cast<char*>(&parameters) + baseValueSpec._pinnedAddress.value());
//        }
//    }
//    return nullptr;
//}

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

    _parametersSpec = ParametersSpec().groups({
        ParameterGroupSpec().name("General").parameters({
            ParameterSpec()
                .name("Project name")
                .reference(Char64Spec().member(&SimulationParameters::projectName)),
        }),
        ParameterGroupSpec()
            .name("Visualization")
            .parameters({
                ParameterSpec().name("Background color").reference(ColorPickerSpec().member(&SimulationParametersZoneValues::backgroundColor)),
                ParameterSpec()
                    .name("Primary cell coloring")
                    .reference(AlternativeSpec()
                                   .member(&SimulationParameters::primaryCellColoring)
                        .alternatives({{"Energy", {}}})
                                   .alternatives({
                                       {"Energy", {}},
                                       {"Standard cell color", {}},
                                       {"Mutant", {}},
                                       {"Mutant and cell function", {}},
                                       {"Cell state", {}},
                                       {"Genome complexity", {}},
                                       {"Specific cell function",
                                        {ParameterSpec()
                                             .name("Highlighted cell function")
                                             .reference(AlternativeSpec()
                                                            .member(&SimulationParameters::highlightedCellType)
                                                            .alternatives(cellTypeStrings))
                                             .tooltip("The specific cell function type to be highlighted can be selected here.")}},
                                       {"Every cell function", {}},
                                   }))
                    .tooltip(coloringTooltip),
                ParameterSpec()
                    .name("Cell radius")
                    .reference(FloatSpec().member(&SimulationParameters::cellRadius).min(0.0f).max(0.5f))
                    .tooltip("Specifies the radius of the drawn cells in unit length."),
                ParameterSpec()
                    .name("Zoom level for neural activity")
                    .reference(FloatSpec().member(&SimulationParameters::zoomLevelForNeuronVisualization).min(0.0f).max(32.f).infinity(true))
                    .tooltip("The zoom level from which the neuronal activities become visible."),
                ParameterSpec()
                    .name("Attack visualization")
                    .reference(BoolSpec().member(&SimulationParameters::attackVisualization))
                    .tooltip("If activated, successful attacks of attacker cells are visualized."),
                ParameterSpec()
                    .name("Muscle movement visualization")
                    .reference(BoolSpec().member(&SimulationParameters::muscleMovementVisualization))
                    .tooltip("If activated, the direction in which muscle cells are moving are visualized."),
                ParameterSpec()
                    .name("Borderless rendering")
                    .reference(BoolSpec().member(&SimulationParameters::borderlessRendering))
                    .tooltip("If activated, the simulation is rendered periodically in the view port."),
                ParameterSpec()
                    .name("Grid lines")
                    .reference(BoolSpec().member(&SimulationParameters::gridLines))
                    .tooltip("This option draws a suitable grid in the background depending on the zoom level."),
                ParameterSpec()
                    .name("Mark reference domain")
                    .reference(BoolSpec().member(&SimulationParameters::markReferenceDomain))
                    .tooltip("This option draws a suitable grid in the background depending on the zoom level."),
                ParameterSpec()
                    .name("Show radiation sources")
                    .reference(BoolSpec().member(&SimulationParameters::showRadiationSources))
                    .tooltip("This option draws red crosses in the center of radiation sources."),
            }),
        ParameterGroupSpec()
            .name("Numerics")
            .parameters({
                ParameterSpec()
                    .name("Time step size")
                    .reference(FloatSpec().member(&SimulationParameters::timestepSize).min(0.05f).max(1.0f))
                    .tooltip("The time duration calculated in a single simulation step. Smaller values increase the accuracy of the simulation while larger "
                             "values can lead to numerical instabilities."),
            }),
        ParameterGroupSpec()
            .name("Physics: Motion")
            .parameters({
                ParameterSpec()
                    .name("Motion type")
                    .reference(
                        AlternativeSpec()
                            .member(&SimulationParameters::motionType)
                            .alternatives(
                                {{"Fluid solver",
                                  {
                                      ParameterSpec()
                                          .name("Smoothing length")
                                          .reference(FloatSpec().member(&SimulationParameters::smoothingLength).min(0).max(3.0f))
                                          .tooltip(
                                              "The smoothing length determines the region of influence of the neighboring particles for the calculation of "
                                              "density, pressure and viscosity. Values that are too small lead to numerical instabilities, while values that "
                                              "are too large cause the particles to drift apart."),
                                      ParameterSpec()
                                          .name("Pressure")
                                          .reference(FloatSpec().member(&SimulationParameters::pressureStrength).min(0).max(0.3f))
                                          .tooltip("This parameter allows to control the strength of the pressure."),
                                      ParameterSpec()
                                          .name("Viscosity")
                                          .reference(FloatSpec().member(&SimulationParameters::viscosityStrength).min(0).max(0.3f))
                                          .tooltip(
                                              "This parameter be used to control the strength of the viscosity. Larger values lead to a smoother movement."),
                                  }},
                                 {"Collision-based solver",
                                  {
                                      ParameterSpec()
                                          .name("Repulsion strength")
                                          .reference(FloatSpec().member(&SimulationParameters::repulsionStrength).min(0).max(0.3f))
                                          .tooltip("The strength of the repulsive forces, between two cells that are not connected."),
                                      ParameterSpec()
                                          .name("Maximum collision distance")
                                          .reference(FloatSpec().member(&SimulationParameters::maxCollisionDistance).min(0).max(3.0f))
                                          .tooltip("Maximum distance up to which a collision of two cells is possible."),
                                  }}}))
                    .tooltip(std::string(
                        "The algorithm for the particle motions is defined here. If 'Fluid dynamics' is selected, an SPH fluid solver is used for the "
                        "calculation of the forces. The particles then behave like (compressible) liquids or gases. The other option 'Collision-based' "
                        "calculates the forces based on particle collisions and should be preferred for mechanical simulation with solids.")),
                ParameterSpec()
                    .name("Friction")
                    .reference(FloatSpec().member(&SimulationParametersZoneValues::friction).min(0).max(1.0f).logarithmic(true).format("%.4f"))
                    .tooltip("This specifies the fraction of the velocity that is slowed down per time step."),
                ParameterSpec()
                    .name("Rigidity")
                    .reference(FloatSpec().member(&SimulationParametersZoneValues::rigidity).min(0).max(3.0f))
                    .tooltip("Controls the rigidity of connected cells. A higher value will cause connected cells to move more uniformly as a rigid body."),
            }),
        ParameterGroupSpec()
            .name("Physics: Thresholds")
            .parameters({
                ParameterSpec()
                    .name("Maximum velocity")
                    .reference(FloatSpec().member(&SimulationParameters::maxVelocity).min(0.0f).max(6.0f))
                    .tooltip("Maximum velocity that a cell can reach."),
                ParameterSpec()
                    .name("Maximum force")
                    .reference(FloatSpec().member(&SimulationParametersZoneValues::cellMaxForce).min(0.0f).max(3.0f))
                    .tooltip("Maximum force that can be applied to a cell without causing it to disintegrate."),
                ParameterSpec()
                    .name("Minimum distance")
                    .reference(FloatSpec().member(&SimulationParameters::minCellDistance).min(0.0f).max(1.0f))
                    .tooltip("Minimum distance between two cells."),
            }),
        ParameterGroupSpec()
            .name("Physics: Binding")
            .parameters({
                ParameterSpec()
                    .name("Maximum distance")
                    .reference(FloatSpec().member(&SimulationParameters::maxBindingDistance).min(0.0f).max(5.0f))
                    .tooltip("Maximum distance up to which a connection of two cells is possible."),
                ParameterSpec()
                    .name("Fusion velocity")
                    .reference(FloatSpec().member(&SimulationParametersZoneValues::cellFusionVelocity).min(0.0f).max(2.0f))
                    .tooltip("Maximum force that can be applied to a cell without causing it to disintegrate."),
                ParameterSpec()
                    .name("Maximum energy")
                    .reference(
                        FloatSpec()
                            .member(&SimulationParametersZoneValues::cellMaxBindingEnergy)
                            .min(50.0f)
                            .max(10000000.0f)
                            .logarithmic(true)
                            .infinity(true)
                            .format("%.0f"))
                    .tooltip("Maximum energy of a cell at which it can contain bonds to adjacent cells. If the energy of a cell exceeds this "
                             "value, all bonds will be destroyed."),
            }),
        ParameterGroupSpec()
            .name("Physics: Radiation")
            .parameters({
                ParameterSpec()
                    .name("Relative strength")
                    .reference(FloatSpec().member(FloatGetterSetter{radiationStrengthGetter, radiationStrengthSetter}).min(0.0f).max(1.0f))
                    .tooltip("Cells can emit energy particles over time. A portion of this energy can be released directly near the cell, while the rest is "
                             "utilized by one of the available radiation sources. This parameter determines the fraction of energy assigned to the emitted "
                             "energy particle in the vicinity of the cell. Values between 0 and 1 are permitted."),
                ParameterSpec()
                    .name("Absorption factor")
                    .reference(FloatSpec().member(&SimulationParametersZoneValues::radiationAbsorption).min(0.0f).max(1.0f).logarithmic(true).format("%.4f"))
                    .tooltip("The fraction of energy that a cell can absorb from an incoming energy particle can be specified here."),
                ParameterSpec()
                    .name("Radiation type I: Strength")
                    .reference(
                        FloatSpec().member(&SimulationParametersZoneValues::radiationType1_strength).min(0.0f).max(0.01f).logarithmic(true).format("%.6f"))
                    .tooltip("Indicates how energetic the emitted particles of aged cells are."),
                ParameterSpec()
                    .name("Radiation type I: Minimum age")
                    .reference(IntSpec().member(&SimulationParameters::radiationType1_minimumAge).min(0).max(10000000).logarithmic(true).infinity(true))
                    .tooltip("The minimum age of a cell can be defined here, from which it emits energy particles."),
                ParameterSpec()
                    .name("Radiation type II: Strength")
                    .reference(FloatSpec().member(&SimulationParameters::radiationType2_strength).min(0.0f).max(0.01f).logarithmic(true).format("%.6f"))
                    .tooltip("Indicates how energetic the emitted particles of high energy cells are."),
                ParameterSpec()
                    .name("Radiation type II: Energy threshold")
                    .reference(
                        FloatSpec()
                            .member(&SimulationParameters::radiationType2_energyThreshold)
                            .min(0.0f)
                            .max(100000.0f)
                            .logarithmic(true)
                            .infinity(true)
                            .format("%.1f"))
                    .tooltip("The minimum energy of a cell can be defined here, from which it emits energy particles."),
                ParameterSpec()
                    .name("Minimum split energy")
                    .reference(
                        FloatSpec().member(&SimulationParameters::particleSplitEnergy).min(1.0f).max(10000.0f).logarithmic(true).infinity(true).format("%.0f"))
                    .tooltip("The minimum energy of an energy particle after which it can split into two particles, whereby it receives a small momentum. The "
                             "splitting does not occur immediately, but only after a certain time."),
                ParameterSpec()
                    .name("Energy to cell transformation")
                    .reference(BoolSpec().member(&SimulationParameters::particleTransformationAllowed))
                    .tooltip("If activated, an energy particle will transform into a cell if the energy of the particle exceeds the normal energy value."),
            }),
        ParameterGroupSpec()
            .name("Cell life cycle")
            .parameters({
                ParameterSpec()
                    .name("Maximum age")
                    .reference(IntSpec().member(&SimulationParameters::maxCellAge).min(1).max(1e7).logarithmic(true).infinity(true))
                    .tooltip("Defines the maximum age of a cell. If a cell exceeds this age it will be transformed to an energy particle."),
                ParameterSpec()
                    .name("Minimum energy")
                    .reference(FloatSpec().member(&SimulationParametersZoneValues::minCellEnergy).min(10.0f).max(200.0f))
                    .tooltip("Minimum energy a cell needs to exist."),
                ParameterSpec()
                    .name("Normal energy")
                    .reference(FloatSpec().member(&SimulationParameters::normalCellEnergy).min(10.0f).max(200.0f))
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
                    .reference(FloatSpec().member(&SimulationParametersZoneValues::cellDeathProbability).min(1e-6f).max(1e-1f).format("%.6f").logarithmic(true))
                    .tooltip("The probability per time step with which a cell will disintegrate (i.e. transform into an energy particle) when it is in the "
                             "state 'Dying'. This can occur when one of the following conditions is satisfied:\n\n" ICON_FA_CHEVRON_RIGHT
                             " The cell has too low energy.\n\n" ICON_FA_CHEVRON_RIGHT " The cell has exceeded its maximum age."),
                ParameterSpec()
                    .name("Cell death consequences")
                    .reference(AlternativeSpec()
                                   .member(&SimulationParameters::cellDeathConsequences)
                                   .alternatives({{"None", {}}, {"Entire creature dies", {}}, {"Detached creature parts die", {}}}))
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
                    .reference(FloatSpec().member(&SimulationParametersZoneValues::copyMutationNeuronData).min(0.0f).max(1.0f).format("%.7f").logarithmic(true))
                    .tooltip(
                        "This type of mutation can change the weights, biases and activation functions of neural networks of each neuron cell encoded in the "
                        "genome."),
                ParameterSpec()
                    .name("Cell properties")
                    .reference(
                        FloatSpec().member(&SimulationParametersZoneValues::copyMutationCellProperties).min(0.0f).max(1.0f).format("%.7f").logarithmic(true))
                    .tooltip("This type of mutation changes a random property (e.g. (input) execution order number, required energy, block output and "
                             "function-specific properties such as minimum density for sensors, neural net weights etc.). The spatial structure, color, cell "
                             "function type and self-replication capabilities are not changed. This mutation is applied to each encoded cell in the genome."),
                ParameterSpec()
                    .name("Geometry")
                    .reference(FloatSpec().member(&SimulationParametersZoneValues::copyMutationGeometry).min(0.0f).max(1.0f).format("%.7f").logarithmic(true))
                    .tooltip("This type of mutation changes the geometry type, connection distance, stiffness and single construction flag. The probability of "
                             "a change is given by the specified value times the number of coded cells in the genome."),
                ParameterSpec()
                    .name("Custom geometry")
                    .reference(
                        FloatSpec().member(&SimulationParametersZoneValues::copyMutationCustomGeometry).min(0.0f).max(1.0f).format("%.7f").logarithmic(true))
                    .tooltip("This type of mutation only changes angles and required connections of custom geometries. The probability of a change is given by "
                             "the specified value times the number of coded cells in the genome."),
                ParameterSpec()
                    .name("Cell function type")
                    .reference(FloatSpec().member(&SimulationParametersZoneValues::copyMutationCellType).min(0.0f).max(1.0f).format("%.7f").logarithmic(true))
                    .tooltip("This type of mutation changes the type of cell function. The changed cell function will have random properties. The probability "
                             "of a change is given by the specified value times the number of coded cells in the genome. If the flag 'Preserve "
                             "self-replication' is disabled it can also alter self-replication capabilities by changing a constructor to "
                             "something else or vice versa."),
                ParameterSpec()
                    .name("Insertion")
                    .reference(FloatSpec().member(&SimulationParametersZoneValues::copyMutationInsertion).min(0.0f).max(1.0f).format("%.7f").logarithmic(true))
                    .tooltip("This type of mutation inserts a new cell description to the genome at a random position. The probability of a change is given by "
                             "the specified value times the number of coded cells in the genome."),
                ParameterSpec()
                    .name("Deletion")
                    .reference(FloatSpec().member(&SimulationParametersZoneValues::copyMutationDeletion).min(0.0f).max(1.0f).format("%.7f").logarithmic(true))
                    .tooltip("This type of mutation deletes a cell description from the genome at a random position. The probability of a change is given by "
                             "the specified value times the number of coded cells in the genome."),
                ParameterSpec()
                    .name("Translation")
                    .reference(
                        FloatSpec().member(&SimulationParametersZoneValues::copyMutationTranslation).min(0.0f).max(1.0f).format("%.7f").logarithmic(true))
                    .tooltip("This type of mutation moves a block of cell descriptions from the genome at a random position to a new random position."),
                ParameterSpec()
                    .name("Duplication")
                    .reference(
                        FloatSpec().member(&SimulationParametersZoneValues::copyMutationDuplication).min(0.0f).max(1.0f).format("%.7f").logarithmic(true))
                    .tooltip("This type of mutation copies a block of cell descriptions from the genome at a random position to a new random position."),
                ParameterSpec()
                    .name("Individual cell color")
                    .reference(FloatSpec().member(&SimulationParametersZoneValues::copyMutationCellColor).min(0.0f).max(1.0f).format("%.7f").logarithmic(true))
                    .tooltip("This type of mutation alters the color of a single cell descriptions in a genome by using the specified color transitions. The "
                             "probability of a change is given by the specified value times the number of coded cells in the genome."),
                ParameterSpec()
                    .name("Sub-genome color")
                    .reference(
                        FloatSpec().member(&SimulationParametersZoneValues::copyMutationSubgenomeColor).min(0.0f).max(1.0f).format("%.7f").logarithmic(true))
                    .tooltip("This type of mutation alters the color of all cell descriptions in a sub-genome by using the specified color transitions."),
                ParameterSpec()
                    .name("Genome color")
                    .reference(
                        FloatSpec().member(&SimulationParametersZoneValues::copyMutationGenomeColor).min(0.0f).max(1.0f).format("%.7f").logarithmic(true))
                    .tooltip("This type of mutation alters the color of all cell descriptions in a genome by using the specified color transitions."),
                ParameterSpec()
                    .name("Color transitions")
                    .reference(BoolSpec().member(&SimulationParameters::copyMutationColorTransitions))
                    .tooltip("The color transitions are used for color mutations. The row index indicates the source color and the column index the target "
                             "color."),
                ParameterSpec()
                    .name("Prevent genome depth increase")
                    .reference(BoolSpec().member(&SimulationParameters::copyMutationPreventDepthIncrease))
                    .tooltip("A genome has a tree-like structure because it can contain sub-genomes. If this flag is activated, the mutations will "
                             "not increase the depth of the genome structure."),
                ParameterSpec()
                    .name("Mutate self-replication")
                    .reference(BoolSpec().member(&SimulationParameters::copyMutationSelfReplication))
                    .tooltip("If activated, a mutation can also alter self-replication capabilities in the genome by changing a constructor cell to "
                             "something else or vice versa."),
            }),
        ParameterGroupSpec()
            .name("Cell type: Attacker")
            .parameters({
                ParameterSpec()
                    .name("Energy cost")
                    .reference(FloatSpec().member(&SimulationParametersZoneValues::attackerEnergyCost).min(0).max(1.0f).logarithmic(true).format("%.5f"))
                    .tooltip("Amount of energy lost by an attempted attack of a cell in form of emitted energy particles."),
                ParameterSpec()
                    .name("Food chain color matrix")
                    .reference(FloatSpec().member(&SimulationParametersZoneValues::attackerFoodChainColorMatrix).min(0.0f).max(1.0f).format("%.2f"))
                    .tooltip(
                        "This matrix can be used to determine how well one cell can attack another cell. The color of the attacking cell correspond to the "
                        "row number and the color of the attacked cell to the column number. A value of 0 means that the attacked cell cannot be digested, "
                        "i.e. no energy can be obtained. A value of 1 means that the maximum energy can be obtained in the digestion process.\n\nExample: If "
                        "a zero is entered in row 2 (red) and column 3 (green), it means that red cells cannot eat green cells."),
                ParameterSpec()
                    .name("Attack strength")
                    .reference(FloatSpec().member(&SimulationParameters::attackerStrength).min(0).max(0.5f).logarithmic(true))
                    .tooltip("Indicates the portion of energy through which a successfully attacked cell is weakened. However, this energy portion can be "
                             "influenced by other factors adjustable within the attacker's simulation parameters."),
                ParameterSpec()
                    .name("Attack radius")
                    .reference(FloatSpec().member(&SimulationParameters::attackerRadius).min(0).max(3.0f))
                    .tooltip("The maximum distance over which an attacker cell can attack another cell."),
                ParameterSpec()
                    .name("Complex creature protection")
                    .reference(FloatSpec().member(&SimulationParametersZoneValues::attackerComplexCreatureProtection).min(0).max(20.0f).format("%.2f"))
                    .tooltip("The larger this parameter is, the less energy can be gained by attacking creatures with more complex genomes."),
                ParameterSpec()
                    .name("Destroy cells")
                    .reference(BoolSpec().member(&SimulationParameters::attackerDestroyCells))
                    .tooltip("If activated, the attacker cell is able to destroy other cells. If deactivated, it only damages them."),
            }),
        ParameterGroupSpec()
            .name("Cell type: Constructor")
            .parameters({
                ParameterSpec()
                    .name("Connection distance")
                    .reference(FloatSpec().member(&SimulationParameters::constructorConnectingCellDistance).min(0.1f).max(3.0f))
                    .tooltip("The constructor can automatically connect constructed cells to other cells in the vicinity within this distance."),
                ParameterSpec()
                    .name("Completeness check")
                    .reference(BoolSpec().member(&SimulationParameters::constructorCompletenessCheck))
                    .tooltip("If activated, a self-replication process can only start when all other non-self-replicating constructors in the cell network are "
                             "finished."),
            }),
        ParameterGroupSpec()
            .name("Cell type: Defender")
            .parameters({
                ParameterSpec()
                    .name("Anti-attacker strength")
                    .reference(FloatSpec().member(&SimulationParameters::defenderAntiAttackerStrength).min(0.0f).max(5.0f))
                    .tooltip("If an attacked cell is connected to defender cells or itself a defender cell the attack strength is reduced by this factor."),
                ParameterSpec()
                    .name("Anti-injector strength")
                    .reference(FloatSpec().member(&SimulationParameters::defenderAntiInjectorStrength).min(0.0f).max(5.0f))
                    .tooltip("If a constructor cell is attacked by an injector and connected to defender cells, the injection duration is increased by this "
                             "factor."),
            }),
        ParameterGroupSpec()
            .name("Cell type: Injector")
            .parameters({
                ParameterSpec()
                    .name("Injection radius")
                    .reference(FloatSpec().member(&SimulationParameters::injectorInjectionRadius).min(0.1f).max(4.0f))
                    .tooltip("The maximum distance over which an injector cell can infect another cell."),
                ParameterSpec()
                    .name("Injection time")
                    .reference(IntSpec().member(&SimulationParameters::injectorInjectionTime).min(0).max(100000).logarithmic(true))
                    .tooltip("The number of activations an injector cell requires to infect another cell. One activation usually takes 6 time steps. The row "
                             "number determines the color of the injector cell, while the column number corresponds to the color of the infected cell."),
            }),
        ParameterGroupSpec()
            .name("Cell type: Muscle")
            .parameters({
                ParameterSpec()
                    .name("Energy cost")
                    .reference(FloatSpec().member(&SimulationParameters::muscleEnergyCost).min(0).max(5.0f).format("%.5f").logarithmic(true))
                    .tooltip("Amount of energy lost by a muscle action of a cell in form of emitted energy particles."),
                ParameterSpec()
                    .name("Movement acceleration")
                    .reference(FloatSpec().member(&SimulationParameters::muscleMovementAcceleration).min(0).max(10.0f).logarithmic(true))
                    .tooltip("The maximum value by which a muscle cell can modify its velocity during activation. This parameter applies only to muscle cells "
                             "which are in movement mode."),
                ParameterSpec()
                    .name("Crawling acceleration")
                    .reference(FloatSpec().member(&SimulationParameters::muscleCrawlingAcceleration).min(0).max(10.0f).logarithmic(true))
                    .tooltip("Amount of energy lost by a muscle action of a cell in form of emitted energy particles."),
                ParameterSpec()
                    .name("Bending acceleration")
                    .reference(FloatSpec().member(&SimulationParameters::muscleBendingAcceleration).min(0).max(10.0f).logarithmic(true))
                    .tooltip("The maximum value by which a muscle cell can modify its velocity during a bending action. This parameter applies "
                             "only to muscle cells which are in bending mode."),
            }),
        ParameterGroupSpec()
            .name("Cell type: Sensor")
            .parameters({
                ParameterSpec()
                    .name("Radius")
                    .reference(FloatSpec().member(&SimulationParameters::sensorRadius).min(10.0f).max(800.0f))
                    .tooltip("The maximum radius in which a sensor cell can detect mass concentrations."),
            }),
        ParameterGroupSpec()
            .name("Cell type: Transmitter")
            .parameters({
                ParameterSpec()
                    .name("Energy distribution radius")
                    .reference(FloatSpec().member(&SimulationParameters::transmitterEnergyDistributionRadius).min(0).max(5.0f))
                    .tooltip("The maximum distance over which a transmitter cell transfers its additional energy to nearby transmitter or constructor cells."),
                ParameterSpec()
                    .name("Energy distribution Value")
                    .reference(FloatSpec().member(&SimulationParameters::transmitterEnergyDistributionValue).min(0).max(20.0f))
                    .tooltip("The amount of energy which a transmitter cell can transfer to nearby transmitter or constructor cells or to connected cells."),
                ParameterSpec()
                    .name("Same creature energy distribution")
                    .reference(BoolSpec().member(&SimulationParameters::transmitterEnergyDistributionSameCreature))
                    .tooltip("If activated, the transmitter cells can only transfer energy to nearby cells belonging to the same creature."),
            }),
        ParameterGroupSpec()
            .name("Cell type: Reconnector")
            .parameters({
                ParameterSpec()
                    .name("Radius")
                    .reference(FloatSpec().member(&SimulationParameters::reconnectorRadius).min(0).max(3.0f))
                    .tooltip("The maximum radius in which a reconnector cell can establish or destroy connections to other cells."),
            }),
        ParameterGroupSpec()
            .name("Cell type: Detonator")
            .parameters({
                ParameterSpec()
                    .name("Blast radius")
                    .reference(FloatSpec().member(&SimulationParameters::detonatorRadius).min(0).max(10.0f))
                    .tooltip("The radius of the detonation."),
                ParameterSpec()
                    .name("Chain explosion probability")
                    .reference(FloatSpec().member(&SimulationParameters::detonatorChainExplosionProbability).min(0).max(1.0f))
                    .tooltip("The probability that the explosion of one detonator will trigger the explosion of other detonators within the blast radius."),
            }),
        ParameterGroupSpec()
            .name("Advanced energy absorption control")
            .expertToggleAddress(EXPERT_VALUE_OFFSET(advancedAbsorptionControl))
            .parameters({
                ParameterSpec()
                    .name("Low genome complexity penalty")
                    .reference(
                        FloatSpec().member(&SimulationParametersZoneValues::radiationAbsorptionLowGenomeComplexityPenalty).min(0).max(1.0f).format("%.2f"))
                    .tooltip("When this parameter is increased, cells with fewer genome complexity will absorb less energy from an incoming energy particle."),
                ParameterSpec()
                    .name("Low connection penalty")
                    .reference(FloatSpec().member(&SimulationParameters::radiationAbsorptionLowConnectionPenalty).min(0).max(5.0f).format("%.1f"))
                    .tooltip("When this parameter is increased, cells with fewer cell connections will absorb less energy from an incoming energy particle."),
                ParameterSpec()
                    .name("High velocity penalty")
                    .reference(
                        FloatSpec().member(&SimulationParameters::radiationAbsorptionHighVelocityPenalty).min(0).max(30.0f).logarithmic(true).format("%.2f"))
                    .tooltip("When this parameter is increased, fast moving cells will absorb less energy from an incoming energy particle."),
                ParameterSpec()
                    .name("Low velocity penalty")
                    .reference(FloatSpec().member(&SimulationParametersZoneValues::radiationAbsorptionLowVelocityPenalty).min(0).max(1.0f).format("%.2f"))
                    .tooltip("When this parameter is increased, slowly moving cells will absorb less energy from an incoming energy particle."),
            }),
        ParameterGroupSpec()
            .name("Advanced attacker control")
            .expertToggleAddress(EXPERT_VALUE_OFFSET(advancedAttackerControl))
            .parameters({
                ParameterSpec()
                    .name("Same mutant protection")
                    .reference(FloatSpec().member(&SimulationParameters::attackerSameMutantProtection).min(0).max(1.0f).format("%.2f"))
                    .tooltip("The larger this parameter is, the less energy can be gained by attacking creatures with the same mutation id."),
                ParameterSpec()
                    .name("New complex mutant protection")
                    .reference(FloatSpec().member(&SimulationParametersZoneValues::attackerNewComplexMutantProtection).min(0).max(1.0f))
                    .tooltip("A high value protects new mutants with equal or greater genome complexity from being attacked."),
                ParameterSpec()
                    .name("Sensor detection factor")
                    .reference(FloatSpec().member(&SimulationParameters::attackerSensorDetectionFactor).min(0).max(1.0f))
                    .tooltip("This parameter controls whether the target must be previously detected with sensors in order to be attacked. The larger this "
                             "value is, the less energy can be gained during the attack if the target has not already been detected. For this purpose, the "
                             "attacker cell searches for connected (or connected-connected) sensor cells to see which cell networks they have detected last "
                             "time and "
                             "compares them with the attacked target."),
                ParameterSpec()
                    .name("Geometry deviation protection")
                    .reference(FloatSpec().member(&SimulationParametersZoneValues::attackerGeometryDeviationProtection).min(0).max(5.0f))
                    .tooltip("The larger this value is, the less energy a cell can gain from an attack if the local geometry of the attacked cell does not "
                             "match the attacking cell."),
                ParameterSpec()
                    .name("Connections mismatch protection")
                    .reference(FloatSpec().member(&SimulationParametersZoneValues::attackerConnectionsMismatchProtection).min(0).max(1.0f))
                    .tooltip("The larger this parameter is, the more difficult it is to attack cells that contain more connections."),
            }),
        ParameterGroupSpec()
            .name("Cell age limiter")
            .expertToggleAddress(EXPERT_VALUE_OFFSET(cellAgeLimiter))
            .parameters({
                ParameterSpec()
                    .name("Maximum inactive cell age")
                    .reference(
                        FloatSpec()
                            .member(&SimulationParametersZoneValues::maxAgeForInactiveCells)
                            .min(1.0f)
                            .max(1e7f)
                            .format("%.0f")
                            .logarithmic(true)
                            .infinity(true))
                    .enabled(EnabledSpec()
                                 .base(&SimulationParameters::maxAgeForInactiveCellsEnabled)
                                 .zone(&SimulationParametersZoneEnabledValues::maxAgeForInactiveCellsEnabled))
                    .tooltip("Here, you can set the maximum age for a cell whose function or those of its neighbors have not been triggered. Cells which "
                             "are in state 'Under construction' are not affected by this option."),
                ParameterSpec()
                    .name("Maximum free cell age")
                    .reference(IntSpec().member(&SimulationParameters::freeCellMaxAge).min(1).max(1e7).logarithmic(true).infinity(true))
                    .enabled(EnabledSpec().base(&SimulationParameters::freeCellMaxAgeEnabled))
                    .tooltip("The maximal age of free cells (= cells that arise from energy particles) can be set here."),
                ParameterSpec()
                    .name("Reset age after construction")
                    .reference(BoolSpec().member(&SimulationParameters::resetCellAgeAfterActivation))
                    .tooltip("If this option is activated, the age of the cells is reset to 0 after the construction of their cell network is completed, "
                             "i.e. when the state of the cells changes from 'Under construction' to 'Ready'. This option is particularly useful if a low "
                             "'Maximum inactive cell age' is set, as cell networks that are under construction are inactive and could die immediately after "
                             "completion if their construction takes a long time."),
                ParameterSpec()
                    .name("Maximum age balancing")
                    //.value(BaseValueSpec()
                    //           .valueAddress(BASE_VALUE_OFFSET(maxCellAgeBalancerInterval))
                    //           .enabledValueAddress(BASE_VALUE_OFFSET(maxCellAgeBalancerEnabled)))
                    .reference(IntSpec().member(&SimulationParameters::maxCellAgeBalancerInterval).min(1e3).max(1e6).logarithmic(true))
                    .tooltip("Adjusts the maximum age at regular intervals. It increases the maximum age for the cell color where the fewest "
                             "replicators exist. Conversely, the maximum age is decreased for the cell color with the most replicators."),
            }),
        ParameterGroupSpec()
            .name("Cell color transition rules")
            .expertToggleAddress(EXPERT_VALUE_OFFSET(cellColorTransitionRules))
            .parameters({
                ParameterSpec()
                    .name("Target color and duration")
                    .reference(ColorTransitionRulesSpec().member(&SimulationParametersZoneValues::colorTransitionRules))
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
                    .reference(AlternativeSpec()
                                   .member(&SimulationParameters::cellGlowColoring)
                                   .alternatives(
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
                    .reference(FloatSpec().member(&SimulationParameters::cellGlowRadius).min(1.0f).max(8.0f))
                    .tooltip("The radius of the glow. Please note that a large radius affects the performance."),
                ParameterSpec()
                    .name("Strength")
                    .reference(FloatSpec().member(&SimulationParameters::cellGlowStrength).min(0.0f).max(1.0f))
                    .tooltip("The strength of the glow."),
            }),
        ParameterGroupSpec()
            .name("Customize deletion mutations")
            .expertToggleAddress(EXPERT_VALUE_OFFSET(customizeDeletionMutations))
            .parameters({
                ParameterSpec()
                    .name("Minimum size")
                    .reference(IntSpec().member(&SimulationParameters::cellCopyMutationDeletionMinSize).min(0).max(1000).logarithmic(true))
                    .tooltip("The minimum size of genomes (on the basis of the coded cells) is determined here that can result from delete mutations. The "
                             "default is 0."),
            }),
        ParameterGroupSpec()
            .name("Customize neuron mutations")
            .expertToggleAddress(EXPERT_VALUE_OFFSET(customizeNeuronMutations))
            .parameters({
                ParameterSpec()
                    .name("Affected weights")
                    .reference(FloatSpec().member(&SimulationParameters::cellCopyMutationNeuronDataWeight).min(0.0f).max(1.0f).format("%.3f"))
                    .tooltip("The proportion of weights in the neuronal network of a cell that are changed within a neuron mutation. The default is 0.2."),
                ParameterSpec()
                    .name("Affected biases")
                    .reference(FloatSpec().member(&SimulationParameters::cellCopyMutationNeuronDataBias).min(0.0f).max(1.0f).format("%.3f"))
                    .tooltip("The proportion of biases in the neuronal network of a cell that are changed within a neuron mutation. The default is 0.2."),
                ParameterSpec()
                    .name("Affected activation functions")
                    .reference(FloatSpec().member(&SimulationParameters::cellCopyMutationNeuronDataActivationFunction).min(0.0f).max(1.0f).format("%.3f"))
                    .tooltip("The proportion of activation functions in the neuronal network of a cell that are changed within a neuron mutation. The default "
                             "is 0.05."),
                ParameterSpec()
                    .name("Reinforcement factor")
                    .reference(FloatSpec().member(&SimulationParameters::cellCopyMutationNeuronDataReinforcement).min(1.0f).max(1.2f).format("%.3f"))
                    .tooltip(
                        "If a weight or bias of the neural network is adjusted by a mutation, it can either be reinforced, weakened or shifted by an offset. "
                        "The factor that is used for reinforcement is defined here. The default is 1.05."),
                ParameterSpec()
                    .name("Damping factor")
                    .reference(FloatSpec().member(&SimulationParameters::cellCopyMutationNeuronDataDamping).min(1.0f).max(1.2f).format("%.3f"))
                    .tooltip(
                        "If a weight or bias of the neural network is adjusted by a mutation, it can either be reinforced, weakened or shifted by an offset. "
                        "The factor that is used for weakening is defined here. The default is 1.05."),
                ParameterSpec()
                    .name("Offset")
                    .reference(FloatSpec().member(&SimulationParameters::cellCopyMutationNeuronDataOffset).min(0.0f).max(0.2f).format("%.3f"))
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
                    .reference(
                        FloatSpec().member(&SimulationParameters::externalEnergy).min(0.0f).max(100000000.0f).format("%.0f").logarithmic(true).infinity(true))
                    .tooltip(
                        "This parameter can be used to set the amount of energy of an external energy pool. This type of energy can then be "
                        "transferred to all constructor cells at a certain rate (see inflow settings).\n\nWarning: Too much external energy can result in a "
                        "massive production of cells and slow down or even crash the simulation."),
                ParameterSpec()
                    .name("Inflow")
                    .reference(FloatSpec().member(&SimulationParameters::externalEnergyInflowFactor).min(0.0f).max(1.0f).format("%.5f").logarithmic(true))
                    .tooltip(
                        "Here one can specify the fraction of energy transferred to constructor cells.\n\nFor example, a value of 0.05 means that "
                        "each time a constructor cell tries to build a new cell, 5% of the required energy is transferred for free from the external energy "
                        "source."),
                ParameterSpec()
                    .name("Conditional inflow")
                    .reference(
                        FloatSpec().member(&SimulationParameters::externalEnergyConditionalInflowFactor).min(0.00f).max(1.0f).format("%.5f").logarithmic(true))
                    .tooltip("Here one can specify the fraction of energy transferred to constructor cells if they can provide the remaining energy for the "
                             "construction process.\n\nFor example, a value of 0.6 means that a constructor cell receives 60% of the energy required to "
                             "build the new cell for free from the external energy source. However, it must provide 40% of the energy required by itself. "
                             "Otherwise, no energy will be transferred."),
                ParameterSpec()
                    .name("Inflow only for non-replicators")
                    .reference(BoolSpec().member(&SimulationParameters::externalEnergyInflowOnlyForNonSelfReplicators))
                    .tooltip("If activated, external energy can only be transferred to constructor cells that are not self-replicators. "
                             "This option can be used to foster the evolution of additional body parts."),
                ParameterSpec()
                    .name("Backflow")
                    .reference(FloatSpec().member(&SimulationParameters::externalEnergyBackflowFactor).min(0.0f).max(1.0f))
                    .tooltip("The proportion of energy that flows back from the simulation to the external energy pool. Each time a cell loses energy "
                             "or dies a fraction of its energy will be taken. The remaining "
                             "fraction of the energy stays in the simulation and will be used to create a new energy particle."),
                ParameterSpec()
                    .name("Backflow limit")
                    .reference(
                        FloatSpec()
                            .member(&SimulationParameters::externalEnergyBackflowLimit)
                            .min(0.0f)
                            .max(1e8f)
                            .format("%.0f")
                            .logarithmic(true)
                            .infinity(true))
                    .tooltip("Energy from the simulation can only flow back into the external energy pool as long as the amount of external energy is "
                             "below this value."),
            }),
        ParameterGroupSpec()
            .name("Genome complexity measurement")
            .expertToggleAddress(EXPERT_VALUE_OFFSET(genomeComplexityMeasurement))
            .parameters({
                ParameterSpec()
                    .name("Size factor")
                    .reference(FloatSpec().member(&SimulationParameters::genomeComplexitySizeFactor).min(0.0f).max(1.0f).format("%.2f"))
                    .tooltip("This parameter controls how the number of encoded cells in the genome influences the calculation of its complexity."),
                ParameterSpec()
                    .name("Ramification factor")
                    .reference(FloatSpec().member(&SimulationParameters::genomeComplexityRamificationFactor).min(0.0f).max(20.0f).format("%.2f"))
                    .tooltip("With this parameter, the number of ramifications of the cell structure to the genome is taken into account for the "
                             "calculation of the genome complexity. For instance, genomes that contain many sub-genomes or many construction branches will "
                             "then have a high complexity value."),
                ParameterSpec()
                    .name("Depth level")
                    .reference(IntSpec().member(&SimulationParameters::genomeComplexityDepthLevel).min(1).max(20).infinity(true))
                    .tooltip("This allows to specify up to which level of the sub-genomes the complexity calculation should be carried out. For example, a "
                             "value of 2 means that the sub- and sub-sub-genomes are taken into account in addition to the main genome."),
            }),
    });
}

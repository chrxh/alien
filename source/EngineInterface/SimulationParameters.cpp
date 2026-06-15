#include "SimulationParameters.h"

#include <Fonts/IconsFontAwesome5.h>

#include "LocationHelper.h"
#include "ParametersEditService.h"
#include "SimulationParametersSpecification.h"

ParametersSpec const& SimulationParameters::getSpec()
{
    static std::optional<ParametersSpec> spec;
    if (!spec.has_value()) {
        std::vector<std::pair<std::string, std::vector<ParameterSpec>>> cellTypeStrings;
        for (int i = 0; i < CellType_Count; ++i) {
            cellTypeStrings.emplace_back(std::make_pair(Const::CellTypeStrings.at(i), std::vector<ParameterSpec>()));
        }

        auto radiationStrengthGetter = [](SimulationParameters const& parameters, int orderNumber) {
            auto strength = ParametersEditService::get().getRadiationStrengths(parameters);
            if (orderNumber == 0) {
                return strength.values.at(0);
            } else {
                auto sourceIndex = LocationHelper::findLocationArrayIndex(parameters, orderNumber);
                return parameters.sourceRelativeStrength.sourceValues[sourceIndex].value;
            }
        };

        auto radiationStrengthSetter = [](float value, SimulationParameters& parameters, int orderNumber) {
            auto& editService = ParametersEditService::get();
            auto strength = ParametersEditService::get().getRadiationStrengths(parameters);
            auto editedStrength = strength;
            int strengthIndex = orderNumber == 0 ? 0 : LocationHelper::findLocationArrayIndex(parameters, orderNumber) + 1;
            editedStrength.values.at(strengthIndex) = value;
            editService.adaptRadiationStrengths(editedStrength, strength, strengthIndex);
            editService.applyRadiationStrengths(parameters, editedStrength);
        };

        std::string const coloringTooltip =
            "Here, one can set how the cells are to be colored during rendering. \n\n" ICON_FA_CHEVRON_RIGHT
            " Energy: The more energy a cell has, the brighter it is displayed. A grayscale is used.\n\n" ICON_FA_CHEVRON_RIGHT
            " Customization colors: Each cell is assigned one of 7 default colors, which is displayed with this option. \n\n" ICON_FA_CHEVRON_RIGHT
            " Mutants: Different mutants are represented by different colors (only larger structural mutations such as translations or "
            "duplications are taken into account).\n\n" ICON_FA_CHEVRON_RIGHT
            " Mutant and cell function: Combination of mutants and cell function coloring.\n\n" ICON_FA_CHEVRON_RIGHT
            " Cell state: blue = ready, green = under construction, white = activating, pink = detached, pale blue = reviving, red = "
            "dying\n\n" ICON_FA_CHEVRON_RIGHT
            " Genome complexity: This property can be utilized by attacker cells when the parameter 'Complex creature protection' is "
            "activated (see tooltip there). The coloring is as follows: blue = creature with low bonus (usually small or simple genome structure), "
            "red = large bonus\n\n" ICON_FA_CHEVRON_RIGHT " Specific cell function: A specific type of cell function can be highlighted, which is "
            "selected in the next parameter.\n\n" ICON_FA_CHEVRON_RIGHT " Every cell function: The cells are colored according to their cell function.";

        spec = ParametersSpec().groups({
            ParameterGroupSpec().name("General").parameters({
                ParameterSpec().name("Project name").reference(Char64Spec().member(&SimulationParameters::projectName)),
                ParameterSpec().name("Layer name").reference(Char64Spec().member(&SimulationParameters::layerName)),
                ParameterSpec().name("Source name").reference(Char64Spec().member(&SimulationParameters::sourceName)),
                ParameterSpec()
                    .name("Opacity")
                    .reference(FloatSpec().member(&SimulationParameters::layerOpacity).min(0.0f).max(1.0f))
                    .description("Layers can overwrite basic parameters and exert force fields. The opacity is specified with this parameter between "
                                 "0 and 1. A value of 1 means that parameters in the core area of the layer can be completely overwritten, while a value "
                                 "of 0 means that the layer has no effect."),
                ParameterSpec()
                    .name("Relative strength")
                    .reference(FloatSpec()
                                   .member(&SimulationParameters::sourceRelativeStrength)
                                   .getterSetter(FloatGetterSetter{radiationStrengthGetter, radiationStrengthSetter})
                                   .min(0.0f)
                                   .max(1.0f))
                    .description(
                        "Cells can emit energy particles over time. A portion of this energy can be released directly near the cell, while the rest is "
                        "utilized by one of the available radiation sources. This parameter determines the fraction of energy assigned to the emitted "
                        "energy particle in the current radiation source. Values between 0 and 1 are permitted."),
            }),
            ParameterGroupSpec()
                .name("Visualization")
                .parameters({
                    ParameterSpec().name("Background color").reference(ColorSpec().member(&SimulationParameters::backgroundColor)),
                    ParameterSpec()
                        .name("Customization colors")
                        .reference(ColorSpec().member(&SimulationParameters::customizationColors))
                        .description("Defines the 10 standard customization colors used for cell rendering and color-based editing throughout the simulation."),
                    ParameterSpec()
                        .name("Object coloring")
                        .reference(AlternativeSpec()
                                       .member(&SimulationParameters::objectColoring)
                                       .alternatives({{"Energy", {}}})
                                       .alternatives({
                                           {"Energy", {}},
                                           {"Customization", {}},
                                           {"Lineage + Customization", {}},
                                       }))
                        .description(coloringTooltip),
                    ParameterSpec()
                        .name("Glow")
                        .reference(FloatSpec().member(&SimulationParameters::glow).min(0.0f).max(1.0f))
                        .description("Controls the bloom effect intensity. At 0, no glow is applied. At 1, maximum glow is applied."),
                    ParameterSpec()
                        .name("Borderless rendering")
                        .reference(BoolSpec().member(&SimulationParameters::borderlessRendering))
                        .description("If activated, the simulation is rendered periodically in the view port."),
                    ParameterSpec()
                        .name("Grid lines")
                        .reference(BoolSpec().member(&SimulationParameters::gridLines))
                        .description("This option draws a suitable grid in the background depending on the zoom level."),
                    ParameterSpec()
                        .name("Mark reference domain")
                        .reference(BoolSpec().member(&SimulationParameters::markReferenceDomain))
                        .description("This option draws a suitable grid in the background depending on the zoom level."),
                    ParameterSpec()
                        .name("Show radiation center")
                        .reference(BoolSpec().member(&SimulationParameters::sourceShowRadiationCenter))
                        .description("This option draws a red cross in the center of the radiation source."),
                }),
            ParameterGroupSpec()
                .name("Location")
                .parameters({
                    ParameterSpec()
                        .name("Position (x,y)")
                        .reference(Float2Spec()
                                       .member(&SimulationParameters::layerPosition)
                                       .min(RealVector2D{0.0f, 0.0f})
                                       .max(WorldSize())
                                       .format("%.2f")
                                       .mousePicker(true)),
                    ParameterSpec()
                        .name("Velocity (x,y)")
                        .reference(Float2Spec()
                                       .member(&SimulationParameters::layerVelocity)
                                       .min(RealVector2D{-4.0f, -4.0f})
                                       .max(RealVector2D{4.0f, 4.0f})
                                       .format("%.3f")),
                    ParameterSpec()
                        .name("Position (x,y)")
                        .reference(Float2Spec()
                                       .member(&SimulationParameters::sourcePosition)
                                       .min(RealVector2D{0.0f, 0.0f})
                                       .max(WorldSize())
                                       .format("%.2f")
                                       .mousePicker(true)),
                    ParameterSpec()
                        .name("Velocity (x,y)")
                        .reference(Float2Spec()
                                       .member(&SimulationParameters::sourceVelocity)
                                       .min(RealVector2D{-4.0f, -4.0f})
                                       .max(RealVector2D{4.0f, 4.0f})
                                       .format("%.3f")),
                }),
            ParameterGroupSpec().name("Shape").parameters({
                ParameterSpec().name("Shape").reference(
                    AlternativeSpec()
                        .member(&SimulationParameters::layerShape)
                        .alternatives(
                            {{"Circular",
                              {ParameterSpec()
                                   .name("Core radius")
                                   .reference(FloatSpec().member(&SimulationParameters::layerCoreRadius).min(0.0f).max(MaxWorldRadiusSize()).format("%.2f"))}},
                             {"Rectangular",
                              {ParameterSpec()
                                   .name("Core size (width,height)")
                                   .reference(Float2Spec()
                                                  .member(&SimulationParameters::layerCoreRect)
                                                  .min(RealVector2D{0.0f, 0.0f})
                                                  .max(WorldSize())
                                                  .format("%.2f"))}}})),
                ParameterSpec()
                    .name("Fade-out radius")
                    .reference(FloatSpec().member(&SimulationParameters::layerFadeoutRadius).min(0.0f).max(MaxWorldRadiusSize()).format("%.2f")),
                ParameterSpec().name("Shape").reference(
                    AlternativeSpec()
                        .member(&SimulationParameters::sourceShapeType)
                        .alternatives(
                            {{"Circular",
                              {
                                  ParameterSpec().name("Radius").reference(
                                      FloatSpec().member(&SimulationParameters::sourceCircularRadius).min(0.0f).max(MaxWorldRadiusSize()).format("%.2f")),
                              }},
                             {"Rectangular",
                              {
                                  ParameterSpec()
                                      .name("Size (width, height)")
                                      .reference(Float2Spec()
                                                     .member(&SimulationParameters::sourceRectangularRect)
                                                     .min(RealVector2D{0.0f, 0.0f})
                                                     .max(WorldSize())
                                                     .format("%.2f")),
                              }}})),
            }),
            ParameterGroupSpec()
                .name("Force field")
                .parameters({
                    ParameterSpec()
                        .name("Field type")
                        .reference(
                            AlternativeSpec()
                                .member(&SimulationParameters::layerForceFieldType)
                                .alternatives(
                                    {{"None", {}},
                                     {"Radial",
                                      {
                                          ParameterSpec()
                                              .name("Orientation")
                                              .reference(AlternativeSpec()
                                                             .member(&SimulationParameters::layerRadialForceFieldOrientation)
                                                             .alternatives({{"Clockwise", {}}, {"Counter clockwise", {}}})),
                                          ParameterSpec()
                                              .name("Strength")
                                              .reference(
                                                  FloatSpec()
                                                      .member(&SimulationParameters::layerRadialForceFieldStrength)
                                                      .min(0.0f)
                                                      .max(0.5f)
                                                      .format("%.6f")
                                                      .logarithmic(true)),
                                          ParameterSpec()
                                              .name("Drift angle")
                                              .reference(
                                                  FloatSpec()
                                                      .member(&SimulationParameters::layerRadialForceFieldDriftAngle)
                                                      .min(-180.00f)
                                                      .max(180.0f)
                                                      .format("%.1f")),
                                      }},
                                     {"Central",
                                      {
                                          ParameterSpec()
                                              .name("Strength")
                                              .reference(
                                                  FloatSpec()
                                                      .member(&SimulationParameters::layerCentralForceFieldStrength)
                                                      .min(0.0f)
                                                      .max(0.5f)
                                                      .logarithmic(true)
                                                      .format("%.6f")),
                                      }},
                                     {"Linear",
                                      {
                                          ParameterSpec().name("Angle").reference(
                                              FloatSpec().member(&SimulationParameters::layerLinearForceFieldAngle).min(-180.0f).max(180.0f).format("%.1f")),
                                          ParameterSpec()
                                              .name("Strength")
                                              .reference(
                                                  FloatSpec()
                                                      .member(&SimulationParameters::layerLinearForceFieldStrength)
                                                      .min(0.0f)
                                                      .max(0.5f)
                                                      .logarithmic(true)
                                                      .format("%.6f")),
                                      }},
                                     {"Perlin noise",
                                      {
                                          ParameterSpec()
                                              .name("Strength")
                                              .reference(
                                                  FloatSpec()
                                                      .member(&SimulationParameters::layerPerlinNoiseForceFieldStrength)
                                                      .min(0.0f)
                                                      .max(0.5f)
                                                      .logarithmic(true)
                                                      .format("%.6f")),
                                          ParameterSpec()
                                              .name("Spatial structure size")
                                              .reference(
                                                  FloatSpec()
                                                      .member(&SimulationParameters::layerPerlinNoiseForceFieldSpatialSize)
                                                      .min(0.1f)
                                                      .max(1000.0f)
                                                      .logarithmic(true)
                                                      .format("%.0f")),
                                          ParameterSpec()
                                              .name("Temporal structure size")
                                              .reference(
                                                  FloatSpec()
                                                      .member(&SimulationParameters::layerPerlinNoiseForceFieldTemporalSize)
                                                      .min(1.0f)
                                                      .max(1000000.0f)
                                                      .logarithmic(true)
                                                      .format("%.0f")),
                                      }}})),
                }),
            ParameterGroupSpec()
                .name("Physics: Motion")
                .parameters({
                    ParameterSpec()
                        .name("Time step size")
                        .reference(FloatSpec().member(&SimulationParameters::timestepSize).min(0.01f).max(1.0f))
                        .description(
                            "The time duration calculated in a single simulation step. Smaller values increase the accuracy of the simulation while larger "
                            "values can lead to numerical instabilities."),
                    ParameterSpec()
                        .name("Smoothing length")
                        .reference(FloatSpec().member(&SimulationParameters::smoothingLength).min(0.0f).max(3.0f))
                        .description("The smoothing length determines the region of influence of the neighboring particles for the calculation of "
                                     "density, pressure and viscosity. Values that are too small lead to numerical instabilities, while values "
                                     "that are too large cause the particles to drift apart."),
                    ParameterSpec()
                        .name("Pressure")
                        .reference(FloatSpec().member(&SimulationParameters::pressureStrength).min(0.0f).max(0.3f))
                        .description("This parameter allows to control the strength of the pressure."),
                    ParameterSpec()
                        .name("Viscosity")
                        .reference(FloatSpec().member(&SimulationParameters::viscosityStrength).min(0.0f).max(0.3f))
                        .description("This parameter be used to control the strength of the viscosity. Larger values lead to a smoother movement."),
                    ParameterSpec()
                        .name("Friction")
                        .reference(FloatSpec().member(&SimulationParameters::friction).min(0.0f).max(1.0f).logarithmic(true).format("%.4f"))
                        .description("This specifies the fraction of the velocity that is slowed down per time step."),
                    ParameterSpec()
                        .name("Inner friction")
                        .reference(FloatSpec().member(&SimulationParameters::innerFriction).min(0.0f).max(1.0f).logarithmic(true).format("%.4f"))
                        .visible(false),
                    ParameterSpec()
                        .name("Rigidity")
                        .reference(FloatSpec().member(&SimulationParameters::rigidity).min(0.0f).max(1.0f).format("%.2f"))
                        .description(
                            "Controls the rigidity of connected cells. A higher value will cause connected cells to move more uniformly as a rigid body."),
                }),
            ParameterGroupSpec()
                .name("Physics: Thresholds")
                .parameters({
                    ParameterSpec()
                        .name("Maximum velocity")
                        .reference(FloatSpec().member(&SimulationParameters::maxVelocity).min(0.0f).max(6.0f))
                        .description("Maximum velocity that a cell can reach."),
                    ParameterSpec()
                        .name("Maximum force")
                        .reference(FloatSpec().member(&SimulationParameters::maxForce).min(0.0f).max(3.0f))
                        .description("Maximum force that can be applied to a cell without causing it to disintegrate."),
                    ParameterSpec()
                        .name("Minimum distance")
                        .reference(FloatSpec().member(&SimulationParameters::minObjectDistance).min(0.0f).max(1.0f))
                        .description("Minimum distance between two cells."),
                    ParameterSpec()
                        .name("Maximum distance")
                        .reference(FloatSpec().member(&SimulationParameters::maxBindingDistance).min(0.0f).max(5.0f))
                        .description("Maximum distance up to which a connection of two cells is possible."),
                    ParameterSpec()
                        .name("Fusion velocity")
                        .reference(FloatSpec().member(&SimulationParameters::objectFusionVelocity).min(0.0f).max(2.0f))
                        .description("Maximum force that can be applied to a cell without causing it to disintegrate."),
                }),
            ParameterGroupSpec()
                .name("Radiation")
                .parameters({
                    ParameterSpec()
                        .name("Relative strength")
                        .reference(
                            FloatSpec()
                                .member(&SimulationParameters::relativeStrengthBasePin)
                                .getterSetter(FloatGetterSetter{radiationStrengthGetter, radiationStrengthSetter})
                                .min(0.0f)
                                .max(1.0f))
                        .description(
                            "Cells can emit energy particles over time. A portion of this energy can be released directly near the cell, while the rest is "
                            "utilized by one of the available radiation sources. This parameter determines the fraction of energy assigned to the emitted "
                            "energy particle in the vicinity of the cell. Values between 0 and 1 are permitted."),
                    ParameterSpec()
                        .name("Disable radiation sources")
                        .reference(BoolSpec().member(&SimulationParameters::disableRadiationSources))
                        .description("If activated, all radiation sources within this layer are deactivated."),
                    ParameterSpec()
                        .name("Absorption factor")
                        .reference(FloatSpec().member(&SimulationParameters::radiationAbsorption).min(0.0f).max(1.0f).logarithmic(true).format("%.4f"))
                        .description("The fraction of energy that a cell can absorb from an incoming energy particle can be specified here."),
                    ParameterSpec()
                        .name("Radiation type I: Strength")
                        .reference(FloatSpec().member(&SimulationParameters::radiationType1_strength).min(0.0f).max(0.01f).logarithmic(true).format("%.6f"))
                        .description("Indicates how energetic the emitted particles of aged cells are."),
                    ParameterSpec()
                        .name("Radiation type I: Minimum age")
                        .reference(IntSpec().member(&SimulationParameters::radiationType1_minimumAge).min(0).max(10000000).logarithmic(true).infinity(true))
                        .description("The minimum age of a cell can be defined here, from which it emits energy particles."),
                    ParameterSpec()
                        .name("Radiation type II: Strength")
                        .reference(FloatSpec().member(&SimulationParameters::radiationType2_strength).min(0.0f).max(0.01f).logarithmic(true).format("%.6f"))
                        .description("Indicates how energetic the emitted particles of high energy cells are."),
                    ParameterSpec()
                        .name("Radiation type II: Threshold")
                        .reference(
                            FloatSpec()
                                .member(&SimulationParameters::radiationType2_energyThreshold)
                                .min(0.0f)
                                .max(100000.0f)
                                .logarithmic(true)
                                .infinity(true)
                                .format("%.1f"))
                        .description("The minimum energy of a cell can be defined here, from which it emits energy particles."),
                    ParameterSpec()
                        .name("Minimum split energy")
                        .reference(
                            FloatSpec()
                                .member(&SimulationParameters::particleSplitEnergy)
                                .min(1.0f)
                                .max(10000.0f)
                                .logarithmic(true)
                                .infinity(true)
                                .format("%.0f"))
                        .description(
                            "The minimum energy of an energy particle after which it can split into two particles, whereby it receives a small momentum. The "
                            "splitting does not occur immediately, but only after a certain time."),
                    ParameterSpec()
                        .name("Energy to cell transformation")
                        .reference(BoolSpec().member(&SimulationParameters::particleTransformationAllowed))
                        .description(
                            "If activated, an energy particle will transform into a cell if the energy of the particle exceeds the normal energy value."),
                    ParameterSpec()
                        .name("Radiation angle")
                        .reference(FloatSpec().member(&SimulationParameters::sourceRadiationAngle).min(-180.0f).max(180.0f).format("%.1f")),
                }),
            ParameterGroupSpec()
                .name("Cell life cycle")
                .parameters({
                    ParameterSpec()
                        .name("Maximum age")
                        .reference(IntSpec().member(&SimulationParameters::maxCellAge).min(1).max(1e7).logarithmic(true).infinity(true))
                        .description("Defines the maximum age of a cell. If a cell exceeds this age it will be transformed to an energy particle."),
                    ParameterSpec()
                        .name("Minimum energy")
                        .reference(FloatSpec().member(&SimulationParameters::minCellEnergy).min(10.0f).max(200.0f).format("%.1f"))
                        .description("Minimum energy a cell needs to exist."),
                    ParameterSpec()
                        .name("Normal energy")
                        .reference(
                            FloatSpec()
                                .member(&SimulationParameters::normalCellEnergy)
                                .min(10.0f)
                                .max(200.0f)
                                .format("%.1f")
                                .greaterThan(&SimulationParameters::minCellEnergy))
                        .description(
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
                        .reference(FloatSpec().member(&SimulationParameters::cellDeathProbability).min(1e-6f).max(1e-1f).format("%.6f").logarithmic(true))
                        .description(
                            "The probability per time step with which a cell will disintegrate (i.e. transform into an energy particle) when it is in the "
                            "state 'Dying'. This can occur when one of the following conditions is satisfied:\n\n" ICON_FA_CHEVRON_RIGHT
                            " The cell has too low energy.\n\n" ICON_FA_CHEVRON_RIGHT " The cell has exceeded its maximum age."),
                }),
            ParameterGroupSpec()
                .name("Cell construction")
                .parameters({
                    ParameterSpec()
                        .name("Connection distance")
                        .reference(FloatSpec().member(&SimulationParameters::constructorConnectingCellDistance).min(0.1f).max(4.0f))
                        .description("The constructor can automatically connect constructed cells to other cells in the vicinity within this distance."),
                }),
            ParameterGroupSpec()
                .name("Meta mutations")
                .parameters({
                    ParameterSpec()
                        .name("Neuron mutations sigma")
                        .reference(FloatSpec().member(&SimulationParameters::metaMutationNeuronsSigma).min(0.0f).max(1.0f).logarithmic(true).format("%.5f")),
                    ParameterSpec()
                        .name("Connection mutations sigma")
                        .reference(
                            FloatSpec().member(&SimulationParameters::metaMutationConnectionsSigma).min(0.0f).max(1.0f).logarithmic(true).format("%.5f")),
                    ParameterSpec()
                        .name("Cell type property mutations sigma")
                        .reference(
                            FloatSpec().member(&SimulationParameters::metaMutationCellTypePropertiesSigma).min(0.0f).max(1.0f).logarithmic(true).format("%.5f")),
                    ParameterSpec()
                        .name("Cell type mode mutations sigma")
                        .reference(
                            FloatSpec().member(&SimulationParameters::metaMutationCellTypeModeSigma).min(0.0f).max(1.0f).logarithmic(true).format("%.5f")),
                    ParameterSpec()
                        .name("Cell type mutations sigma")
                        .reference(
                            FloatSpec().member(&SimulationParameters::metaMutationCellTypeSigma).min(0.0f).max(1.0f).logarithmic(true).format("%.5f")),
                    ParameterSpec()
                        .name("Void mutations sigma")
                        .reference(
                            FloatSpec().member(&SimulationParameters::metaMutationVoidSigma).min(0.0f).max(1.0f).logarithmic(true).format("%.5f")),
                    ParameterSpec()
                        .name("Constructor mutations sigma")
                        .reference(
                            FloatSpec().member(&SimulationParameters::metaMutationConstructorSigma).min(0.0f).max(1.0f).logarithmic(true).format("%.5f")),
                    ParameterSpec()
                        .name("New lineage threshold")
                        .reference(
                            FloatSpec().member(&SimulationParameters::newLineageThreshold).min(0.0f).max(10000.0f).infinity(true).logarithmic(true).format("%.5f")),
                }),
            ParameterGroupSpec()
                .name("Cell type: Attacker")
                .parameters({
                    ParameterSpec()
                        .name("Energy cost")
                        .reference(FloatSpec().member(&SimulationParameters::attackerEnergyCost).min(0.0f).max(1.0f).logarithmic(true).format("%.5f"))
                        .description("Amount of energy lost by an attempted attack of a cell in form of emitted energy particles."),
                    ParameterSpec()
                        .name("Food chain color matrix")
                        .reference(FloatSpec().member(&SimulationParameters::attackerFoodChainColorMatrix).min(0.0f).max(1.0f).format("%.2f"))
                        .description(
                            "This matrix can be used to determine how well one cell can attack another cell. The color of the attacking cell correspond to the "
                            "row number and the color of the attacked cell to the column number. A value of 0 means that the attacked cell cannot be digested, "
                            "i.e. no energy can be obtained. A value of 1 means that the maximum energy can be obtained in the digestion process.\n\nExample: "
                            "If "
                            "a zero is entered in row 2 (red) and column 3 (green), it means that red cells cannot eat green cells."),
                    ParameterSpec()
                        .name("Attack strength")
                        .reference(FloatSpec().member(&SimulationParameters::attackerStrength).min(0.0f).max(0.5f).logarithmic(true))
                        .description(
                            "Indicates the portion of energy through which a successfully attacked cell is weakened. However, this energy portion can be "
                            "influenced by other factors adjustable within the attacker's simulation parameters."),
                    ParameterSpec()
                        .name("Related lineage protection")
                        .reference(FloatSpec().member(&SimulationParameters::attackerRelatedLineageProtection).min(0.0f).max(1.0f)),
                    ParameterSpec()
                        .name("Attack radius")
                        .reference(FloatSpec().member(&SimulationParameters::attackerRadius).min(0.0f).max(4.0f))
                        .description("The maximum distance over which an attacker cell can attack another cell."),
                }),
            ParameterGroupSpec()
                .name("Cell type: Digestor")
                .parameters({
                    ParameterSpec()
                        .name("Max raw energy conductivity")
                        .reference(FloatSpec().member(&SimulationParameters::maxRawEnergyConductivity).min(0.0f).max(6.0f).format("%.3f")),
                    ParameterSpec()
                        .name("Max raw energy conversion")
                        .reference(FloatSpec().member(&SimulationParameters::maxRawEnergyConversion).min(0.0f).max(1.0f).format("%.3f")),
                }),
            ParameterGroupSpec()
                .name("Cell type: Defender")
                .parameters({
                    ParameterSpec()
                        .name("Anti-attacker strength")
                        .reference(FloatSpec().member(&SimulationParameters::defenderAntiAttackerStrength).min(0.0f).max(5.0f))
                        .description(
                            "If an attacked cell is connected to defender cells or itself a defender cell the attack strength is reduced by this factor."),
                    ParameterSpec()
                        .name("Anti-injector strength")
                        .reference(FloatSpec().member(&SimulationParameters::defenderAntiInjectorStrength).min(0.0f).max(5.0f))
                        .description(
                            "If a constructor cell is attacked by an injector and connected to defender cells, the injection duration is increased by this "
                            "factor."),
                }),
            ParameterGroupSpec()
                .name("Cell type: Injector")
                .parameters({
                    ParameterSpec()
                        .name("Energy cost")
                        .reference(FloatSpec().member(&SimulationParameters::injectorEnergyCost).min(0.0f).max(1000.0f).logarithmic(true)),
                    ParameterSpec()
                        .name("Injection radius")
                        .reference(FloatSpec().member(&SimulationParameters::injectorRadius).min(0.1f).max(4.0f))
                        .description("The maximum distance over which an injector cell can infect another cell."),
                }),
            ParameterGroupSpec()
                .name("Cell type: Muscle")
                .parameters({
                    ParameterSpec()
                        .name("Energy cost")
                        .reference(FloatSpec().member(&SimulationParameters::muscleEnergyCost).min(0.0f).max(5.0f).format("%.5f").logarithmic(true))
                        .description("Amount of energy lost by a muscle action of a cell in form of emitted energy particles."),
                    ParameterSpec()
                        .name("Movement acceleration")
                        .reference(FloatSpec().member(&SimulationParameters::muscleMovementAcceleration).min(0.0f).max(10.0f).logarithmic(true))
                        .description(
                            "The maximum value by which a muscle cell can modify its velocity during activation. This parameter applies only to muscle cells "
                            "which are in movement mode."),
                    ParameterSpec()
                        .name("Crawling acceleration")
                        .reference(FloatSpec().member(&SimulationParameters::muscleCrawlingAcceleration).min(0.0f).max(10.0f).logarithmic(true))
                        .description("Amount of energy lost by a muscle action of a cell in form of emitted energy particles."),
                    ParameterSpec()
                        .name("Bending acceleration")
                        .reference(FloatSpec().member(&SimulationParameters::muscleBendingAcceleration).min(0.0f).max(10.0f).logarithmic(true))
                        .description("The maximum value by which a muscle cell can modify its velocity during a bending action. This parameter applies "
                                     "only to muscle cells which are in bending mode."),
                }),
            ParameterGroupSpec()
                .name("Cell type: Sensor")
                .parameters({
                    ParameterSpec()
                        .name("Radius")
                        .reference(FloatSpec().member(&SimulationParameters::sensorRadius).min(10.0f).max(800.0f))
                        .description("The maximum radius in which a sensor cell can detect mass concentrations."),
                }),
            ParameterGroupSpec()
                .name("Cell type: Reconnector")
                .parameters({
                    ParameterSpec()
                        .name("Radius")
                        .reference(FloatSpec().member(&SimulationParameters::reconnectorRadius).min(0.0f).max(3.0f))
                        .description("The maximum radius in which a reconnector cell can establish or destroy connections to other cells."),
                }),
            ParameterGroupSpec()
                .name("Cell type: Detonator")
                .parameters({
                    ParameterSpec()
                        .name("Blast radius")
                        .reference(FloatSpec().member(&SimulationParameters::detonatorRadius).min(0.0f).max(10.0f))
                        .description("The radius of the detonation."),
                    ParameterSpec()
                        .name("Chain explosion probability")
                        .reference(FloatSpec().member(&SimulationParameters::detonatorChainExplosionProbability).min(0.0f).max(1.0f))
                        .description(
                            "The probability that the explosion of one detonator will trigger the explosion of other detonators within the blast radius."),
                }),
            ParameterGroupSpec()
                .name("Advanced energy absorption control")
                .expertToggle(&SimulationParameters::advancedAbsorptionControlToggle)
                .parameters({
                    ParameterSpec()
                        .name("Low genome complexity penalty")
                        .reference(FloatSpec().member(&SimulationParameters::radiationAbsorptionLowNumCellsPenalty).min(0.0f).max(1.0f).format("%.2f"))
                        .description(
                            "When this parameter is increased, cells with fewer genome complexity will absorb less energy from an incoming energy particle."),
                    ParameterSpec()
                        .name("Low connection penalty")
                        .reference(FloatSpec().member(&SimulationParameters::radiationAbsorptionLowConnectionPenalty).min(0.0f).max(5.0f).format("%.1f"))
                        .description(
                            "When this parameter is increased, cells with fewer cell connections will absorb less energy from an incoming energy particle."),
                    ParameterSpec()
                        .name("High velocity penalty")
                        .reference(
                            FloatSpec()
                                .member(&SimulationParameters::radiationAbsorptionHighVelocityPenalty)
                                .min(0.0f)
                                .max(30.0f)
                                .logarithmic(true)
                                .format("%.2f"))
                        .description("When this parameter is increased, fast moving cells will absorb less energy from an incoming energy particle."),
                    ParameterSpec()
                        .name("Low velocity penalty")
                        .reference(FloatSpec().member(&SimulationParameters::radiationAbsorptionLowVelocityPenalty).min(0.0f).max(1.0f).format("%.2f"))
                        .description("When this parameter is increased, slowly moving cells will absorb less energy from an incoming energy particle."),
                }),
            ParameterGroupSpec()
                .name("Cell age limiter")
                .expertToggle(&SimulationParameters::cellAgeLimiterToggle)
                .parameters({
                    ParameterSpec()
                        .name("Maximum free cell age")
                        .reference(IntSpec().member(&SimulationParameters::freeCellMaxAge).min(1).max(1e7).logarithmic(true).infinity(true))
                        .description("The maximal age of free cells (= cells that arise from energy particles) can be set here."),
                    ParameterSpec()
                        .name("Reset age after construction")
                        .reference(BoolSpec().member(&SimulationParameters::resetCellAgeAfterActivation))
                        .description(
                            "If this option is activated, the age of the cells is reset to 0 after the construction of their cell network is completed, "
                            "i.e. when the state of the cells changes from 'Under construction' to 'Ready'. This option is particularly useful if a low "
                            "'Maximum inactive cell age' is set, as cell networks that are under construction are inactive and could die immediately after "
                            "completion if their construction takes a long time."),
                    ParameterSpec()
                        .name("Maximum age balancing")
                        .reference(IntSpec().member(&SimulationParameters::maxCellAgeBalancerInterval).min(1e3).max(1e6).logarithmic(true))
                        .description("Adjusts the maximum age at regular intervals. It increases the maximum age for the cell color where the fewest "
                                     "replicators exist. Conversely, the maximum age is decreased for the cell color with the most replicators."),
                }),
            ParameterGroupSpec()
                .name("Cell color transition rules")
                .expertToggle(&SimulationParameters::colorTransitionRulesToggle)
                .parameters({
                    ParameterSpec()
                        .name("Target color and duration")
                        .reference(ColorTransitionRulesSpec().member(&SimulationParameters::colorTransitionRules))
                        .description("Rules can be defined that describe how the colors of cells will change over time. For this purpose, a subsequent "
                                     "color can be defined for each cell color. In addition, durations must be specified that define how many time steps the "
                                     "corresponding color are kept."),
                }),
            ParameterGroupSpec()
                .name("Customize deletion mutations")
                .expertToggle(&SimulationParameters::customizeDeletionMutationsToggle)
                .parameters({
                    ParameterSpec()
                        .name("Minimum size")
                        .reference(IntSpec().member(&SimulationParameters::cellCopyMutationDeletionMinSize).min(0).max(1000).logarithmic(true))
                        .description(
                            "The minimum size of genomes (on the basis of the coded cells) is determined here that can result from delete mutations. The "
                            "default is 0."),
                }),
            ParameterGroupSpec()
                .name("External energy control")
                .expertToggle(&SimulationParameters::externalEnergyControlToggle)
                .parameters({
                    ParameterSpec()
                        .name("External energy amount")
                        .reference(
                            FloatSpec().member(&SimulationParameters::externalEnergy).min(0.0f).max(100000000.0f).format("%.0f").logarithmic(true).infinity(true))
                        .description("This parameter can be used to set the amount of energy of an external energy pool. This type of energy can then be "
                                     "transferred to all constructor cells at a certain rate (see inflow settings).\n\nWarning: Too much external energy can "
                                     "result in a "
                                     "massive production of cells and slow down or even crash the simulation."),
                    ParameterSpec()
                        .name("Inflow")
                        .reference(FloatSpec().member(&SimulationParameters::externalEnergyInflowFactor).min(0.0f).max(1.0f).format("%.5f").logarithmic(true))
                        .description(
                            "Here one can specify the fraction of energy transferred to constructor cells.\n\nFor example, a value of 0.05 means that "
                            "each time a constructor cell tries to build a new cell, 5% of the required energy is transferred for free from the external "
                            "energy "
                            "source."),
                    ParameterSpec()
                        .name("Inflow threshold")
                        .reference(
                            FloatSpec().member(&SimulationParameters::externalEnergyInflowThresholdFactor).min(0.0f).max(1.0f).format("%.5f").logarithmic(true))
                        .description(
                            "Here one can specify the fraction of energy that constructor cells must provide by themselves before constructing with external "
                            "energy inflow.\n\nFor example, a value of 0.6 means that a constructor cell needs energy amounting to at least 60% of the "
                            "energy required to build the new cell by itself. Otherwise, external energy inflow is requested."),
                    ParameterSpec()
                        .name("Inflow only for first offspring")
                        .reference(BoolSpec().member(&SimulationParameters::externalEnergyInflowOnlyForFirstOffspring))
                        .description("If activated, external energy can only be transferred to constructor cells that have not yet produced any offspring. "
                                     "This option can be used to foster the evolution of additional body parts."),
                    ParameterSpec()
                        .name("Backflow")
                        .reference(FloatSpec().member(&SimulationParameters::externalEnergyBackflowFactor).min(0.0f).max(1.0f))
                        .description("The proportion of energy that flows back from the simulation to the external energy pool. Each time a cell loses energy "
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
                        .description("Energy from the simulation can only flow back into the external energy pool as long as the amount of external energy is "
                                     "below this value."),
                }),
        });
    }
    return spec.value();
}

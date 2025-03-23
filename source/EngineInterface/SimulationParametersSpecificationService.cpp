#include "SimulationParametersSpecificationService.h"

#include <Fonts/IconsFontAwesome5.h>

#include "CellTypeStrings.h"

#define BASE_VALUE_OFFSET(X) offsetof(SimulationParameters, X)
#define ZONE_VALUE_OFFSET(X) offsetof(SimulationParametersZoneValues, X)

ParametersSpec SimulationParametersSpecificationService::createParametersSpec() const
{
    std::vector<std::pair<std::string, std::vector<ParameterSpec>>> cellTypeStrings;
    for (int i = 0; i < CellType_Count; ++i) {
        cellTypeStrings.emplace_back(std::make_pair(Const::CellTypeToStringMap.at(i), std::vector<ParameterSpec>()));
    }

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
                    .tooltip(
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
                                                                      "selected in the next parameter.\n\n" ICON_FA_CHEVRON_RIGHT
                        " Every cell function: The cells are colored according to their cell function.")
                    .type(SwitcherSpec().alternativeToParameters({
                        {std::string("Energy"), {}},
                        {std::string("Standard cell color"), {}},
                        {std::string("Mutant"), {}},
                        {std::string("Mutant and cell function"), {}},
                        {std::string("Cell state"), {}},
                        {std::string("Genome complexity"), {}},
                        {std::string("Specific cell function"),
                         {ParameterSpec()
                              .name("Highlighted cell function")
                              .valueAddress(BASE_VALUE_OFFSET(highlightedCellType))
                              .tooltip("The specific cell function type to be highlighted can be selected here.")
                              .type(SwitcherSpec().alternativeToParameters(cellTypeStrings))
                         }},
                        {std::string("Every cell function"), {}},
                    })),
                ParameterSpec()
                    .name("Cell radius")
                    .valueAddress(BASE_VALUE_OFFSET(cellRadius))
                    .tooltip("Specifies the radius of the drawn cells in unit length.")
                    .type(FloatSpec().min(0.0f).max(0.5f)),
                ParameterSpec()
                    .name("Zoom level for neural activity")
                    .valueAddress(BASE_VALUE_OFFSET(zoomLevelForNeuronVisualization))
                    .tooltip("The zoom level from which the neuronal activities become visible.")
                    .type(FloatSpec().min(0.0f).max(32.f).infinity(true)),
                ParameterSpec()
                    .name("Attack visualization")
                    .valueAddress(BASE_VALUE_OFFSET(attackVisualization))
                    .tooltip("If activated, successful attacks of attacker cells are visualized.")
                    .type(BoolSpec()),
                ParameterSpec()
                    .name("Muscle movement visualization")
                    .valueAddress(BASE_VALUE_OFFSET(muscleMovementVisualization))
                    .tooltip("If activated, the direction in which muscle cells are moving are visualized.")
                    .type(BoolSpec()),
                ParameterSpec()
                    .name("Borderless rendering")
                    .valueAddress(BASE_VALUE_OFFSET(borderlessRendering))
                    .tooltip("If activated, the simulation is rendered periodically in the view port.")
                    .type(BoolSpec()),
                ParameterSpec()
                    .name("Grid lines")
                    .valueAddress(BASE_VALUE_OFFSET(gridLines))
                    .tooltip("This option draws a suitable grid in the background depending on the zoom level.")
                    .type(BoolSpec()),
                ParameterSpec()
                    .name("Mark reference domain")
                    .valueAddress(BASE_VALUE_OFFSET(markReferenceDomain))
                    .tooltip("This option draws a suitable grid in the background depending on the zoom level.")
                    .type(BoolSpec()),
                ParameterSpec()
                    .name("Show radiation sources")
                    .valueAddress(BASE_VALUE_OFFSET(showRadiationSources))
                    .tooltip("This option draws red crosses in the center of radiation sources.")
                    .type(BoolSpec()),
            }),
        ParameterGroupSpec()
            .name("Numerics")
            .parameters({
                ParameterSpec()
                    .name("Time step size")
                    .valueAddress(BASE_VALUE_OFFSET(timestepSize))
                    .tooltip("The time duration calculated in a single simulation step. Smaller values increase the accuracy of the simulation while larger "
                             "values can lead to numerical instabilities.")
                    .type(FloatSpec().min(0.0f).max(1.0f)),
            }),
        ParameterGroupSpec()
            .name("Physics: Motion")
            .parameters({
                ParameterSpec()
                    .name("Friction")
                    .valueAddress(ZONE_VALUE_OFFSET(friction))
                    .visibleInZone(true)
                    .tooltip("This specifies the fraction of the velocity that is slowed down per time step.")
                    .type(FloatSpec().min(0.0f).max(1.0f).format("%.4f").logarithmic(true)),
            }),
    });
}

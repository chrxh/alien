
add_library(EngineInterface
    ArraySizesForGpuEntities.h
    ArraySizesForTOs.h
    CellTypeConstants.h
    Colors.h
    KernelLaunchSettings.h
    DataPointCollection.cpp
    DataPointCollection.h
    Definitions.h
    DescEditService.cpp
    DescEditService.h
    Descs.cpp
    Descs.h
    EngineConstants.h
    ExtinctLineageAccumulator.cpp
    ExtinctLineageAccumulator.h
    GenomeDesc.cpp
    GenomeDesc.h
    GenomeDescEditService.cpp
    GenomeDescEditService.h
    GenomeDescHash.h
    GenomeDescInfoService.cpp
    GenomeDescInfoService.h
    DescValidationService.cpp
    DescValidationService.h
    GeometryBuffers.cpp
    GeometryBuffers.h
    Ids.h
    InspectedEntityIds.h
    LocationHelper.cpp
    LocationHelper.h
    NeuralNetWeight.h
    NumberGenerator.cpp
    NumberGenerator.h
    StatisticsEntry.h
    ParametersEditService.cpp
    ParametersEditService.h
    ParametersFilter.h
    ParametersValidationService.cpp
    ParametersValidationService.h
    PerlinNoiseSource.h
    PreviewDesc.cpp
    PreviewDesc.h
    PreviewDescConverterService.cpp
    PreviewDescConverterService.h
    SelectionShallowData.h
    SettingsForSimulation.h
    ShallowUpdateSelectionData.h
    ShapeGenerator.h
    SimulationFacade.cpp
    SimulationFacade.h
    SimulationParameters.cpp
    SimulationParameters.h
    SimulationParametersTypes.h
    SimulationParametersUpdateConfig.h
    SimulationParametersSpecification.h
    SpaceCalculator.cpp
    SpaceCalculator.h
    SpecificationEvaluationService.cpp
    SpecificationEvaluationService.h
    SpecificationFilterService.cpp
    SpecificationFilterService.h
    StatisticsConverterService.cpp
    StatisticsConverterService.h
    StatisticsHistory.cpp
    StatisticsHistory.h)

target_link_libraries(EngineInterface Base)

target_link_libraries(EngineInterface Boost::boost)
target_link_libraries(EngineInterface glad::glad)

target_precompile_headers(EngineInterface PRIVATE
    <algorithm>
    <array>
    <cmath>
    <cstdint>
    <functional>
    <map>
    <memory>
    <numeric>
    <optional>
    <ranges>
    <set>
    <string>
    <unordered_map>
    <unordered_set>
    <variant>
    <vector>
)

if (MSVC)
    target_compile_options(EngineInterface PRIVATE "/MP")
endif()

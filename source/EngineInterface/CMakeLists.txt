
add_library(EngineInterface
    ArraySizesForGpuEntities.h
    ArraySizesForTOs.h
    Definitions.h
    ExtinctLineageAccumulator.cpp
    ExtinctLineageAccumulator.h
    GeometryBuffers.cpp
    GeometryBuffers.h
    InspectedEntityIds.h
    KernelLaunchSettings.h
    PerlinNoiseSource.h
    SelectionShallowData.h
    SettingsForSimulation.h
    ShallowUpdateSelectionData.h
    ShapeGenerator.h
    SimulationFacade.cpp
    SimulationFacade.h
    SimulationParametersUpdateConfig.h
    StatisticsConverterService.cpp
    StatisticsConverterService.h
    StatisticsEntry.h)

target_link_libraries(EngineInterface Base)
target_link_libraries(EngineInterface Data)

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

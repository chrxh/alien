
add_library(Data
    CellTypeConstants.h
    Colors.h
    CreatorService.cpp
    CreatorService.h
    DataPointCollection.cpp
    DataPointCollection.h
    Definitions.h
    DescEditService.cpp
    DescEditService.h
    Descs.cpp
    Descs.h
    DescValidationService.cpp
    DescValidationService.h
    EngineConstants.h
    GenomeDesc.cpp
    GenomeDesc.h
    GenomeDescAccessService.cpp
    GenomeDescAccessService.h
    GenomeDescEditService.cpp
    GenomeDescEditService.h
    GenomeDescHash.h
    LocationAccessService.cpp
    LocationAccessService.h
    LocationEditService.cpp
    LocationEditService.h
    MassOperationsService.cpp
    MassOperationsService.h
    MultiplierService.cpp
    MultiplierService.h
    NeuralNetWeight.h
    ObjectColoring.h
    ParametersAccessService.cpp
    ParametersAccessService.h
    ParametersFilter.h
    ParametersFilterHash.h
    ParametersValidationService.cpp
    ParametersValidationService.h
    PreviewDesc.cpp
    PreviewDesc.h
    PreviewDescConverterService.cpp
    PreviewDescConverterService.h
    RadiationStrengthService.cpp
    RadiationStrengthService.h
    SimulationParameters.cpp
    SimulationParameters.h
    SimulationParametersSpecification.h
    SimulationParametersTypes.h
    SpaceCalculator.cpp
    SpaceCalculator.h
    SpecificationEvaluationService.cpp
    SpecificationEvaluationService.h
    SpecificationFilterService.cpp
    SpecificationFilterService.h
    StatisticsHistory.cpp
    StatisticsHistory.h)

target_link_libraries(Data Base)

target_link_libraries(Data Boost::boost)

target_precompile_headers(Data PRIVATE
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
    target_compile_options(Data PRIVATE "/MP")
endif()

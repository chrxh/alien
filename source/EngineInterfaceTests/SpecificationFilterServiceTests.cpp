#include <gtest/gtest.h>

#include <EngineInterface/ParametersFilter.h>
#include <EngineInterface/SimulationParametersSpecification.h>
#include <EngineInterface/SpecificationFilterService.h>

class SpecificationFilterServiceTests : public ::testing::Test
{
protected:
    SpecificationFilterService const& _service = SpecificationFilterService::get();
};

TEST_F(SpecificationFilterServiceTests, filter_emptyFilter)
{
    // Create a simple spec with one group and one parameter
    ParametersSpec spec;
    ParameterGroupSpec group;
    group._name = "TestGroup";
    ParameterSpec param;
    param._name = "TestParameter";
    param._reference = FloatSpec();
    group._parameters.push_back(param);
    spec._groups.push_back(group);

    // Empty filter should return the same spec
    ParametersFilter filter;
    auto result = _service.filter(spec, filter);

    ASSERT_EQ(1, result._groups.size());
    EXPECT_EQ("TestGroup", result._groups[0]._name);
    ASSERT_EQ(1, result._groups[0]._parameters.size());
    EXPECT_EQ("TestParameter", result._groups[0]._parameters[0]._name);
}

TEST_F(SpecificationFilterServiceTests, filter_noMatchingParameters)
{
    // Create a spec with parameters that don't match the filter
    ParametersSpec spec;
    ParameterGroupSpec group;
    group._name = "TestGroup";
    
    ParameterSpec param1;
    param1._name = "FirstParameter";
    param1._reference = FloatSpec();
    group._parameters.push_back(param1);
    
    ParameterSpec param2;
    param2._name = "SecondParameter";
    param2._reference = IntSpec();
    group._parameters.push_back(param2);
    
    spec._groups.push_back(group);

    // Filter with text that doesn't match any parameter
    ParametersFilter filter;
    filter.containedText = "NonExistent";
    auto result = _service.filter(spec, filter);

    // No groups should remain since no parameters matched
    EXPECT_EQ(0, result._groups.size());
}

TEST_F(SpecificationFilterServiceTests, filter_matchingParameterName)
{
    // Create a spec with multiple parameters
    ParametersSpec spec;
    ParameterGroupSpec group;
    group._name = "TestGroup";
    
    ParameterSpec param1;
    param1._name = "EnergyParameter";
    param1._reference = FloatSpec();
    group._parameters.push_back(param1);
    
    ParameterSpec param2;
    param2._name = "SpeedParameter";
    param2._reference = FloatSpec();
    group._parameters.push_back(param2);
    
    spec._groups.push_back(group);

    // Filter for "Energy"
    ParametersFilter filter;
    filter.containedText = "Energy";
    auto result = _service.filter(spec, filter);

    // Should have one group with one parameter
    ASSERT_EQ(1, result._groups.size());
    EXPECT_EQ("TestGroup", result._groups[0]._name);
    ASSERT_EQ(1, result._groups[0]._parameters.size());
    EXPECT_EQ("EnergyParameter", result._groups[0]._parameters[0]._name);
}

TEST_F(SpecificationFilterServiceTests, filter_matchingGroupName_includesAllParameters)
{
    // Create a spec with a group whose name matches the filter
    ParametersSpec spec;
    ParameterGroupSpec group;
    group._name = "EnergySettings";
    
    ParameterSpec param1;
    param1._name = "MinValue";
    param1._reference = FloatSpec();
    group._parameters.push_back(param1);
    
    ParameterSpec param2;
    param2._name = "MaxValue";
    param2._reference = FloatSpec();
    group._parameters.push_back(param2);
    
    spec._groups.push_back(group);

    // Filter for "Energy" which matches the group name
    ParametersFilter filter;
    filter.containedText = "Energy";
    auto result = _service.filter(spec, filter);

    // Should have one group with ALL parameters
    ASSERT_EQ(1, result._groups.size());
    EXPECT_EQ("EnergySettings", result._groups[0]._name);
    ASSERT_EQ(2, result._groups[0]._parameters.size());
    EXPECT_EQ("MinValue", result._groups[0]._parameters[0]._name);
    EXPECT_EQ("MaxValue", result._groups[0]._parameters[1]._name);
}

TEST_F(SpecificationFilterServiceTests, filter_multipleGroups_mixedMatching)
{
    // Create a spec with multiple groups
    ParametersSpec spec;
    
    // First group - name matches
    ParameterGroupSpec group1;
    group1._name = "EnergyGroup";
    ParameterSpec param1;
    param1._name = "Param1";
    param1._reference = FloatSpec();
    group1._parameters.push_back(param1);
    spec._groups.push_back(group1);
    
    // Second group - parameter matches
    ParameterGroupSpec group2;
    group2._name = "SpeedGroup";
    ParameterSpec param2;
    param2._name = "EnergyParam";
    param2._reference = FloatSpec();
    group2._parameters.push_back(param2);
    ParameterSpec param3;
    param3._name = "OtherParam";
    param3._reference = FloatSpec();
    group2._parameters.push_back(param3);
    spec._groups.push_back(group2);
    
    // Third group - no match
    ParameterGroupSpec group3;
    group3._name = "SizeGroup";
    ParameterSpec param4;
    param4._name = "Width";
    param4._reference = FloatSpec();
    group3._parameters.push_back(param4);
    spec._groups.push_back(group3);

    // Filter for "Energy"
    ParametersFilter filter;
    filter.containedText = "Energy";
    auto result = _service.filter(spec, filter);

    // Should have two groups
    ASSERT_EQ(2, result._groups.size());
    
    // First group should have all parameters (group name matched)
    EXPECT_EQ("EnergyGroup", result._groups[0]._name);
    EXPECT_EQ(1, result._groups[0]._parameters.size());
    
    // Second group should have only matching parameter
    EXPECT_EQ("SpeedGroup", result._groups[1]._name);
    EXPECT_EQ(1, result._groups[1]._parameters.size());
    EXPECT_EQ("EnergyParam", result._groups[1]._parameters[0]._name);
}

TEST_F(SpecificationFilterServiceTests, filter_alternativeSpec_matchingNestedParameter)
{
    // Create a spec with AlternativeSpec
    ParametersSpec spec;
    ParameterGroupSpec group;
    group._name = "TestGroup";
    
    ParameterSpec param;
    param._name = "ModeSelection";
    
    AlternativeSpec altSpec;
    
    // First alternative with matching parameter
    ParameterSpec altParam1;
    altParam1._name = "EnergyMode";
    altParam1._reference = FloatSpec();
    altSpec._alternatives.push_back({"Mode1", {altParam1}});
    
    // Second alternative without matching parameter
    ParameterSpec altParam2;
    altParam2._name = "SpeedMode";
    altParam2._reference = FloatSpec();
    altSpec._alternatives.push_back({"Mode2", {altParam2}});
    
    param._reference = altSpec;
    group._parameters.push_back(param);
    spec._groups.push_back(group);

    // Filter for "Energy"
    ParametersFilter filter;
    filter.containedText = "Energy";
    auto result = _service.filter(spec, filter);

    // Should have the parameter because nested parameter matches
    ASSERT_EQ(1, result._groups.size());
    ASSERT_EQ(1, result._groups[0]._parameters.size());
    EXPECT_EQ("ModeSelection", result._groups[0]._parameters[0]._name);
    EXPECT_TRUE(result._groups[0]._parameters[0]._visible);
}

TEST_F(SpecificationFilterServiceTests, filter_caseInsensitiveSubstring)
{
    // Create a simple spec
    ParametersSpec spec;
    ParameterGroupSpec group;
    group._name = "TestGroup";
    
    ParameterSpec param;
    param._name = "cell_max_energy";
    param._reference = FloatSpec();
    group._parameters.push_back(param);
    
    spec._groups.push_back(group);

    // Filter for substring
    ParametersFilter filter;
    filter.containedText = "energy";
    auto result = _service.filter(spec, filter);

    // Should match
    ASSERT_EQ(1, result._groups.size());
    ASSERT_EQ(1, result._groups[0]._parameters.size());
    EXPECT_EQ("cell_max_energy", result._groups[0]._parameters[0]._name);
}

TEST_F(SpecificationFilterServiceTests, filter_preservesGroupMetadata)
{
    // Create a spec with group metadata
    ParametersSpec spec;
    ParameterGroupSpec group;
    group._name = "EnergyGroup";
    group._description = "Energy related settings";
    
    ParameterSpec param;
    param._name = "MaxEnergy";
    param._reference = FloatSpec();
    group._parameters.push_back(param);
    
    spec._groups.push_back(group);

    // Filter that matches group name
    ParametersFilter filter;
    filter.containedText = "Energy";
    auto result = _service.filter(spec, filter);

    // Should preserve group metadata
    ASSERT_EQ(1, result._groups.size());
    EXPECT_EQ("EnergyGroup", result._groups[0]._name);
    EXPECT_EQ("Energy related settings", result._groups[0]._description.value());
}

TEST_F(SpecificationFilterServiceTests, filter_multipleMatchingParameters)
{
    // Create a spec with multiple matching parameters
    ParametersSpec spec;
    ParameterGroupSpec group;
    group._name = "TestGroup";
    
    ParameterSpec param1;
    param1._name = "cell_min_energy";
    param1._reference = FloatSpec();
    group._parameters.push_back(param1);
    
    ParameterSpec param2;
    param2._name = "cell_max_energy";
    param2._reference = FloatSpec();
    group._parameters.push_back(param2);
    
    ParameterSpec param3;
    param3._name = "cell_velocity";
    param3._reference = FloatSpec();
    group._parameters.push_back(param3);
    
    spec._groups.push_back(group);

    // Filter for "energy"
    ParametersFilter filter;
    filter.containedText = "energy";
    auto result = _service.filter(spec, filter);

    // Should have two matching parameters
    ASSERT_EQ(1, result._groups.size());
    ASSERT_EQ(2, result._groups[0]._parameters.size());
    EXPECT_EQ("cell_min_energy", result._groups[0]._parameters[0]._name);
    EXPECT_EQ("cell_max_energy", result._groups[0]._parameters[1]._name);
}

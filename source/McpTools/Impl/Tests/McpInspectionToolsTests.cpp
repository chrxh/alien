#include <string>
#include <vector>

#include <gtest/gtest.h>

#include <boost/json.hpp>

#include <Data/Descs.h>

#include <McpTools/Impl/McpInspectionTools.h>

class McpInspectionToolsTests : public ::testing::Test
{
protected:
    // Inspected data as the engine provides it for cell 2: it also contains the connected cells 1 and 3, where cell 3 belongs to another creature
    ContentDesc createInspectedData() const
    {
        auto result =
            ContentDesc()
                .addCreature({ObjectDesc().id(1).pos({10.0f, 10.0f}), ObjectDesc().id(2).pos({11.0f, 10.0f})}, CreatureDesc().id(10), GenomeDesc().id(100))
                .addCreature({ObjectDesc().id(3).pos({12.0f, 10.0f})}, CreatureDesc().id(11), GenomeDesc().id(101));
        result.addConnection(1, 2);
        result.addConnection(2, 3);
        result._energies.emplace_back(EnergyDesc().id(4));
        return result;
    }

    std::vector<std::string> getObjectIds(boost::json::object const& inspectionResult) const
    {
        std::vector<std::string> result;
        for (auto const& entry : inspectionResult.at("objects").as_array()) {
            result.emplace_back(entry.as_object().at("object").as_object().at("id").as_string());
        }
        return result;
    }

    McpInspectionTools const& _tools = McpInspectionTools::get();
};

TEST_F(McpInspectionToolsTests, inspectionResultContainsOnlyRequestedObjects)
{
    auto result = _tools.createInspectionResult(createInspectedData(), {2, 4}, true, true);

    EXPECT_EQ(std::vector<std::string>{"2"}, getObjectIds(result));
    auto const& object = result.at("objects").as_array().at(0).as_object().at("object").as_object();
    EXPECT_EQ(2, object.at("connections").as_array().size());
    EXPECT_EQ(1, result.at("energy_particles").as_array().size());
    auto const& genomes = result.at("genomes_by_id").as_object();
    EXPECT_EQ(1, genomes.size());
    EXPECT_TRUE(genomes.contains("100"));
    EXPECT_FALSE(result.contains("not_found_ids"));
}

TEST_F(McpInspectionToolsTests, inspectionResultFollowsOrderOfIdsAndReportsUnknownIdsOnce)
{
    auto result = _tools.createInspectionResult(createInspectedData(), {3, 1, 3, 99, 99}, false, true);

    EXPECT_EQ((std::vector<std::string>{"3", "1"}), getObjectIds(result));
    EXPECT_FALSE(result.contains("genomes_by_id"));
    auto const& missingIds = result.at("not_found_ids").as_array();
    ASSERT_EQ(1, missingIds.size());
    EXPECT_EQ("99", std::string(missingIds.at(0).as_string()));
}

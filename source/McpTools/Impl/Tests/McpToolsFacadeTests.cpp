#include <set>
#include <string>

#include <gtest/gtest.h>

#include <McpTools/Impl/McpToolsFacadeImpl.h>

class McpToolsFacadeTests : public ::testing::Test
{
public:
    McpToolsFacadeTests() { _facade.setup(_context); }

protected:
    class Context : public McpToolContext
    {
        RealVector2D getVisibleAreaCenter() const override { return {}; }
        RealVector2D getVisibleAreaSize() const override { return {}; }
        float getZoomFactor() const override { return 1.0f; }
        void setVisibleArea(RealVector2D const&, float) override {}
        void createSimulation(std::string const&, IntVector2D const&) override {}
        void applySimulation(SimulationDesc const&) override {}
        std::string createPicture(IntVector2D const&, McpPictureFormat) override { return {}; }
        std::optional<std::string> createSimulationPreviewJpg() override { return std::nullopt; }
    };

    Context _context;
    _McpToolsFacadeImpl _facade;
};

TEST_F(McpToolsFacadeTests, toolNamesAreUnique)
{
    std::set<std::string> toolNames;
    for (auto const& toolGroup : _facade.getToolGroups()) {
        EXPECT_FALSE(toolGroup.name.empty());
        for (auto const& toolName : toolGroup.toolNames) {
            EXPECT_TRUE(toolNames.insert(toolName).second) << toolName;
        }
    }
    EXPECT_TRUE(toolNames.contains("take_screenshot"));
    EXPECT_TRUE(toolNames.contains("step_forward"));
}

TEST_F(McpToolsFacadeTests, serverIsStoppedOnShutdown)
{
    auto constexpr AnyFreePort = 0;

    EXPECT_FALSE(_facade.isServerRunning());
    ASSERT_TRUE(_facade.startServer(AnyFreePort));
    EXPECT_TRUE(_facade.isServerRunning());

    _facade.shutdown();
    EXPECT_FALSE(_facade.isServerRunning());
}

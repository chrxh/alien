#include <gtest/gtest.h>

#include "Base/NumberGenerator.h"
#include "Network/NetworkDataTO.h"
#include "Network/BrowserDataService.h"

class BrowserDataServiceTests : public ::testing::Test
{
public:
    BrowserDataServiceTests()
    {}
    ~BrowserDataServiceTests() = default;
};

TEST_F(BrowserDataServiceTests, singleLeaf)
{
    std::vector<NetworkDataTO> inputTOs;
    auto inputTO = std::make_shared<_NetworkDataTO>();
    inputTO->simName = "test";
    inputTOs.emplace_back(inputTO);

    auto outputTOs = BrowserDataService::createBrowserData(inputTOs);

    ASSERT_EQ(1, outputTOs.size());
}

#include <gtest/gtest.h>

#include "Base/NumberGenerator.h"
#include "Network/NetworkDataTO.h"
#include "Network/BrowserDataService.h"

#include "Network/BrowserDataTO.h"

class BrowserDataServiceTests : public ::testing::Test
{
public:
    BrowserDataServiceTests()
    {}
    ~BrowserDataServiceTests() = default;
};

TEST_F(BrowserDataServiceTests, nameWithoutFolder)
{
    std::vector<NetworkDataTO> inputTOs;
    auto inputTO = std::make_shared<_NetworkDataTO>();
    inputTO->simName = "test";
    inputTOs.emplace_back(inputTO);

    auto outputTOs = BrowserDataService::createBrowserData(inputTOs);

    ASSERT_EQ(1, outputTOs.size());
}

TEST_F(BrowserDataServiceTests, nameWithFolder)
{
    std::vector<NetworkDataTO> inputTOs;
    auto inputTO = std::make_shared<_NetworkDataTO>();
    inputTO->simName = "folder/test";
    inputTOs.emplace_back(inputTO);

    auto outputTOs = BrowserDataService::createBrowserData(inputTOs);

    ASSERT_EQ(2, outputTOs.size());
    {
        auto outputTO = outputTOs.front();
        EXPECT_FALSE(outputTO->isLeaf());
        EXPECT_EQ(1, outputTO->location.size());
        EXPECT_EQ(std::string("folder"), outputTO->location.front());
    }
    {
        auto outputTO = outputTOs.back();
        EXPECT_TRUE(outputTO->isLeaf());
        EXPECT_EQ(1, outputTO->location.size());
        EXPECT_EQ(std::string("folder"), outputTO->location.front());
    }
}

TEST_F(BrowserDataServiceTests, nameWithTwoFolders)
{
    std::vector<NetworkDataTO> inputTOs;
    auto inputTO = std::make_shared<_NetworkDataTO>();
    inputTO->simName = "folder1/folder2/test";
    inputTOs.emplace_back(inputTO);

    auto outputTOs = BrowserDataService::createBrowserData(inputTOs);

    ASSERT_EQ(3, outputTOs.size());
    {
        auto outputTO = outputTOs.at(0);
        EXPECT_FALSE(outputTO->isLeaf());
        EXPECT_EQ(1, outputTO->location.size());
        EXPECT_EQ(std::string("folder1"), outputTO->location.at(0));
    }
    {
        auto outputTO = outputTOs.at(1);
        EXPECT_FALSE(outputTO->isLeaf());
        EXPECT_EQ(2, outputTO->location.size());
        EXPECT_EQ(std::string("folder1"), outputTO->location.at(0));
        EXPECT_EQ(std::string("folder2"), outputTO->location.at(1));
    }
    {
        auto outputTO = outputTOs.at(2);
        EXPECT_TRUE(outputTO->isLeaf());
        EXPECT_EQ(2, outputTO->location.size());
        EXPECT_EQ(std::string("folder1"), outputTO->location.at(0));
        EXPECT_EQ(std::string("folder2"), outputTO->location.at(1));
    }
}

#include <gtest/gtest.h>

#include "Base/NumberGenerator.h"
#include "Network/NetworkResourceRawTO.h"
#include "Network/NetworkResourceService.h"

#include "Network/NetworkResourceTreeTO.h"

class BrowserDataServiceTests : public ::testing::Test
{
public:
    BrowserDataServiceTests()
    {}
    ~BrowserDataServiceTests() = default;
};

TEST_F(BrowserDataServiceTests, nameWithoutFolder)
{
    std::vector<NetworkResourceRawTO> inputTOs;
    auto inputTO = std::make_shared<_NetworkResourceRawTO>();
    inputTO->simName = "test";
    inputTOs.emplace_back(inputTO);

    auto outputTOs = NetworkResourceService::createBrowserData(inputTOs);

    ASSERT_EQ(1, outputTOs.size());
}

TEST_F(BrowserDataServiceTests, nameWithFolder)
{
    std::vector<NetworkResourceRawTO> inputTOs;
    auto inputTO = std::make_shared<_NetworkResourceRawTO>();
    inputTO->simName = "folder/test";
    inputTOs.emplace_back(inputTO);

    auto outputTOs = NetworkResourceService::createBrowserData(inputTOs);

    ASSERT_EQ(2, outputTOs.size());
    {
        auto outputTO = outputTOs.front();
        EXPECT_FALSE(outputTO->isLeaf());
        EXPECT_EQ(1, outputTO->folderNames.size());
        EXPECT_EQ(std::string("folder"), outputTO->folderNames.front());
    }
    {
        auto outputTO = outputTOs.back();
        EXPECT_TRUE(outputTO->isLeaf());
        EXPECT_EQ(1, outputTO->folderNames.size());
        EXPECT_EQ(std::string("folder"), outputTO->folderNames.front());
        EXPECT_EQ(std::string("test"), outputTO->getLeaf().simName);
    }
}

TEST_F(BrowserDataServiceTests, nameWithTwoFolders)
{
    std::vector<NetworkResourceRawTO> inputTOs;
    auto inputTO = std::make_shared<_NetworkResourceRawTO>();
    inputTO->simName = "folder1/folder2/test";
    inputTOs.emplace_back(inputTO);

    auto outputTOs = NetworkResourceService::createBrowserData(inputTOs);

    ASSERT_EQ(3, outputTOs.size());
    {
        auto outputTO = outputTOs.at(0);
        EXPECT_FALSE(outputTO->isLeaf());
        EXPECT_EQ(1, outputTO->folderNames.size());
        EXPECT_EQ(std::string("folder1"), outputTO->folderNames.at(0));
    }
    {
        auto outputTO = outputTOs.at(1);
        EXPECT_FALSE(outputTO->isLeaf());
        EXPECT_EQ(2, outputTO->folderNames.size());
        EXPECT_EQ(std::string("folder1"), outputTO->folderNames.at(0));
        EXPECT_EQ(std::string("folder2"), outputTO->folderNames.at(1));
    }
    {
        auto outputTO = outputTOs.at(2);
        EXPECT_TRUE(outputTO->isLeaf());
        EXPECT_EQ(2, outputTO->folderNames.size());
        EXPECT_EQ(std::string("folder1"), outputTO->folderNames.at(0));
        EXPECT_EQ(std::string("folder2"), outputTO->folderNames.at(1));
        EXPECT_EQ(std::string("test"), outputTO->getLeaf().simName);
    }
}


TEST_F(BrowserDataServiceTests, twoNamesWithTwoFolders)
{
    std::vector<NetworkResourceRawTO> inputTOs;
    {
        auto inputTO = std::make_shared<_NetworkResourceRawTO>();
        inputTO->simName = "A/B/C";
        inputTOs.emplace_back(inputTO);
    }
    {
        auto inputTO = std::make_shared<_NetworkResourceRawTO>();
        inputTO->simName = "X/Y/Z";
        inputTOs.emplace_back(inputTO);
    }

    auto outputTOs = NetworkResourceService::createBrowserData(inputTOs);

    ASSERT_EQ(6, outputTOs.size());
    {
        auto outputTO = outputTOs.at(0);
        EXPECT_FALSE(outputTO->isLeaf());
        EXPECT_EQ(1, outputTO->folderNames.size());
        EXPECT_EQ(std::string("A"), outputTO->folderNames.at(0));
    }
    {
        auto outputTO = outputTOs.at(1);
        EXPECT_FALSE(outputTO->isLeaf());
        EXPECT_EQ(2, outputTO->folderNames.size());
        EXPECT_EQ(std::string("A"), outputTO->folderNames.at(0));
        EXPECT_EQ(std::string("B"), outputTO->folderNames.at(1));
    }
    {
        auto outputTO = outputTOs.at(2);
        EXPECT_TRUE(outputTO->isLeaf());
        EXPECT_EQ(2, outputTO->folderNames.size());
        EXPECT_EQ(std::string("A"), outputTO->folderNames.at(0));
        EXPECT_EQ(std::string("B"), outputTO->folderNames.at(1));
        EXPECT_EQ(std::string("C"), outputTO->getLeaf().simName);
    }
    {
        auto outputTO = outputTOs.at(3);
        EXPECT_FALSE(outputTO->isLeaf());
        EXPECT_EQ(1, outputTO->folderNames.size());
        EXPECT_EQ(std::string("X"), outputTO->folderNames.at(0));
    }
    {
        auto outputTO = outputTOs.at(4);
        EXPECT_FALSE(outputTO->isLeaf());
        EXPECT_EQ(2, outputTO->folderNames.size());
        EXPECT_EQ(std::string("X"), outputTO->folderNames.at(0));
        EXPECT_EQ(std::string("Y"), outputTO->folderNames.at(1));
    }
    {
        auto outputTO = outputTOs.at(5);
        EXPECT_TRUE(outputTO->isLeaf());
        EXPECT_EQ(2, outputTO->folderNames.size());
        EXPECT_EQ(std::string("X"), outputTO->folderNames.at(0));
        EXPECT_EQ(std::string("Y"), outputTO->folderNames.at(1));
        EXPECT_EQ(std::string("Z"), outputTO->getLeaf().simName);
    }
}

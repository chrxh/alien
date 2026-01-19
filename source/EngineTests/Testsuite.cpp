#include <cstring>

#include <gtest/gtest.h>

#include <Base/GlobalSettings.h>

#include "IntegrationTestFramework.h"

namespace
{
    bool hasArgument(int argc, char** argv, const char* arg)
    {
        for (int i = 1; i < argc; ++i) {
            if (strcmp(argv[i], arg) == 0) {
                return true;
            }
        }
        return false;
    }
}

class TestCleanupListener : public ::testing::EmptyTestEventListener
{
    void OnTestProgramEnd(const ::testing::UnitTest& /*unit_test*/) override { IntegrationTestFramework::cleanupGlobalContext(); }
};

int main(int argc, char** argv)
{
    auto inDebugMode = hasArgument(argc, argv, "-d");
    GlobalSettings::get().setDebugMode(inDebugMode);

    ::testing::InitGoogleTest(&argc, argv);

    ::testing::TestEventListeners& listeners = ::testing::UnitTest::GetInstance()->listeners();
    listeners.Append(new TestCleanupListener);

    return RUN_ALL_TESTS();
}

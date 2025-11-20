#include <gtest/gtest.h>

#include "IntegrationTestFramework.h"

class TestCleanupListener : public ::testing::EmptyTestEventListener
{
    void OnTestProgramEnd(const ::testing::UnitTest& /*unit_test*/) override
    {
        IntegrationTestFramework::cleanupGlobalContext();
    }
};

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    
    ::testing::TestEventListeners& listeners = ::testing::UnitTest::GetInstance()->listeners();
    listeners.Append(new TestCleanupListener);
    
    return RUN_ALL_TESTS();
}

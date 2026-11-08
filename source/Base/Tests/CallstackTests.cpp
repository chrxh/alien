#include <gtest/gtest.h>

#ifdef _WIN32
#include <Windows.h>
#endif

#include <Base/AlienExceptions.h>
#include <Base/Callstack.h>

class CallstackTests : public ::testing::Test
{};

TEST_F(CallstackTests, capture)
{
    auto callstack = Callstack::capture();
    EXPECT_FALSE(callstack.empty());
    EXPECT_NE("unavailable", callstack);
}

TEST_F(CallstackTests, exceptionCallstack)
{
    auto callstack = AlienException("test").getCallstack();
    EXPECT_FALSE(callstack.empty());
    EXPECT_NE("unavailable", callstack);
}

#ifdef _WIN32
TEST_F(CallstackTests, captureFromContext)
{
    CONTEXT context;
    RtlCaptureContext(&context);
    auto callstack = Callstack::capture(context);
    EXPECT_FALSE(callstack.empty());
    EXPECT_NE("unavailable", callstack);
}
#endif

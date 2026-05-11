#include <filesystem>
#include <stdexcept>
#include <string>

#include <gtest/gtest.h>

#include <Base/Macros.h>

TEST(MacrosTests, checkExceptionContainsCallstack)
{
    try {
        CHECK(false);
        FAIL();
    } catch (std::runtime_error const& exception) {
        auto const message = std::string(exception.what());
        EXPECT_NE(message.find("check failed"), std::string::npos);
        EXPECT_NE(message.find("false"), std::string::npos);
        EXPECT_NE(message.find("Callstack:"), std::string::npos);
        EXPECT_NE(message.find(std::filesystem::path(__FILE__).filename().string()), std::string::npos);
    }
}

#include <filesystem>
#include <fstream>

#include <gtest/gtest.h>

#include <Base/AlienExceptions.h>

#include <Video/VideoWriter.h>

class VideoWriterTests : public ::testing::Test
{
protected:
    std::filesystem::path createFilename(std::string const& name) const
    {
        auto result = std::filesystem::temp_directory_path() / name;
        std::filesystem::remove(result);
        return result;
    }

    RgbImage createImage(IntVector2D const& resolution, int frame) const
    {
        RgbImage result{.width = resolution.x, .height = resolution.y};
        for (int y = 0; y < resolution.y; ++y) {
            for (int x = 0; x < resolution.x; ++x) {
                result.pixels.emplace_back(static_cast<uint8_t>(x * 4 + frame));
                result.pixels.emplace_back(static_cast<uint8_t>(y * 4));
                result.pixels.emplace_back(static_cast<uint8_t>(128));
            }
        }
        return result;
    }
};

TEST_F(VideoWriterTests, oddResolution_isRejected)
{
    EXPECT_THROW(_VideoWriter(createFilename("alien-video-writer-test-odd.mp4"), {63, 48}, 30, 1000000), AlienException);
}

// Media Foundation is part of Windows, other systems need an installed ffmpeg
#ifdef _WIN32
TEST_F(VideoWriterTests, writesMp4File)
{
    auto filename = createFilename("alien-video-writer-test.mp4");
    IntVector2D resolution{64, 48};
    {
        _VideoWriter writer(filename, resolution, 30, 1000000);
        for (int frame = 0; frame < 30; ++frame) {
            writer.addFrame(createImage(resolution, frame));
        }
        writer.finish();
        EXPECT_FALSE(writer.getError().has_value());
    }

    ASSERT_TRUE(std::filesystem::exists(filename));
    std::ifstream stream(filename, std::ios::binary);
    std::string header(8, '\0');
    stream.read(header.data(), header.size());
    EXPECT_EQ("ftyp", header.substr(4, 4));
    stream.close();

    std::filesystem::remove(filename);
}
#endif

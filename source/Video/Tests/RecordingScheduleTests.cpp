#include <cmath>

#include <gtest/gtest.h>

#include <Video/RecordingSchedule.h>

class RecordingScheduleTests : public ::testing::Test
{
protected:
    CameraPath createPath(float duration, float startTimeLapse, float endTimeLapse) const
    {
        return CameraPath(
            {
                {.center = {0.0f, 0.0f}, .timeLapse = startTimeLapse},
                {.center = {100.0f, 0.0f}, .duration = duration, .timeLapse = endTimeLapse},
            },
            0);
    }
};

TEST_F(RecordingScheduleTests, framesCoverWholePath)
{
    RecordingSchedule schedule(createPath(10.0f, 1.0f, 1.0f), 30, 60.0);

    EXPECT_EQ(301, schedule.getNumFrames());
    EXPECT_FLOAT_EQ(0.0f, schedule.getCameraState(0).center.x);
    EXPECT_FLOAT_EQ(50.0f, schedule.getCameraState(150).center.x);
    EXPECT_FLOAT_EQ(100.0f, schedule.getCameraState(300).center.x);
}

TEST_F(RecordingScheduleTests, normalSpeed)
{
    RecordingSchedule schedule(createPath(10.0f, 1.0f, 1.0f), 30, 60.0);

    for (int frame = 0; frame < 300; ++frame) {
        EXPECT_EQ(2u, schedule.getTimestepsAfterFrame(frame));
    }
    EXPECT_EQ(0u, schedule.getTimestepsAfterFrame(300));
    EXPECT_EQ(600u, schedule.getTotalTimesteps());
}

TEST_F(RecordingScheduleTests, timeLapse_multipliesTimesteps)
{
    RecordingSchedule schedule(createPath(10.0f, 5.0f, 5.0f), 30, 60.0);

    EXPECT_EQ(10u, schedule.getTimestepsAfterFrame(0));
    EXPECT_EQ(3000u, schedule.getTotalTimesteps());
}

TEST_F(RecordingScheduleTests, fractionsOfTimesteps_areCarriedOver)
{
    RecordingSchedule schedule(createPath(3.0f, 1.0f, 1.0f), 30, 20.0);

    uint64_t sum = 0;
    for (int frame = 0; frame < schedule.getNumFrames(); ++frame) {
        auto timesteps = schedule.getTimestepsAfterFrame(frame);
        EXPECT_LE(timesteps, 1u);
        sum += timesteps;
    }
    EXPECT_EQ(60u, sum);
    EXPECT_EQ(60u, schedule.getTotalTimesteps());
}

TEST_F(RecordingScheduleTests, increasingTimeLapse_speedsUpSimulation)
{
    RecordingSchedule schedule(createPath(10.0f, 1.0f, 10.0f), 30, 60.0);

    EXPECT_LT(schedule.getTimestepsAfterFrame(0), schedule.getTimestepsAfterFrame(150));
    EXPECT_LT(schedule.getTimestepsAfterFrame(150), schedule.getTimestepsAfterFrame(299));

    // The geometric interpolation of the time-lapse from 1 to 10 integrates to 9 / ln(10) times the duration
    auto expectedTotal = 60.0 * 10.0 * 9.0 / std::log(10.0);
    EXPECT_NEAR(expectedTotal, toDouble(schedule.getTotalTimesteps()), expectedTotal * 0.001);
}
